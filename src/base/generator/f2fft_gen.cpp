// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "f2fft_gen.hpp"
#include "bbfft/bad_configuration.hpp"
#include "bbfft/prime_factorization.hpp"
#include "math.hpp"
#include "mixed_radix_fft.hpp"
#include "root_of_unity.hpp"

#include "clir/attr_defs.hpp"
#include "clir/builder.hpp"
#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/visitor/codegen_opencl.hpp"
#include "clir/visitor/unique_names.hpp"
#include "clir/visitor/unsafe_simplification.hpp"

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace clir;

namespace bbfft {

template <typename BodyBuilder>
void parallel_2d_loop(block_builder &bb, expr n_local, std::size_t Nb, std::size_t N,
                      BodyBuilder &&builder, std::string var_name = "j1j2") {
    if (N == Nb) {
        bb.add(block_builder{}.body(builder(n_local)).get_product());
    } else {
        auto loop_var = var(var_name);
        bb.add(for_loop_builder(declaration_assignment(generic_short(), loop_var, n_local),
                                loop_var < static_cast<short>(N), add_into(loop_var, Nb))
                   .body(builder(loop_var))
                   .get_product());
    }
}

void f2fft_gen::generate(std::ostream &os, factor2_slm_configuration const &cfg,
                         std::string_view name) const {
    if (cfg.factorization.size() < 2) {
        throw bad_configuration("At least 2 factors are required.");
    }

    auto in = var("in");
    auto out = var("out");
    auto twiddle = var("twiddle");
    auto K = var("K");
    auto user_data = var("user_data");

    auto const tw2N_offset = [](std::vector<int> const &factorization) {
        int N = factorization[0] * factorization[1];
        int offset = N;
        for (std::size_t k = 2; k < factorization.size(); ++k) {
            N *= factorization[k];
            offset += N;
        }
        return offset;
    };

    auto fph = precision_helper{cfg.fp};
    auto in_ty = fph.type(p_.in_components, address_space::global_t);
    auto out_ty = fph.type(p_.out_components, address_space::global_t);
    auto slm_ty = fph.type(2, address_space::local_t);

    auto fft_inplace = &generate_fft::pair_optimization_inplace;

    auto fb = kernel_builder{name.empty() ? cfg.identifier() : std::string(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(pointer_to(fph.type(2, address_space::constant_t)), twiddle);
    fb.argument(generic_ulong(), K);
    if (cfg.load_function != nullptr || cfg.store_function != nullptr) {
        fb.argument(pointer_to(data_type(builtin_type::void_t, address_space::global_t)),
                    user_data);
    }
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Nb),
                                      static_cast<int>(cfg.Kb)));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));

    std::shared_ptr<tensor_accessor> in_acc, out_acc;
    if (cfg.load_function) {
        in_acc =
            std::make_shared<callback_accessor>(in, in_ty, cfg.load_function, nullptr, user_data);
    } else {
        in_acc = std::make_shared<array_accessor>(in, in_ty);
    }
    if (cfg.store_function) {
        out_acc = std::make_shared<callback_accessor>(out, out_ty, nullptr, cfg.store_function,
                                                      user_data);
    } else {
        out_acc = std::make_shared<array_accessor>(out, out_ty);
    }

    fb.body([&](block_builder &bb) {
        auto X1 = bb.declare(array_of(slm_ty, cfg.Kb * p_.N_slm * cfg.Mb), "X1");
        auto kk = bb.declare_assign(generic_size(), "kk", get_global_id(2));
        auto mm = bb.declare_assign(generic_size(), "mm", get_global_id(0));
        auto n_local = bb.declare_assign(generic_size(), "n_local", get_local_id(1));

        auto X1_view = tensor_view(std::make_shared<array_accessor>(X1, slm_ty),
                                   std::array<expr, 3u>{cfg.Mb, p_.N_slm, cfg.Kb})
                           .subview(bb, get_local_id(0), slice{}, get_local_id(2));

        auto in_view =
            tensor_view(in_acc, {cfg.M, p_.N_in, K},
                        std::array<expr, 3u>{cfg.istride[0], cfg.istride[1], cfg.istride[2]});
        auto out_view =
            tensor_view(out_acc, {cfg.M, p_.N_out, K},
                        std::array<expr, 3u>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]});

        auto compute_stage = [&](block_builder &bb, int const f, int const J1, int const Nf,
                                 int const J2, expr j1, expr j2, int const tw_offset) {
            auto xy_ty_Nf = data_type(array_of(fph.type(2), Nf));
            auto x = bb.declare(xy_ty_Nf, "x");
            auto x_acc = std::make_shared<array_accessor>(x, xy_ty_Nf);
            auto x_view = tensor_view(x_acc, std::array<expr, 1u>{Nf});

            if (f == static_cast<int>(cfg.factorization.size() - 1)) {
                load(bb, copy_params{cfg, fph, in_view, X1_view, x_view, x_acc, mm, kk, K, j1});
            } else {
                auto X1_view_1d = X1_view.reshaped_mode(0, std::array<expr, 3u>{J1, Nf, J2})
                                      .subview(bb, j1, slice{}, j2);
                copy_N_block(bb, X1_view_1d, x_view, Nf, Nf);
            }

            auto factor = trial_division(Nf);
            expr tw_j1 = nullptr;
            if (f > 0) {
                tw_j1 = bb.declare_assign(pointer_to(fph.type(2, address_space::constant_t)),
                                          "tw_j1", twiddle + tw_offset + j1 * Nf);
            }
            fft_inplace(bb, cfg.fp, cfg.direction, factor, x, tw_j1);

            auto X1_view_1d = X1_view.reshaped_mode(0, std::array<expr, 3u>{J1, Nf, J2})
                                  .subview(bb, j1, slice{}, j2);
            copy_N_block_with_permutation(bb, x_view, X1_view_1d, Nf, unscrambler(factor));
        };

        preprocess(bb, prepost_params{cfg, fph, in_view, X1_view, mm, n_local, kk, K,
                                      twiddle + tw2N_offset(cfg.factorization)});

        int J1 = product(cfg.factorization.begin(), cfg.factorization.end(), 1);
        int J2 = 1;
        int tw_offset = 0;
        int const L = cfg.factorization.size();
        for (int f = L - 1; f >= 0; --f) {
            int const Nf = cfg.factorization[f];
            J1 /= Nf;

            parallel_2d_loop(bb, n_local, cfg.Nb, J1 * J2, [&](expr &loop_var) {
                return [&](block_builder &bb) {
                    if (J2 == 1) {
                        compute_stage(bb, f, J1, Nf, J2, loop_var, 0, tw_offset);
                    } else if (J1 == 1) {
                        compute_stage(bb, f, J1, Nf, J2, 0, loop_var, tw_offset);
                    } else {
                        auto j1 = bb.declare_assign(generic_short(), "j1", loop_var % J1);
                        auto j2 = bb.declare_assign(generic_short(), "j2", loop_var / J1);
                        compute_stage(bb, f, J1, Nf, J2, j1, j2, tw_offset);
                    }
                };
            });

            J2 *= Nf;
            tw_offset += J1 * Nf;
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        }

        auto unscramble =
            unscrambler<clir::expr>(cfg.factorization.begin(), cfg.factorization.end());
        unscramble.in0toN(true);
        postprocess(bb, prepost_params{cfg, fph, out_view, X1_view, mm, n_local, kk, K,
                                       twiddle + tw2N_offset(cfg.factorization),
                                       std::move(unscramble)});
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

void f2fft_gen::global_load(block_builder &bb, copy_params const &cp, expr k,
                            tensor_view<3u> const &view) const {
    auto const &f = cp.cfg.factorization;
    auto const J1 = product(f.begin(), f.begin() + f.size() - 1, 1);
    auto const Nf = f.back();
    bb.add(if_selection_builder(cp.mm < cp.cfg.M && k < cp.K)
               .then([&](block_builder &bb) {
                   auto view_sub = view.reshaped_mode(1, std::array<expr, 2u>{J1, Nf})
                                       .subview(bb, cp.mm, cp.j1, slice{}, k);
                   copy_N_block(bb, view_sub, cp.x_view, Nf, Nf);
               })
               .get_product());
}

void f2fft_gen_c2c::load(block_builder &bb, copy_params cp) const {
    global_load(bb, cp, cp.kk, cp.view);
}

void f2fft_gen_c2c::postprocess(block_builder &bb, prepost_params pp) const {
    auto const &f = pp.cfg.factorization;
    auto const Nf = f.front();
    auto const J2 = product(f.begin() + 1, f.end(), 1);
    auto unscramble = unscrambler<clir::expr>(f.begin() + 1, f.end());
    unscramble.in0toN(true);
    parallel_2d_loop(
        bb, pp.n_local, pp.cfg.Nb, J2,
        [&](expr &j2) {
            return [&](block_builder &bb) {
                auto X1_view_1d = pp.X1_view.reshaped_mode(0, std::array<expr, 2u>{Nf, J2})
                                      .subview(bb, slice{}, unscramble(j2));
                bb.add(if_selection_builder(pp.mm < pp.cfg.M && pp.kk < pp.K)
                           .then([&](block_builder &bb) {
                               auto view_sub =
                                   pp.view.reshaped_mode(1, std::array<expr, 2u>{J2, Nf})
                                       .subview(bb, pp.mm, j2, slice{}, pp.kk);
                               copy_N_block(bb, X1_view_1d, view_sub, Nf);
                           })
                           .get_product());
            };
        },
        "j2");
}

void f2fft_gen_r2c_half::load(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    auto src = cp.view.reshaped_mode(1, std::array<expr, 2u>{2, p().N_fft});
    global_load(bb, cp, cp.kk, src.subview(bb, slice{}, 0, slice{}, slice{}));
    cp.x_acc->component(1);
    global_load(bb, cp, cp.kk, src.subview(bb, slice{}, 1, slice{}, slice{}));
    cp.x_acc->component(-1);
}

void f2fft_gen_r2c_half::postprocess(block_builder &bb, prepost_params pp) const {
    bb.add(if_selection_builder(pp.mm < pp.cfg.M && pp.kk < pp.K)
               .then([&](block_builder &bb) {
                   auto view = pp.view.subview(bb, pp.mm, slice{}, pp.kk);
                   auto j1 = bb.declare(generic_short(), "j1");
                   bb.add(for_loop_builder(assignment(j1, pp.n_local), j1 <= pp.cfg.N / 4,
                                           add_into(j1, pp.cfg.Nb))
                              .body([&](block_builder &bb) {
                                  postprocess_i(bb, pp.fph, j1, pp.twiddle, pp.cfg.N, pp.X1_view,
                                                view, pp.unscramble);
                              })
                              .get_product());
               })
               .get_product());
}

void f2fft_gen_r2c_half::postprocess_i(block_builder &bb, precision_helper fph, expr i,
                                       expr twiddle, std::size_t N, tensor_view<1u> const &y,
                                       tensor_view<1u> const &X,
                                       unscrambler<clir::expr> const &unscramble) {
    auto i_other = N / 2 - i;
    auto i_load = bb.declare_assign(generic_short(), "i_load", i % (N / 2));
    auto i_other_load = bb.declare_assign(generic_short(), "i_other_load", i_other % (N / 2));
    var y1 = bb.declare_assign(fph.type(2), "yi", y(unscramble(i_load)));
    var y2 = bb.declare_assign(fph.type(2), "yN_i", y(unscramble(i_other_load)));
    bb.assign(y2, init_vector(fph.type(2), {y2.s(0), -y2.s(1)}));
    var a = bb.declare_assign(fph.type(2), "a", (y2 + y1) / fph.constant(2.0));
    var b = bb.declare_assign(fph.type(2), "b", (y2 - y1) / fph.constant(2.0));
    bb.assign(b, complex_mul(fph)(b, twiddle[i]));
    bb.add(X.store(a + b, i));
    if (i != i_other) {
        expr x_other = init_vector(fph.type(2), {a.s(0) - b.s(0), b.s(1) - a.s(1)});
        bb.add(X.store(x_other, i_other));
    }
}

void f2fft_gen_r2c_double::load(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    global_load(bb, cp, 2 * cp.kk, cp.view);
    cp.x_acc->component(1);
    auto const Nf = cp.cfg.factorization.back();
    auto zero = tensor_view(std::make_shared<zero_accessor>(cp.cfg.fp), std::array<expr, 1u>{Nf});
    copy_N_block_with_permutation(bb, zero, cp.x_view, Nf);
    global_load(bb, cp, 2 * cp.kk + 1, cp.view);
    cp.x_acc->component(-1);
}

void f2fft_gen_r2c_double::postprocess(block_builder &bb, prepost_params pp) const {
    auto const N = p().N_fft;
    auto store_fac = [&](block_builder &bb, tensor_view<1u> const &view_a,
                         tensor_view<1u> const &view_b) {
        auto j1 = bb.declare(generic_short(), "j1");
        bb.add(for_loop_builder(assignment(j1, pp.n_local), j1 <= N / 2, add_into(j1, pp.cfg.Nb))
                   .body([&](block_builder &bb) {
                       postprocess_i(bb, pp.fph, j1, N, pp.X1_view, view_a, view_b, pp.unscramble);
                   })
                   .get_product());
    };
    auto view_a = pp.view.subview(bb, pp.mm, slice{}, 2 * pp.kk);
    bb.add(if_selection_builder(pp.mm < pp.cfg.M && 2 * pp.kk + 1 < pp.K)
               .then([&](block_builder &bb) {
                   auto view_b = pp.view.subview(bb, pp.mm, slice{}, 2 * pp.kk + 1);
                   store_fac(bb, view_a, view_b);
               })
               .otherwise([&](block_builder &bb) {
                   bb.add(if_selection_builder(pp.mm < pp.cfg.M && 2 * pp.kk < pp.K)
                              .then([&](block_builder &bb) {
                                  auto zero =
                                      tensor_view(std::make_shared<zero_accessor>(pp.cfg.fp),
                                                  std::array<expr, 1u>{N});
                                  store_fac(bb, view_a, zero);
                              })
                              .get_product());
               })
               .get_product());
}

void f2fft_gen_r2c_double::postprocess_i(block_builder &bb, precision_helper fph, expr i,
                                         std::size_t N, tensor_view<1u> const &x,
                                         tensor_view<1u> const &ya, tensor_view<1u> const &yb,
                                         unscrambler<clir::expr> const &unscramble) {
    auto i_other = bb.declare_assign(generic_short(), "i_other", (N - i) % N);
    auto number_type = fph.type(2);
    var y1 = bb.declare_assign(number_type, "yi", x(unscramble(i)));
    var y2 = bb.declare_assign(number_type, "yN_i", x(unscramble(i_other)));
    bb.assign(y2, init_vector(number_type, {y2.s(0), -y2.s(1)}));
    var tmp = bb.declare_assign(number_type, "tmp", (y2 + y1) / fph.constant(2.0));
    bb.add(ya.store(tmp, i));
    if (yb.store(tmp, i)) {
        bb.assign(tmp, (y2 - y1) / fph.constant(2.0));
        bb.assign(tmp, init_vector(number_type, {-tmp.s(1), tmp.s(0)}));
        bb.add(yb.store(tmp, i));
    }
}

void f2fft_gen_c2r_half::preprocess(block_builder &bb, prepost_params pp) const {
    bb.add(if_selection_builder(pp.mm < pp.cfg.M && pp.kk < pp.K)
               .then([&](block_builder &bb) {
                   auto view = pp.view.subview(bb, pp.mm, slice{}, pp.kk);
                   auto j1 = bb.declare(generic_short(), "j1");
                   bb.add(for_loop_builder(assignment(j1, pp.n_local), j1 <= pp.cfg.N / 4,
                                           add_into(j1, pp.cfg.Nb))
                              .body([&](block_builder &bb) {
                                  preprocess_i(bb, pp.fph, j1, pp.twiddle, pp.cfg.N, view,
                                               pp.X1_view);
                              })
                              .get_product());
               })
               .get_product());
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
}

void f2fft_gen_c2r_half::preprocess_i(block_builder &bb, precision_helper fph, expr i, expr twiddle,
                                      std::size_t N, tensor_view<1u> const &x,
                                      tensor_view<1u> const &X1) {
    auto i_other = N / 2 - i;
    auto i_store = i;
    auto i_other_store = i_other;

    var x1 = bb.declare_assign(fph.type(2), "xi", x(i));
    // ensure that imaginary part of zero-frequency term is zero
    bb.assign(x1.s(1), select(x1.s(1), fph.zero(), cast(fph.select_type(), i == 0)));
    var x2 = bb.declare_assign(fph.type(2), "xN_i", x(i_other));
    bb.assign(x2, init_vector(fph.type(2), {x2.s(0), -x2.s(1)}));
    var a = bb.declare_assign(fph.type(2), "a", x1 + x2);
    var b = bb.declare_assign(fph.type(2), "b", x1 - x2);
    bb.assign(b, complex_mul(fph)(b, twiddle[i]));
    bb.add(X1.store(a + b, i_store));
    bb.add(if_selection_builder(i != 0)
               .then([&](block_builder &bb) {
                   expr x_other = init_vector(fph.type(2), {a.s(0) - b.s(0), b.s(1) - a.s(1)});
                   bb.add(X1.store(x_other, i_other_store));
               })
               .get_product());
}

void f2fft_gen_c2r_half::load(block_builder &bb, copy_params cp) const {
    auto const &f = cp.cfg.factorization;
    auto const J1 = product(f.begin(), f.begin() + f.size() - 1, 1);
    auto const Nf = f.back();
    auto X1_view_1d =
        cp.X1_view.reshaped_mode(0, std::array<expr, 2u>{J1, Nf}).subview(bb, cp.j1, slice{});
    copy_N_block_with_permutation(bb, X1_view_1d, cp.x_view, Nf);
}

void f2fft_gen_c2r_half::postprocess(block_builder &bb, prepost_params pp) const {
    auto const &f = pp.cfg.factorization;
    auto const Nf = f.front();
    auto const J2 = product(f.begin() + 1, f.end(), 1);
    auto unscramble = unscrambler<clir::expr>(f.begin() + 1, f.end());
    unscramble.in0toN(true);
    auto view_4d = pp.view.reshaped_mode(1, std::array<expr, 2u>{2, p().N_fft});
    parallel_2d_loop(
        bb, pp.n_local, pp.cfg.Nb, J2,
        [&](expr &j2) {
            return [&](block_builder &bb) {
                auto X1_view_1d = pp.X1_view.reshaped_mode(0, std::array<expr, 2u>{Nf, J2})
                                      .subview(bb, slice{}, unscramble(j2));
                bb.add(if_selection_builder(pp.mm < pp.cfg.M && pp.kk < pp.K)
                           .then([&](block_builder &bb) {
                               auto view_sub =
                                   view_4d.reshaped_mode(2, std::array<expr, 2u>{J2, Nf})
                                       .subview(bb, pp.mm, slice{}, j2, slice{}, pp.kk);
                               for (int j1 = 0; j1 < Nf; ++j1) {
                                   bb.add(view_sub.store(X1_view_1d(j1).s(0), 0, j1));
                                   bb.add(view_sub.store(X1_view_1d(j1).s(1), 1, j1));
                               }
                           })
                           .get_product());
            };
        },
        "j2");
}

void f2fft_gen_c2r_double::preprocess(block_builder &bb, prepost_params pp) const {
    auto N = p().N_fft;
    auto load_fac2 = [&](block_builder &bb, tensor_view<1u> const &view_a,
                         tensor_view<1u> const &view_b) {
        auto i = var("i");
        bb.add(for_loop_builder(declaration_assignment(generic_short(), i, pp.n_local), i <= N / 2,
                                add_into(i, pp.cfg.Nb))
                   .body([&](block_builder &bb) {
                       preprocess_i(bb, pp.fph, i, N, view_a, view_b, pp.X1_view);
                   })
                   .get_product());
    };
    auto view_a = pp.view.subview(bb, pp.mm, slice{}, 2 * pp.kk);
    bb.add(if_selection_builder(pp.mm < pp.cfg.M && 2 * pp.kk + 1 < pp.K)
               .then([&](block_builder &bb) {
                   auto view_b = pp.view.subview(bb, pp.mm, slice{}, 2 * pp.kk + 1);
                   load_fac2(bb, view_a, view_b);
               })
               .otherwise([&](block_builder &bb) {
                   bb.add(if_selection_builder(pp.mm < pp.cfg.M && 2 * pp.kk < pp.K)
                              .then([&](block_builder &bb) {
                                  auto zero =
                                      tensor_view(std::make_shared<zero_accessor>(pp.cfg.fp),
                                                  std::array<expr, 1u>{N});
                                  load_fac2(bb, view_a, zero);
                              })
                              .get_product());
               })
               .get_product());
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
}

void f2fft_gen_c2r_double::preprocess_i(block_builder &bb, precision_helper fph, expr i,
                                        std::size_t N, tensor_view<1u> const &xa,
                                        tensor_view<1u> const &xb, tensor_view<1u> const &y) {
    data_type cast_type = fph.select_type();
    auto number_type = fph.type(2);
    expr i_other = N - i;
    expr i_out = i;
    expr i_other_out = i_other;
    var ai = bb.declare_assign(number_type, "ai", xa(i));
    var bi = bb.declare_assign(number_type, "bi", xb(i));
    // ensure that imaginary part of zero-frequency term is zero
    bb.assign(ai.s(1), select(ai.s(1), fph.zero(), cast(cast_type, i == 0)));
    bb.assign(bi.s(1), select(bi.s(1), fph.zero(), cast(cast_type, i == 0)));
    bb.assign(bi, init_vector(number_type, {-bi.s(1), bi.s(0)}));
    bb.add(y.store(ai + bi, i_out));
    bb.add(if_selection_builder(i_other < N)
               .then([&](block_builder &bb) {
                   auto tmp = bb.declare_assign(number_type, "tmp", ai - bi);
                   bb.add(y.store(init_vector(number_type, {tmp.s(0), -tmp.s(1)}), i_other_out));
               })
               .get_product());
}

void f2fft_gen_c2r_double::load(block_builder &bb, copy_params cp) const {
    auto const &f = cp.cfg.factorization;
    auto const J1 = product(f.begin(), f.begin() + f.size() - 1, 1);
    auto const Nf = f.back();
    auto X1_view_1d =
        cp.X1_view.reshaped_mode(0, std::array<expr, 2u>{J1, Nf}).subview(bb, cp.j1, slice{});
    copy_N_block_with_permutation(bb, X1_view_1d, cp.x_view, Nf);
}

void f2fft_gen_c2r_double::postprocess(block_builder &bb, prepost_params pp) const {
    auto const &f = pp.cfg.factorization;
    auto const Nf = f.front();
    auto const J2 = product(f.begin() + 1, f.end(), 1);
    auto unscramble = unscrambler<clir::expr>(f.begin() + 1, f.end());
    unscramble.in0toN(true);
    parallel_2d_loop(
        bb, pp.n_local, pp.cfg.Nb, J2,
        [&](expr &j2) {
            return [&](block_builder &bb) {
                auto X1_view_1d = pp.X1_view.reshaped_mode(0, std::array<expr, 2u>{Nf, J2})
                                      .subview(bb, slice{}, unscramble(j2));
                auto view_4d = pp.view.reshaped_mode(1, std::array<expr, 2u>{J2, Nf});
                for (int c = 0; c < 2; ++c) {
                    bb.add(if_selection_builder(pp.mm < pp.cfg.M && 2 * pp.kk + c < pp.K)
                               .then([&](block_builder &bb) {
                                   auto view_sub =
                                       view_4d.subview(bb, pp.mm, j2, slice{}, 2 * pp.kk + c);
                                   for (int j1 = 0; j1 < Nf; ++j1) {
                                       bb.add(view_sub.store(X1_view_1d(j1).s(c), j1));
                                   }
                               })
                               .get_product());
                }
            };
        },
        "j2");
}

} // namespace bbfft
