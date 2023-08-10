// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"
#include "bbfft/detail/generator_impl.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "generator/snippet.hpp"
#include "generator/tensor_accessor.hpp"
#include "generator/tensor_view.hpp"
#include "math.hpp"
#include "mixed_radix_fft.hpp"
#include "prime_factorization.hpp"
#include "scrambler.hpp"

#include "clir/attr_defs.hpp"
#include "clir/builder.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/visitor/codegen_opencl.hpp"
#include "clir/visitor/unique_names.hpp"
#include "clir/visitor/unsafe_simplification.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

using namespace clir;

namespace bbfft {

factor2_slm_configuration configure_factor2_slm_fft(configuration const &cfg,
                                                    device_info const &info) {
    std::size_t N = cfg.shape[1];
    std::size_t N_slm = N;
    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    if (is_real) {
        N_slm = N + 1;
    }
    auto fac2 = factor2(N);
    std::size_t N1 = fac2.first;
    std::size_t N2 = fac2.second;
    auto M = cfg.shape[0];
    std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);

    std::size_t sgs = info.min_subgroup_size();
    std::size_t work_group_size_limit = info.max_subgroup_size();

    std::size_t Nb = N1;
    std::size_t max_compute_Mb = info.max_work_group_size / Nb;
    std::size_t max_slm_Mb = info.local_memory_size / (2 * N_slm * sizeof_real);
    std::size_t max_Mb = std::min(max_compute_Mb, max_slm_Mb);
    max_Mb = std::min(max_Mb, work_group_size_limit);
    std::size_t Mb = std::min(max_Mb, min_power_of_2_greater_equal(M));
    std::size_t max_compute_Kb = std::max(std::size_t(1), info.max_work_group_size / (Mb * Nb));
    std::size_t max_slm_Kb = info.local_memory_size / (2 * Mb * N_slm * sizeof_real);
    std::size_t max_Kb = std::min(max_compute_Kb, max_slm_Kb);
    std::size_t min_Kb = (sgs - 1) / (Mb * Nb) + 1;
    std::size_t Kb = std::min(cfg.shape[2], std::min(min_Kb, max_Kb));

    bool inplace_unsupported = is_real && Mb < M;

    auto istride = std::array<std::size_t, 3>{cfg.istride[0], cfg.istride[1], cfg.istride[2]};
    auto ostride = std::array<std::size_t, 3>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]};

    std::stringstream ss;
    return {
        static_cast<int>(cfg.dir),   // direction
        M,                           // M
        Mb,                          // Mb
        N1,                          // N1
        N2,                          // N2
        Nb,                          // Nb
        Kb,                          // Kb
        sgs,                         // sgs
        cfg.fp,                      // precision
        cfg.type,                    // transform_type
        istride,                     // istride
        ostride,                     // ostride
        inplace_unsupported,         // inplace_unsupported
        cfg.callbacks.load_function, // load_function
        cfg.callbacks.store_function // store_function
    };
}

std::string factor2_slm_configuration::identifier() const {
    std::ostringstream oss;
    oss << "f2fft_" << (direction < 0 ? 'm' : 'p') << std::abs(direction) << "_M" << M << "_Mb"
        << Mb << "_N1" << N1 << "_N2" << N2 << "_Nb" << Nb << "_Kb" << Kb << "_sgs" << sgs << "_f"
        << static_cast<int>(fp) * 8 << '_' << to_string(type) << "_is";
    for (auto const &is : istride) {
        oss << is << "_";
    }
    oss << "os";
    for (auto const &os : ostride) {
        oss << os << "_";
    }
    oss << "in" << inplace_unsupported;
    if (load_function) {
        oss << "_" << load_function;
    }
    if (store_function) {
        oss << "_" << store_function;
    }
    return oss.str();
}

void r2c_post_i(block_builder &bb, precision_helper fph, expr i, std::size_t N,
                tensor_view<1u> const &x, tensor_view<1u> const &ya, tensor_view<1u> const &yb) {
    expr i_other = (N - i) % N;
    auto number_type = fph.type(2);
    var y1 = bb.declare_assign(number_type, "yi", x(i));
    var y2 = bb.declare_assign(number_type, "yN_i", x(i_other));
    bb.assign(y2, init_vector(number_type, {y2.s(0), -y2.s(1)}));
    var tmp = bb.declare_assign(number_type, "tmp", (y2 + y1) / fph.constant(2.0));
    bb.add(ya.store(tmp, i));
    if (yb.store(tmp, i)) {
        bb.assign(tmp, (y2 - y1) / fph.constant(2.0));
        bb.assign(tmp, init_vector(number_type, {-tmp.s(1), tmp.s(0)}));
        bb.add(yb.store(tmp, i));
    }
}

void c2r_pre_i(block_builder &bb, precision_helper fph, expr i, std::size_t N1, std::size_t N2,
               tensor_view<1u> const &xa, tensor_view<1u> const &xb, tensor_view<1u> const &y) {
    data_type cast_type = fph.select_type();
    auto number_type = fph.type(2);
    std::size_t N = N1 * N2;
    expr i_other = N - i;
    expr i_out = i / N1 + i % N1 * N2;
    expr i_other_out = i_other / N1 + i_other % N1 * N2;
    var ai = bb.declare_assign(number_type, "ai", xa(i));
    var bi = bb.declare_assign(number_type, "bi", xb(i));
    // ensure that imaginary part of zero-frequency term is zero
    bb.assign(ai.s(1), select(ai.s(1), fph.zero(), cast(cast_type, i == 0)));
    bb.assign(bi.s(1), select(bi.s(1), fph.zero(), cast(cast_type, i == 0)));
    bb.assign(bi, init_vector(number_type, {-bi.s(1), bi.s(0)}));
    bb.add(y.store(ai + bi, i_out));
    bb.add(if_selection_builder(i_other < N1 * N2)
               .then([&](block_builder &bb) {
                   auto tmp = bb.declare_assign(number_type, "tmp", ai - bi);
                   bb.add(y.store(init_vector(number_type, {tmp.s(0), -tmp.s(1)}), i_other_out));
               })
               .get_product());
}

void generate_factor2_slm_fft(std::ostream &os, factor2_slm_configuration const &cfg,
                              std::string_view name) {
    std::size_t N1 = cfg.N1;
    std::size_t N2 = cfg.N2;
    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    auto N = N1 * N2;
    auto N_in = cfg.type == transform_type::c2r ? N / 2 + 1 : N;
    auto N_out = cfg.type == transform_type::r2c ? N / 2 + 1 : N;
    auto N_slm = is_real ? N + 1 : N;

    auto in = var("in");
    auto out = var("out");
    auto twiddle = var("twiddle");
    auto K = var("K");

    auto fph = precision_helper{cfg.fp};
    auto in_ty = cfg.type == transform_type::r2c ? fph.type(address_space::global_t)
                                                 : fph.type(2, address_space::global_t);
    auto out_ty = cfg.type == transform_type::c2r ? fph.type(address_space::global_t)
                                                  : fph.type(2, address_space::global_t);

    auto slm_shape = std::array<expr, 3>{cfg.Mb, N_slm, cfg.Kb};
    auto slm_ty = fph.type(2, address_space::local_t);

    auto fft_inplace = &generate_fft::basic_inplace;

    auto fb = kernel_builder{name.empty() ? cfg.identifier() : std::string(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(pointer_to(fph.type(2, address_space::constant_t)), twiddle);
    fb.argument(generic_ulong(), K);
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Nb),
                                      static_cast<int>(cfg.Kb)));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));

    std::shared_ptr<tensor_accessor> in_acc, out_acc;
    if (cfg.load_function) {
        in_acc = std::make_shared<callback_accessor>(in, in_ty, cfg.load_function);
    } else {
        in_acc = std::make_shared<array_accessor>(in, in_ty);
    }
    if (cfg.store_function) {
        out_acc = std::make_shared<callback_accessor>(out, out_ty, nullptr, cfg.store_function);
    } else {
        out_acc = std::make_shared<array_accessor>(out, out_ty);
    }

    fb.body([&](block_builder &bb) {
        auto X1 = bb.declare(array_of(slm_ty, cfg.Kb * N_slm * cfg.Mb), "X1");
        auto kk = bb.declare_assign(generic_size(), "kk", get_global_id(2));
        auto mm = bb.declare_assign(generic_size(), "mm", get_global_id(0));
        auto n_local = bb.declare_assign(generic_size(), "n_local", get_local_id(1));

        auto X1_view = tensor_view(std::make_shared<array_accessor>(X1, slm_ty),
                                   std::array<expr, 3u>{cfg.Mb, N_slm, cfg.Kb})
                           .subview(bb, get_local_id(0), slice{}, get_local_id(2));

        auto j1 = bb.declare(generic_short(), "j1");

        auto parallel_n2 = [&](block_builder &bb, expr loop_var, std::size_t Nk, auto kernel) {
            if (Nk == cfg.Nb) {
                bb.assign(loop_var, n_local);
                bb.add(block_builder{}.body(kernel).get_product());
            } else {
                bb.add(for_loop_builder(assignment(loop_var, n_local), loop_var < Nk,
                                        add_into(loop_var, cfg.Nb))
                           .body(kernel)
                           .get_product());
            }
        };

        auto compute_fac1 = [&](block_builder &bb, auto load_function) {
            auto xy_ty_N2 = data_type(array_of(fph.type(2), N2));
            auto x = bb.declare(xy_ty_N2, "x");
            auto x_acc = std::make_shared<array_accessor>(x, xy_ty_N2);
            auto x_view = tensor_view(x_acc, std::array<expr, 1u>{N2});

            load_function(bb, x_acc, x_view);

            var tw_n2 = var("tw_n2");
            bb.declare_assign(pointer_to(fph.type(2, address_space::constant_t)), tw_n2,
                              twiddle + j1 * N2);
            auto factor = trial_division(N2);
            fft_inplace(bb, cfg.fp, cfg.direction, factor, x, tw_n2);

            auto X1_view_1d =
                X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1}).subview(bb, slice{}, j1);
            copy_N_block_with_permutation(bb, x_view, X1_view_1d, N2, unscrambler(factor));
        };

        auto in_view =
            tensor_view(in_acc, {cfg.M, N_in, K},
                        std::array<expr, 3u>{cfg.istride[0], cfg.istride[1], cfg.istride[2]});
        if (cfg.type == transform_type::c2r) {
            auto load_fac2 = [&](block_builder &bb, tensor_view<1u> const &in_view_a,
                                 tensor_view<1u> const &in_view_b) {
                auto i = var("i");
                bb.add(for_loop_builder(declaration_assignment(generic_short(), i, n_local),
                                        i <= N1 * N2 / 2, add_into(i, cfg.Nb))
                           .body([&](block_builder &bb) {
                               c2r_pre_i(bb, cfg.fp, i, N1, N2, in_view_a, in_view_b, X1_view);
                           })
                           .get_product());
            };
            auto in_view_a = in_view.subview(bb, mm, slice{}, 2 * kk);
            bb.add(if_selection_builder(mm < cfg.M && 2 * kk + 1 < K)
                       .then([&](block_builder &bb) {
                           auto in_view_b = in_view.subview(bb, mm, slice{}, 2 * kk + 1);
                           load_fac2(bb, in_view_a, in_view_b);
                       })
                       .otherwise([&](block_builder &bb) {
                           bb.add(if_selection_builder(mm < cfg.M && 2 * kk < K)
                                      .then([&](block_builder &bb) {
                                          auto zero =
                                              tensor_view(std::make_shared<zero_accessor>(cfg.fp),
                                                          std::array<expr, 1u>{N});
                                          load_fac2(bb, in_view_a, zero);
                                      })
                                      .get_product());
                       })
                       .get_product());
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            auto my_compute_fac1 = [&](block_builder &bb) {
                compute_fac1(bb, [&](block_builder &bb, auto, auto x_view) {
                    auto X1_view_1d = X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1})
                                          .subview(bb, slice{}, j1);
                    copy_N_block_with_permutation(bb, X1_view_1d, x_view, N2);
                });
            };
            parallel_n2(bb, j1, N1, my_compute_fac1);
        } else {
            auto const load_global = [&](block_builder &bb, auto k, auto x_view) {
                bb.add(if_selection_builder(mm < cfg.M && k < K)
                           .then([&](block_builder &bb) {
                               auto in_view_sub =
                                   in_view.reshaped_mode(1, std::array<expr, 2u>{N1, N2})
                                       .subview(bb, mm, j1, slice{}, k);
                               copy_N_block_with_permutation(bb, in_view_sub, x_view, N2);
                           })
                           .get_product());
            };
            auto my_compute_fac1 = [&](block_builder &bb) {
                compute_fac1(bb, [&](block_builder &bb, auto x_acc, auto x_view) {
                    if (cfg.type == transform_type::r2c) {
                        x_acc->component(0);
                        load_global(bb, 2 * kk, x_view);
                        x_acc->component(1);
                        auto zero = tensor_view(std::make_shared<zero_accessor>(cfg.fp),
                                                std::array<expr, 1u>{N2});
                        copy_N_block_with_permutation(bb, zero, x_view, N2);
                        load_global(bb, 2 * kk + 1, x_view);
                        x_acc->component(-1);
                    } else {
                        load_global(bb, kk, x_view);
                    }
                });
            };
            parallel_n2(bb, j1, N1, my_compute_fac1);
        }
        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        auto n2 = bb.declare(generic_short(), "n2");

        auto compute_fac2 = [&](block_builder &bb, auto store_function) {
            auto xy_ty_N1 = data_type(array_of(fph.type(2), N1));
            auto x = bb.declare(xy_ty_N1, "x");
            auto x_acc = std::make_shared<array_accessor>(x, xy_ty_N1);
            auto x_view = tensor_view(x_acc, std::array<expr, 1u>{N1});

            auto X1_view_1d =
                X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1}).subview(bb, n2, slice{});
            copy_N_block_with_permutation(bb, X1_view_1d, x_view, N1);

            auto factor = trial_division(N1);
            fft_inplace(bb, cfg.fp, cfg.direction, factor, x, nullptr);

            store_function(bb, x_acc, x_view);
        };

        auto out_view =
            tensor_view(out_acc, {cfg.M, N_out, K},
                        std::array<expr, 3u>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]});
        if (cfg.type == transform_type::r2c) {
            auto my_compute_fac2 = [&](block_builder &bb) {
                compute_fac2(bb, [&](block_builder &bb, auto, auto x_view) {
                    auto X1_view_1d = X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1})
                                          .subview(bb, n2, slice{});
                    auto factor = trial_division(N1);
                    copy_N_block_with_permutation(bb, x_view, X1_view_1d, N1, unscrambler(factor));
                });
            };
            parallel_n2(bb, n2, N2, my_compute_fac2);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            auto store_fac = [&](block_builder &bb, tensor_view<1u> const &out_view_a,
                                 tensor_view<1u> const &out_view_b) {
                bb.add(for_loop_builder(assignment(n2, n_local), n2 <= N1 * N2 / 2,
                                        add_into(n2, cfg.Nb))
                           .body([&](block_builder &bb) {
                               r2c_post_i(bb, cfg.fp, n2, N1 * N2, X1_view, out_view_a, out_view_b);
                           })
                           .get_product());
            };
            auto out_view_a = out_view.subview(bb, mm, slice{}, 2 * kk);
            bb.add(if_selection_builder(mm < cfg.M && 2 * kk + 1 < K)
                       .then([&](block_builder &bb) {
                           auto out_view_b = out_view.subview(bb, mm, slice{}, 2 * kk + 1);
                           store_fac(bb, out_view_a, out_view_b);
                       })
                       .otherwise([&](block_builder &bb) {
                           bb.add(if_selection_builder(mm < cfg.M && 2 * kk < K)
                                      .then([&](block_builder &bb) {
                                          auto zero =
                                              tensor_view(std::make_shared<zero_accessor>(cfg.fp),
                                                          std::array<expr, 1u>{N});
                                          store_fac(bb, out_view_a, zero);
                                      })
                                      .get_product());
                       })
                       .get_product());
        } else {
            auto factor = trial_division(N1);
            auto const store_global = [&](block_builder &bb, auto k, auto x_view) {
                bb.add(if_selection_builder(mm < cfg.M && k < K)
                           .then([&](block_builder &bb) {
                               auto out_view_sub =
                                   out_view.reshaped_mode(1, std::array<expr, 2u>{N2, N1})
                                       .subview(bb, mm, n2, slice{}, k);
                               copy_N_block_with_permutation(bb, x_view, out_view_sub, N1,
                                                             unscrambler(factor));
                           })
                           .get_product());
            };
            auto my_compute_fac2 = [&](block_builder &bb) {
                compute_fac2(bb, [&](block_builder &bb, auto x_acc, auto x_view) {
                    if (cfg.type == transform_type::c2r) {
                        x_acc->component(0);
                        store_global(bb, 2 * kk, x_view);
                        x_acc->component(1);
                        store_global(bb, 2 * kk + 1, x_view);
                        x_acc->component(-1);
                    } else {
                        store_global(bb, kk, x_view);
                    }
                });
            };
            parallel_n2(bb, n2, N2, my_compute_fac2);
        }
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

} // namespace bbfft
