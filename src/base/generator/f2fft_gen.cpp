// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "f2fft_gen.hpp"
#include "mixed_radix_fft.hpp"
#include "prime_factorization.hpp"
#include "root_of_unity.hpp"
#include "scrambler.hpp"

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

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace clir;

namespace bbfft {

void f2fft_gen::generate(std::ostream &os, factor2_slm_configuration const &cfg,
                         std::string_view name) const {
    std::size_t N1 = cfg.N1;
    std::size_t N2 = cfg.N2;

    auto in = var("in");
    auto out = var("out");
    auto twiddle = var("twiddle");
    auto K = var("K");

    auto fph = precision_helper{cfg.fp};
    auto in_ty = fph.type(p_.in_components, address_space::global_t);
    auto out_ty = fph.type(p_.out_components, address_space::global_t);
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
        auto X1 = bb.declare(array_of(slm_ty, cfg.Kb * p_.N_slm * cfg.Mb), "X1");
        auto kk = bb.declare_assign(generic_size(), "kk", get_global_id(2));
        auto mm = bb.declare_assign(generic_size(), "mm", get_global_id(0));
        auto n_local = bb.declare_assign(generic_size(), "n_local", get_local_id(1));

        auto X1_view = tensor_view(std::make_shared<array_accessor>(X1, slm_ty),
                                   std::array<expr, 3u>{cfg.Mb, p_.N_slm, cfg.Kb})
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

        auto in_view =
            tensor_view(in_acc, {cfg.M, p_.N_in, K},
                        std::array<expr, 3u>{cfg.istride[0], cfg.istride[1], cfg.istride[2]});
        preprocess(
            bb, prepost_params{cfg, fph, in_view, X1_view, mm, n_local, kk, K, twiddle + N1 * N2});
        auto compute_fac1 = [&](block_builder &bb) {
            auto xy_ty_N2 = data_type(array_of(fph.type(2), N2));
            auto x = bb.declare(xy_ty_N2, "x");
            auto x_acc = std::make_shared<array_accessor>(x, xy_ty_N2);
            auto x_view = tensor_view(x_acc, std::array<expr, 1u>{N2});

            load(bb, copy_params{cfg, fph, in_view, X1_view, x_view, x_acc, mm, kk, K, j1});

            var tw_n2 = var("tw_n2");
            bb.declare_assign(pointer_to(fph.type(2, address_space::constant_t)), tw_n2,
                              twiddle + j1 * N2);
            auto factor = trial_division(N2);
            fft_inplace(bb, cfg.fp, cfg.direction, factor, x, tw_n2);

            auto X1_view_1d =
                X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1}).subview(bb, slice{}, j1);
            copy_N_block_with_permutation(bb, x_view, X1_view_1d, N2, unscrambler(factor));
        };
        parallel_n2(bb, j1, N1, compute_fac1);

        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        auto n2 = bb.declare(generic_short(), "n2");

        auto out_view =
            tensor_view(out_acc, {cfg.M, p_.N_out, K},
                        std::array<expr, 3u>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]});
        auto compute_fac2 = [&](block_builder &bb) {
            auto xy_ty_N1 = data_type(array_of(fph.type(2), N1));
            auto x = bb.declare(xy_ty_N1, "x");
            auto x_acc = std::make_shared<array_accessor>(x, xy_ty_N1);
            auto x_view = tensor_view(x_acc, std::array<expr, 1u>{N1});

            auto X1_view_1d =
                X1_view.reshaped_mode(0, std::array<expr, 2u>{N2, N1}).subview(bb, n2, slice{});
            copy_N_block_with_permutation(bb, X1_view_1d, x_view, N1);

            auto factor = trial_division(N1);
            fft_inplace(bb, cfg.fp, cfg.direction, factor, x, nullptr);

            store(bb, copy_params{cfg, fph, out_view, X1_view, x_view, x_acc, mm, kk, K, n2,
                                  unscrambler(factor)});
        };
        parallel_n2(bb, n2, N2, compute_fac2);
        postprocess(
            bb, prepost_params{cfg, fph, out_view, X1_view, mm, n_local, kk, K, twiddle + N1 * N2});
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

void f2fft_gen::global_load(block_builder &bb, copy_params cp, expr k,
                            tensor_view<3u> const &view) const {
    bb.add(if_selection_builder(cp.mm < cp.cfg.M && k < cp.K)
               .then([&](block_builder &bb) {
                   auto view_sub = view.reshaped_mode(1, std::array<expr, 2u>{cp.cfg.N1, cp.cfg.N2})
                                       .subview(bb, cp.mm, cp.j1, slice{}, k);
                   copy_N_block_with_permutation(bb, view_sub, cp.x_view, cp.cfg.N2);
               })
               .get_product());
}

void f2fft_gen::global_store(block_builder &bb, copy_params cp, expr k,
                             tensor_view<3u> const &view) const {
    bb.add(if_selection_builder(cp.mm < cp.cfg.M && k < cp.K)
               .then([&](block_builder &bb) {
                   auto view_sub = view.reshaped_mode(1, std::array<expr, 2u>{cp.cfg.N2, cp.cfg.N1})
                                       .subview(bb, cp.mm, cp.j1, slice{}, k);
                   copy_N_block_with_permutation(bb, cp.x_view, view_sub, cp.cfg.N1, cp.P);
               })
               .get_product());
}

void f2fft_gen_c2c::load(block_builder &bb, copy_params cp) const {
    global_load(bb, cp, cp.kk, cp.view);
}

void f2fft_gen_c2c::store(block_builder &bb, copy_params cp) const {
    global_store(bb, cp, cp.kk, cp.view);
}

void f2fft_gen_r2c_half::load(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    auto src = cp.view.reshaped_mode(1, std::array<expr, 2u>{2, p().N_fft});
    global_load(bb, cp, cp.kk, src.subview(bb, slice{}, 0, slice{}, slice{}));
    cp.x_acc->component(1);
    global_load(bb, cp, cp.kk, src.subview(bb, slice{}, 1, slice{}, slice{}));
    cp.x_acc->component(-1);
}

void f2fft_gen_r2c_half::store(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    auto X1_view_1d = cp.X1_view.reshaped_mode(0, std::array<expr, 3u>{2u, cp.cfg.N2, cp.cfg.N1});
    copy_N_block_with_permutation(bb, cp.x_view, X1_view_1d.subview(bb, 0u, cp.j1, slice{}),
                                  cp.cfg.N1, cp.P);
    cp.x_acc->component(1);
    copy_N_block_with_permutation(bb, cp.x_view, X1_view_1d.subview(bb, 1u, cp.j1, slice{}),
                                  cp.cfg.N1, cp.P);
    cp.x_acc->component(-1);
}

void f2fft_gen_r2c_half::postprocess(block_builder &bb, prepost_params pp) const {
    auto const N = pp.cfg.N1 * pp.cfg.N2;
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    auto view = pp.view.subview(bb, pp.mm, slice{}, pp.kk);
    auto j1 = bb.declare(generic_short(), "j1");
    bb.add(for_loop_builder(assignment(j1, pp.n_local), j1 <= N / 2, add_into(j1, pp.cfg.Nb))
               .body([&](block_builder &bb) {
                   postprocess_i(bb, pp.fph, j1, pp.twiddle, N, pp.X1_view, view);
               })
               .get_product());
}

void f2fft_gen_r2c_half::postprocess_i(block_builder &bb, precision_helper fph, expr i,
                                       expr twiddle, std::size_t N, tensor_view<1u> const &y,
                                       tensor_view<1u> const &X) {
    auto i_other = N - i;
    auto i_load = i % N;
    auto i_other_load = i_other % N;
    var y1 = bb.declare_assign(fph.type(2), "yi", y(i_load));
    var y2 = bb.declare_assign(fph.type(2), "yN_i", y(i_other_load));
    bb.assign(y2, init_vector(fph.type(2), {y2.s(0), -y2.s(1)}));
    var a = bb.declare_assign(fph.type(2), "a", (y2 + y1) / fph.constant(2.0));
    var b = bb.declare_assign(fph.type(2), "b", (y2 - y1) / fph.constant(2.0));
    bb.assign(b, init_vector(fph.type(2), {-b.s(1), b.s(0)}));
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
    auto zero =
        tensor_view(std::make_shared<zero_accessor>(cp.cfg.fp), std::array<expr, 1u>{cp.cfg.N2});
    copy_N_block_with_permutation(bb, zero, cp.x_view, cp.cfg.N2);
    global_load(bb, cp, 2 * cp.kk + 1, cp.view);
    cp.x_acc->component(-1);
}

void f2fft_gen_r2c_double::store(block_builder &bb, copy_params cp) const {
    auto X1_view_1d = cp.X1_view.reshaped_mode(0, std::array<expr, 2u>{cp.cfg.N2, cp.cfg.N1})
                          .subview(bb, cp.j1, slice{});
    copy_N_block_with_permutation(bb, cp.x_view, X1_view_1d, cp.cfg.N1, cp.P);
}

void f2fft_gen_r2c_double::postprocess(block_builder &bb, prepost_params pp) const {
    auto const N = pp.cfg.N1 * pp.cfg.N2;
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    auto store_fac = [&](block_builder &bb, tensor_view<1u> const &view_a,
                         tensor_view<1u> const &view_b) {
        auto j1 = bb.declare(generic_short(), "j1");
        bb.add(for_loop_builder(assignment(j1, pp.n_local), j1 <= N / 2, add_into(j1, pp.cfg.Nb))
                   .body([&](block_builder &bb) {
                       postprocess_i(bb, pp.fph, j1, N, pp.X1_view, view_a, view_b);
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
                                         tensor_view<1u> const &ya, tensor_view<1u> const &yb) {
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

void f2fft_gen_c2r_half::load(block_builder &bb, copy_params cp) const {
    // copy_mbNkb_block_on_2D_grid(bb, cp.view, cp.X1_view, cp.mb, p().N_in, cp.kb);
    // bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));

    // preprocess_i(bb, cp.fph, cp.X1_1d, cp.x_view, cp.cfg.N);
}

void f2fft_gen_c2r_half::store(block_builder &bb, copy_params cp) const {
    // auto X1_dest = cp.X1_1d.reshaped_mode(0, std::array<expr, 2u>{2, p().N_fft});
    // cp.x_acc->component(0);
    // copy_N_block_with_permutation(bb, cp.x_view, X1_dest.subview(bb, 0, slice{}), p().N_fft,
    // cp.P); cp.x_acc->component(1); copy_N_block_with_permutation(bb, cp.x_view,
    // X1_dest.subview(bb, 1, slice{}), p().N_fft, cp.P); cp.x_acc->component(-1);

    // bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    // copy_mbNkb_block_on_2D_grid(bb, cp.X1_view, cp.view, cp.mb, p().N_out, cp.kb);
}

void f2fft_gen_c2r_half::preprocess_i(block_builder &bb, precision_helper fph,
                                      tensor_view<1u> const &X1, tensor_view<1u> const &x,
                                      std::size_t N) {

    // for (std::size_t i = 0; i <= N / 4; ++i) {
    // auto i_other = N / 2 - i;
    // auto i_store = i % (N / 2);
    // auto i_other_store = i_other % (N / 2);

    // var x1 = bb.declare_assign(fph.type(2), "xi", X1(i));
    //// ensure that imaginary part of zero-frequency term is zero
    // if (i == 0) {
    // bb.assign(x1.s(1), fph.zero());
    //}
    // var x2 = bb.declare_assign(fph.type(2), "xN_i", X1(i_other));
    // bb.assign(x2, init_vector(fph.type(2), {x2.s(0), -x2.s(1)}));
    // var a = bb.declare_assign(fph.type(2), "a", x1 + x2);
    // var b = bb.declare_assign(fph.type(2), "b", x1 - x2);
    // auto tw = power_of_w(i, N) * std::complex<double>{0.0, 1.0};
    // bb.assign(b, complex_mul(fph)(b, tw));
    // bb.add(x.store(a + b, i_store));
    // if (i_store != i_other_store) {
    // expr x_other = init_vector(fph.type(2), {a.s(0) - b.s(0), b.s(1) - a.s(1)});
    // bb.add(x.store(x_other, i_other_store));
    //}
    //}
}

void f2fft_gen_c2r_double::load(block_builder &bb, copy_params cp) const {
    auto X1_view_1d = cp.X1_view.reshaped_mode(0, std::array<expr, 2u>{cp.cfg.N2, cp.cfg.N1})
                          .subview(bb, slice{}, cp.j1);
    copy_N_block_with_permutation(bb, X1_view_1d, cp.x_view, cp.cfg.N2);
}

void f2fft_gen_c2r_double::store(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    global_store(bb, cp, 2 * cp.kk, cp.view);
    cp.x_acc->component(1);
    global_store(bb, cp, 2 * cp.kk + 1, cp.view);
    cp.x_acc->component(-1);
}

void f2fft_gen_c2r_double::preprocess(block_builder &bb, prepost_params pp) const {
    auto N = pp.cfg.N1 * pp.cfg.N2;
    auto load_fac2 = [&](block_builder &bb, tensor_view<1u> const &view_a,
                         tensor_view<1u> const &view_b) {
        auto i = var("i");
        bb.add(for_loop_builder(declaration_assignment(generic_short(), i, pp.n_local), i <= N / 2,
                                add_into(i, pp.cfg.Nb))
                   .body([&](block_builder &bb) {
                       preprocess_i(bb, pp.fph, i, pp.cfg.N1, pp.cfg.N2, view_a, view_b,
                                    pp.X1_view);
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
                                        std::size_t N1, std::size_t N2, tensor_view<1u> const &xa,
                                        tensor_view<1u> const &xb, tensor_view<1u> const &y) {
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

} // namespace bbfft
