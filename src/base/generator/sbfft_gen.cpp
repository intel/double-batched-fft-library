// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "sbfft_gen.hpp"
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

void sbfft_gen::generate(std::ostream &os, small_batch_configuration const &cfg,
                         std::string_view name) const {
    auto in = var("in");
    auto out = var("out");
    auto K = var("K");

    auto fph = precision_helper{cfg.fp};
    auto in_ty = fph.type(p_.in_components, address_space::global_t);
    auto out_ty = fph.type(p_.out_components, address_space::global_t);
    auto slm_in_ty = fph.type(p_.in_components, address_space::local_t);
    auto slm_out_ty = fph.type(p_.out_components, address_space::local_t);

    auto fb = kernel_builder{name.empty() ? cfg.identifier() : std::string(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(generic_ulong(), K);
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Kb), 1));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));

    auto xy_ty = data_type(array_of(fph.type(2), p_.N_fft));

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

    // load in SLM from global memory, load transposed in registers from SLM
    fb.body([&](block_builder &bb) {
        expr mb = nullptr;
        if (cfg.M < cfg.Mb) {
            mb = cfg.M;
        } else if (cfg.M % cfg.Mb == 0) {
            mb = cfg.Mb;
        } else {
            mb = bb.declare_assign(generic_uint(), "mb", cfg.M - get_group_id(0) * cfg.Mb);
        }
        auto k_first =
            bb.declare_assign(generic_size(), "k_first", get_group_id(1) * p_.k_stride * cfg.Kb);
        auto kb = bb.declare_assign(generic_uint(), "kb", (K - k_first - 1) / p_.k_stride + 1);
        expr kb_odd = nullptr;
        if (p_.k_stride == 2) {
            kb_odd = bb.declare_assign(generic_uint(), "kb_odd", kb - K % 2);
        }
        bb.assign(kb, select(cfg.Kb, kb, kb < cfg.Kb));
        if (p_.k_stride == 2) {
            bb.assign(kb_odd, select(cfg.Kb, kb_odd, kb_odd < cfg.Kb));
        }

        auto in_view =
            tensor_view(in_acc, {cfg.M, p_.N_in, K},
                        std::array<expr, 3u>{cfg.istride[0], cfg.istride[1], cfg.istride[2]})
                .subview(bb, slice{get_group_id(0) * cfg.Mb, mb}, slice{}, slice{k_first, kb});

        auto X1 = bb.declare(
            array_of(fph.type(2, address_space::local_t), cfg.Kb * p_.N_slm * cfg.Mb), "X1");
        auto X1_in =
            bb.declare_assign(pointer_to(slm_in_ty), "X1_in", cast(pointer_to(slm_in_ty), X1));

        auto X1_in_view = tensor_view(std::make_shared<array_accessor>(X1_in, slm_in_ty),
                                      std::array<expr, 3u>{cfg.Mb, p_.N_in, cfg.Kb});
        auto X1_in_1d =
            X1_in_view.subview(bb, get_local_id(0), slice{0u, p_.N_in}, get_local_id(1));

        auto x = bb.declare(xy_ty, "x");
        auto x_acc = std::make_shared<array_accessor>(x, xy_ty);
        auto x_view = tensor_view(x_acc, std::array<expr, 1u>{p_.N_fft});

        load(bb, copy_params{cfg, fph, in_view, X1_in_view, X1_in_1d, x_view, x_acc, mb, K, kb,
                             kb_odd});

        auto factorization = trial_division(p_.N_fft);
        generate_fft::basic_inplace(bb, cfg.fp, cfg.direction, factorization, x);
        auto P = unscrambler(factorization);

        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));

        auto out_view =
            tensor_view(out_acc, {cfg.M, p_.N_out, K},
                        std::array<expr, 3u>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]})
                .subview(bb, slice{get_group_id(0) * cfg.Mb, mb}, slice{}, slice{k_first, kb});

        auto X1_out =
            bb.declare_assign(pointer_to(slm_out_ty), "X1_out", cast(pointer_to(slm_out_ty), X1));
        auto X1_out_view = tensor_view(std::make_shared<array_accessor>(X1_out, slm_out_ty),
                                       std::array<expr, 3u>{cfg.Mb, p_.N_out, cfg.Kb});
        auto X1_out_1d =
            X1_out_view.subview(bb, get_local_id(0), slice{0u, p_.N_out}, get_local_id(1));

        store(bb, copy_params{cfg, fph, out_view, X1_out_view, X1_out_1d, x_view, x_acc, mb, K, kb,
                              kb_odd, P});
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

void sbfft_gen::double_load(block_builder &bb, copy_params cp, int k_offset) const {
    auto X_src = cp.view.reshaped_mode(2, std::array<expr, 2u>{p_.k_stride, cp.kb})
                     .subview(bb, slice{}, slice{}, k_offset, slice{});
    copy_mbNkb_block_on_2D_grid(bb, X_src, cp.X1_view, cp.mb, p_.N_in,
                                k_offset == 1 ? cp.kb_odd : cp.kb);
}

void sbfft_gen::double_store(block_builder &bb, copy_params cp, int k_offset) const {
    auto X_dest = cp.view.reshaped_mode(2, std::array<expr, 2u>{p_.k_stride, cp.kb})
                      .subview(bb, slice{}, slice{}, k_offset, slice{});
    copy_mbNkb_block_on_2D_grid(bb, cp.X1_view, X_dest, cp.mb, p_.N_out,
                                k_offset == 1 ? cp.kb_odd : cp.kb);
}

void sbfft_gen_c2c::load(block_builder &bb, copy_params cp) const {
    copy_mbNkb_block_on_2D_grid(bb, cp.view, cp.X1_view, cp.mb, p().N_in, cp.kb);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    copy_N_block_with_permutation(bb, cp.X1_1d, cp.x_view, p().N_fft);
}

void sbfft_gen_c2c::store(block_builder &bb, copy_params cp) const {
    copy_N_block_with_permutation(bb, cp.x_view, cp.X1_1d, p().N_fft, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    copy_mbNkb_block_on_2D_grid(bb, cp.X1_view, cp.view, cp.mb, p().N_out, cp.kb);
}

void sbfft_gen_r2c_half::load(block_builder &bb, copy_params cp) const {
    copy_mbNkb_block_on_2D_grid(bb, cp.view, cp.X1_view, cp.mb, p().N_in, cp.kb);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));

    auto X1_src = cp.X1_1d.reshaped_mode(0, std::array<expr, 2u>{2, p().N_fft});
    cp.x_acc->component(0);
    copy_N_block_with_permutation(bb, X1_src.subview(bb, 0, slice{}), cp.x_view, p().N_fft);
    cp.x_acc->component(1);
    copy_N_block_with_permutation(bb, X1_src.subview(bb, 1, slice{}), cp.x_view, p().N_fft);
    cp.x_acc->component(-1);
}

void sbfft_gen_r2c_half::store(block_builder &bb, copy_params cp) const {
    postprocess(bb, cp.fph, cp.x_view, cp.X1_1d, cp.cfg.N, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    copy_mbNkb_block_on_2D_grid(bb, cp.X1_view, cp.view, cp.mb, p().N_out, cp.kb);
}

void sbfft_gen_r2c_half::postprocess(block_builder &bb, precision_helper fph,
                                     tensor_view<1u> const &y, tensor_view<1u> const &X1,
                                     std::size_t N, permutation_fun P) {
    for (std::size_t i = 0; i <= N / 4; ++i) {
        auto i_other = N / 2 - i;
        auto i_load = P(i % (N / 2));
        auto i_other_load = P(i_other % (N / 2));
        var y1 = bb.declare_assign(fph.type(2), "yi", y(i_load));
        var y2 = bb.declare_assign(fph.type(2), "yN_i", y(i_other_load));
        bb.assign(y2, init_vector(fph.type(2), {y2.s(0), -y2.s(1)}));
        var a = bb.declare_assign(fph.type(2), "a", (y2 + y1) / fph.constant(2.0));
        var b = bb.declare_assign(fph.type(2), "b", (y2 - y1) / fph.constant(2.0));
        bb.assign(b, init_vector(fph.type(2), {-b.s(1), b.s(0)}));
        auto tw = power_of_w(-i, N);
        bb.assign(b, complex_mul(fph)(b, tw));
        bb.add(X1.store(a + b, i));
        if (i != i_other) {
            expr x_other = init_vector(fph.type(2), {a.s(0) - b.s(0), b.s(1) - a.s(1)});
            bb.add(X1.store(x_other, i_other));
        }
    }
}

void sbfft_gen_c2r_half::load(block_builder &bb, copy_params cp) const {
    copy_mbNkb_block_on_2D_grid(bb, cp.view, cp.X1_view, cp.mb, p().N_in, cp.kb);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));

    preprocess(bb, cp.fph, cp.X1_1d, cp.x_view, cp.cfg.N);
}

void sbfft_gen_c2r_half::store(block_builder &bb, copy_params cp) const {
    auto X1_dest = cp.X1_1d.reshaped_mode(0, std::array<expr, 2u>{2, p().N_fft});
    cp.x_acc->component(0);
    copy_N_block_with_permutation(bb, cp.x_view, X1_dest.subview(bb, 0, slice{}), p().N_fft, cp.P);
    cp.x_acc->component(1);
    copy_N_block_with_permutation(bb, cp.x_view, X1_dest.subview(bb, 1, slice{}), p().N_fft, cp.P);
    cp.x_acc->component(-1);

    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    copy_mbNkb_block_on_2D_grid(bb, cp.X1_view, cp.view, cp.mb, p().N_out, cp.kb);
}

void sbfft_gen_c2r_half::preprocess(block_builder &bb, precision_helper fph,
                                    tensor_view<1u> const &X1, tensor_view<1u> const &x,
                                    std::size_t N) {

    for (std::size_t i = 0; i <= N / 4; ++i) {
        auto i_other = N / 2 - i;
        auto i_store = i % (N / 2);
        auto i_other_store = i_other % (N / 2);

        var x1 = bb.declare_assign(fph.type(2), "xi", X1(i));
        // ensure that imaginary part of zero-frequency term is zero
        if (i == 0) {
            bb.assign(x1.s(1), fph.zero());
        }
        var x2 = bb.declare_assign(fph.type(2), "xN_i", X1(i_other));
        bb.assign(x2, init_vector(fph.type(2), {x2.s(0), -x2.s(1)}));
        var a = bb.declare_assign(fph.type(2), "a", x1 + x2);
        var b = bb.declare_assign(fph.type(2), "b", x1 - x2);
        auto tw = power_of_w(i, N) * std::complex<double>{0.0, 1.0};
        bb.assign(b, complex_mul(fph)(b, tw));
        bb.add(x.store(a + b, i_store));
        if (i_store != i_other_store) {
            expr x_other = init_vector(fph.type(2), {a.s(0) - b.s(0), b.s(1) - a.s(1)});
            bb.add(x.store(x_other, i_other_store));
        }
    }
}

void sbfft_gen_r2c_double::load(block_builder &bb, copy_params cp) const {
    double_load(bb, cp, 0);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    cp.x_acc->component(0);
    copy_N_block_with_permutation(bb, cp.X1_1d, cp.x_view, p().N_in);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    set_k_maybe_not_written_to_zero(bb, cp.fph, cp.X1_view, p().N_in, cp.K, cp.cfg.Kb);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_load(bb, cp, 1);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    cp.x_acc->component(1);
    copy_N_block_with_permutation(bb, cp.X1_1d, cp.x_view, p().N_in);
    cp.x_acc->component(-1);
}

void sbfft_gen_r2c_double::store(block_builder &bb, copy_params cp) const {
    postprocess(bb, cp.fph, cp.x_view, cp.X1_1d, cp.cfg.N, 0, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_store(bb, cp, 0);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    postprocess(bb, cp.fph, cp.x_view, cp.X1_1d, cp.cfg.N, 1, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_store(bb, cp, 1);
}

void sbfft_gen_r2c_double::postprocess(block_builder &bb, precision_helper fph,
                                       tensor_view<1u> const &y, tensor_view<1u> const &X1,
                                       std::size_t N, int component, permutation_fun P) {
    for (std::size_t i = 0; i <= N / 2; ++i) {
        std::size_t i_other = (N - i) % N;
        var y1 = bb.declare_assign(fph.type(2), "yi", y(P(i)));
        var y2 = bb.declare_assign(fph.type(2), "yN_i", y(P(i_other)));
        bb.assign(y2, init_vector(fph.type(2), {y2.s(0), -y2.s(1)}));
        if (component == 0) {
            bb.assign(y1, (y2 + y1) / fph.constant(2.0));
        } else {
            bb.assign(y1, (y2 - y1) / fph.constant(2.0));
            bb.assign(y1, init_vector(fph.type(2), {-y1.s(1), y1.s(0)}));
        }
        bb.add(X1.store(y1, i));
        bb.add(sub_group_barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    }
}

void sbfft_gen_c2r_double::load(block_builder &bb, copy_params cp) const {
    double_load(bb, cp, 0);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    preprocess(bb, cp.fph, cp.X1_1d, cp.x_view, cp.cfg.N, 0);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    set_k_maybe_not_written_to_zero(bb, cp.fph, cp.X1_view, p().N_in, cp.K, cp.cfg.Kb);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_load(bb, cp, 1);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    preprocess(bb, cp.fph, cp.X1_1d, cp.x_view, cp.cfg.N, 1);
}

void sbfft_gen_c2r_double::store(block_builder &bb, copy_params cp) const {
    cp.x_acc->component(0);
    copy_N_block_with_permutation(bb, cp.x_view, cp.X1_1d, p().N_out, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_store(bb, cp, 0);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    cp.x_acc->component(1);
    copy_N_block_with_permutation(bb, cp.x_view, cp.X1_1d, p().N_out, cp.P);
    bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    double_store(bb, cp, 1);
    cp.x_acc->component(-1);
}

void sbfft_gen_c2r_double::preprocess(block_builder &bb, precision_helper fph,
                                      tensor_view<1u> const &X1, tensor_view<1u> const &x,
                                      std::size_t N, int component) {
    for (std::size_t i = 0; i <= N / 2; ++i) {
        std::size_t i_other = (N - i) % N;
        expr xi = X1(i);
        if (component == 0) {
            // ensure that imaginary part of zero-frequency term is zero
            if (i == 0) {
                bb.assign(x(i).s(0), xi.s(0));
                bb.assign(x(i).s(1), fph.zero());
            } else {
                bb.assign(x(i), xi);
            }
            if (i != i_other) {
                bb.assign(x(i_other), xi);
            }
        } else {
            // for i == 0 bi.s(1) must be zero
            auto bi = bb.declare_assign(fph.type(2), "bi", xi);
            bb.assign(bi, init_vector(fph.type(2), {-bi.s(1), bi.s(0)}));
            if (i == 0) {
                bb.add(add_into(x(i).s(1), bi.s(1)));
            } else {
                bb.add(add_into(x(i), bi));
            }
            if (i != i_other) {
                bb.add(subtract_from(x(i_other), bi));
                bb.assign(x(i_other),
                          init_vector(fph.type(2), {x(i_other).s(0), -x(i_other).s(1)}));
            }
        }
    }
}

} // namespace bbfft
