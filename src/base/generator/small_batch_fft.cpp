// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/generator_impl.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "generator/snippet.hpp"
#include "generator/tensor_accessor.hpp"
#include "generator/tensor_view.hpp"
#include "generator/utility.hpp"
#include "math.hpp"
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
#include <functional>
#include <sstream>
#include <utility>

using namespace clir;

namespace bbfft {

small_batch_configuration configure_small_batch_fft(configuration const &cfg,
                                                    device_info const &info) {
    auto M = cfg.shape[0];
    std::size_t N = cfg.shape[1];
    std::size_t N_slm = N;
    std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);

    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    if (is_real) {
        N_slm = N / 2 + 1;
    }

    std::size_t sgs = info.min_subgroup_size();
    if (M == 1) {
        auto register_space = info.register_space_max();
        for (auto sgs_i : info.subgroup_sizes) {
            auto required_register_space = 2 * sizeof_real * N * sgs_i;
            if (sgs < sgs_i && required_register_space < register_space / 2) {
                sgs = sgs_i;
            }
        }
    }

    std::size_t Mb = 1;
    std::size_t max_work_group_size = std::min(std::size_t(128), info.max_work_group_size);
    std::size_t work_group_size_limit = info.max_subgroup_size();
    Mb = std::min(min_power_of_2_greater_equal(M), work_group_size_limit);

    std::size_t max_compute_Kb = max_work_group_size / Mb;
    std::size_t max_slm_Kb = info.local_memory_size / (Mb * N_slm * 2 * sizeof_real);
    std::size_t max_Kb = std::min(max_compute_Kb, max_slm_Kb);
    std::size_t Kb = std::min(cfg.shape[2], max_power_of_2_less_equal(max_Kb));

    bool inplace_unsupported = is_real && Mb < M;

    auto istride = std::array<std::size_t, 3>{cfg.istride[0], cfg.istride[1], cfg.istride[2]};
    auto ostride = std::array<std::size_t, 3>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]};

    return {
        static_cast<int>(cfg.dir),   // direction
        M,                           // M
        Mb,                          // Mb
        N,                           // N
        Kb,                          // Kb
        sgs,                         // sgs
        cfg.fp,                      // precision
        cfg.type,                    // transform type
        istride,                     // istride
        ostride,                     // ostride
        inplace_unsupported,         // inplace_unsupported
        cfg.callbacks.load_function, // load_function
        cfg.callbacks.store_function // store_function
    };
}

std::string small_batch_configuration::identifier() const {
    std::ostringstream oss;
    oss << "sbfft_" << (direction < 0 ? 'm' : 'p') << std::abs(direction) << "_M" << M << "_Mb"
        << Mb << "_N" << N << "_Kb" << Kb << "_sgs" << sgs << "_f" << static_cast<int>(fp) * 8
        << '_' << to_string(type) << "_is";
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

void r2c_post(
    block_builder &bb, precision_helper fph, tensor_view<1u> const &y, tensor_view<1u> const &X1,
    std::size_t N, int component,
    std::function<std::size_t(std::size_t)> P = [](std::size_t i) { return i; }) {
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

void c2r_pre(block_builder &bb, precision_helper fph, tensor_view<1u> const &X1,
             tensor_view<1u> const &x, std::size_t N, int component) {
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

void generate_small_batch_fft(std::ostream &os, small_batch_configuration const &cfg,
                              std::string_view name) {
    auto N = cfg.N;
    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    auto N_in = cfg.type == transform_type::c2r ? N / 2 + 1 : N;
    auto N_out = cfg.type == transform_type::r2c ? N / 2 + 1 : N;
    auto N_slm = is_real ? N / 2 + 1 : N;
    auto k_stride = is_real ? 2 : 1;

    auto factorization = trial_division(N);

    auto in = var("in");
    auto out = var("out");
    auto K = var("K");

    auto fph = precision_helper{cfg.fp};
    auto in_ty = cfg.type == transform_type::r2c ? fph.type(address_space::global_t)
                                                 : fph.type(2, address_space::global_t);
    auto out_ty = cfg.type == transform_type::c2r ? fph.type(address_space::global_t)
                                                  : fph.type(2, address_space::global_t);
    auto slm_in_ty = cfg.type == transform_type::r2c ? fph.type(address_space::local_t)
                                                     : fph.type(2, address_space::local_t);
    auto slm_out_ty = cfg.type == transform_type::c2r ? fph.type(address_space::local_t)
                                                      : fph.type(2, address_space::local_t);

    auto fb = kernel_builder{name.empty() ? cfg.identifier() : std::string(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(generic_ulong(), K);
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Kb), 1));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));

    auto xy_ty = data_type(array_of(fph.type(2), N));

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
        auto k_first =
            bb.declare_assign(generic_size(), "k_first", get_group_id(1) * k_stride * cfg.Kb);

        expr mb = nullptr;
        if (cfg.M < cfg.Mb) {
            mb = cfg.M;
        } else if (cfg.M % cfg.Mb == 0) {
            mb = cfg.Mb;
        } else {
            mb = bb.declare_assign(generic_uint(), "mb", cfg.M - get_group_id(0) * cfg.Mb);
        }
        auto kb = bb.declare_assign(generic_uint(), "kb", (K - k_first - 1) / k_stride + 1);
        expr kb_odd = nullptr;
        if (is_real) {
            kb_odd = bb.declare_assign(generic_uint(), "kb_odd", kb - K % 2);
        }
        bb.assign(kb, select(cfg.Kb, kb, kb < cfg.Kb));
        if (is_real) {
            bb.assign(kb_odd, select(cfg.Kb, kb_odd, kb_odd < cfg.Kb));
        }

        auto in_view =
            tensor_view(in_acc, {cfg.M, N_in, K},
                        std::array<expr, 3u>{cfg.istride[0], cfg.istride[1], cfg.istride[2]})
                .subview(bb, slice{get_group_id(0) * cfg.Mb, mb}, slice{}, slice{k_first, kb});

        auto X1 = bb.declare(array_of(fph.type(2, address_space::local_t), cfg.Kb * N_slm * cfg.Mb),
                             "X1");
        auto X1_in =
            bb.declare_assign(pointer_to(slm_in_ty), "X1_in", cast(pointer_to(slm_in_ty), X1));

        auto X1_in_view = tensor_view(std::make_shared<array_accessor>(X1_in, slm_in_ty),
                                      std::array<expr, 3u>{cfg.Mb, N_in, cfg.Kb});
        auto X1_in_1d = X1_in_view.subview(bb, get_local_id(0), slice{0u, N_in}, get_local_id(1));

        auto const load = [&](block_builder &bb, int k_offset) {
            auto X_src = in_view.reshaped_mode(2, std::array<expr, 2u>{k_stride, kb})
                             .subview(bb, slice{}, slice{}, k_offset, slice{});
            copy_mbNkb_block_on_2D_grid(bb, X_src, X1_in_view, mb, N_in,
                                        k_offset == 1 ? kb_odd : kb);
        };

        auto x = bb.declare(xy_ty, "x");
        auto x_acc = std::make_shared<array_accessor>(x, xy_ty);
        auto x_view = tensor_view(x_acc, std::array<expr, 1u>{N});

        if (cfg.type == transform_type::c2r) {
            load(bb, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            c2r_pre(bb, fph, X1_in_1d, x_view, N, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            set_k_maybe_not_written_to_zero(bb, fph, X1_in_view, N_in, K, cfg.Kb);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load(bb, 1);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            c2r_pre(bb, fph, X1_in_1d, x_view, N, 1);
        } else if (cfg.type == transform_type::r2c) {
            load(bb, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            x_acc->component(0);
            copy_N_block_with_permutation(bb, X1_in_1d, x_view, N_in);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            set_k_maybe_not_written_to_zero(bb, fph, X1_in_view, N_in, K, cfg.Kb);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load(bb, 1);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            x_acc->component(1);
            copy_N_block_with_permutation(bb, X1_in_1d, x_view, N_in);
            x_acc->component(-1);
        } else {
            copy_mbNkb_block_on_2D_grid(bb, in_view, X1_in_view, mb, N_in, kb);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            copy_N_block_with_permutation(bb, X1_in_1d, x_view, N_in);
        }

        generate_fft::basic_inplace(bb, cfg.fp, cfg.direction, factorization, x);
        auto P = unscrambler(factorization);

        auto out_view =
            tensor_view(out_acc, {cfg.M, N_out, K},
                        std::array<expr, 3u>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]})
                .subview(bb, slice{get_group_id(0) * cfg.Mb, mb}, slice{}, slice{k_first, kb});

        auto X1_out =
            bb.declare_assign(pointer_to(slm_out_ty), "X1_out", cast(pointer_to(slm_out_ty), X1));
        auto X1_out_view = tensor_view(std::make_shared<array_accessor>(X1_out, slm_out_ty),
                                       std::array<expr, 3u>{cfg.Mb, N_out, cfg.Kb});
        auto X1_out_1d =
            X1_out_view.subview(bb, get_local_id(0), slice{0u, N_out}, get_local_id(1));

        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        auto const store = [&](block_builder &bb, int k_offset) {
            auto X_dest = out_view.reshaped_mode(2, std::array<expr, 2u>{k_stride, kb})
                              .subview(bb, slice{}, slice{}, k_offset, slice{});
            copy_mbNkb_block_on_2D_grid(bb, X1_out_view, X_dest, mb, N_out,
                                        k_offset == 1 ? kb_odd : kb);
        };
        if (cfg.type == transform_type::r2c) {
            r2c_post(bb, fph, x_view, X1_out_1d, N, 0, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            r2c_post(bb, fph, x_view, X1_out_1d, N, 1, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 1);
        } else if (cfg.type == transform_type::c2r) {
            x_acc->component(0);
            copy_N_block_with_permutation(bb, x_view, X1_out_1d, N_out, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            x_acc->component(1);
            copy_N_block_with_permutation(bb, x_view, X1_out_1d, N_out, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 1);
            x_acc->component(-1);
        } else {
            copy_N_block_with_permutation(bb, x_view, X1_out_1d, N_out, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            copy_mbNkb_block_on_2D_grid(bb, X1_out_view, out_view, mb, N_out, kb);
        }
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

} // namespace bbfft
