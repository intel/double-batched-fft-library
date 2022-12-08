// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/generator.hpp"
#include "bbfft/tensor_indexer.hpp"
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

#include <functional>
#include <utility>

using namespace clir;

namespace bbfft {

small_batch_configuration configure_small_batch_fft(configuration const &cfg, device_info info) {
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
        auto register_space = info.register_space();
        for (std::size_t i = 0; i < info.num_subgroup_sizes; ++i) {
            auto sgs_i = info.subgroup_sizes[i];
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

void r2c_post(
    block_builder &bb, precision_helper fph, expr X1, expr y, std::size_t Mb, std::size_t N,
    std::size_t Kb, int component,
    std::function<std::size_t(std::size_t)> P = [](std::size_t i) { return i; }) {
    auto X1_indexer = tensor_indexer<expr, 3, layout::col_major>({Mb, N / 2 + 1, Kb});
    for (std::size_t i = 0; i <= N / 2; ++i) {
        std::size_t i_other = (N - i) % N;
        var y1 = bb.declare_assign(fph.type(2), "yi", y[P(i)]);
        var y2 = bb.declare_assign(fph.type(2), "yN_i", y[P(i_other)]);
        bb.assign(y2, init_vector(fph.type(2), {y2.s(0), -y2.s(1)}));
        if (component == 0) {
            bb.assign(y1, (y2 + y1) / fph.constant(2.0));
        } else {
            bb.assign(y1, (y2 - y1) / fph.constant(2.0));
            bb.assign(y1, init_vector(fph.type(2), {-y1.s(1), y1.s(0)}));
        }
        bb.assign(X1[X1_indexer(get_local_id(0), i, get_local_id(1))], y1);
        bb.add(sub_group_barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    }
}

void c2r_pre(block_builder &bb, precision_helper fph, expr X1, expr x, std::size_t Mb,
             std::size_t N, std::size_t Kb, int component) {
    auto X1_indexer = tensor_indexer<expr, 3, layout::col_major>({Mb, N / 2 + 1, Kb});
    for (std::size_t i = 0; i <= N / 2; ++i) {
        std::size_t i_other = (N - i) % N;
        expr xi = X1[X1_indexer(get_local_id(0), i, get_local_id(1))];
        if (component == 0) {
            // ensure that imaginary part of zero-frequency term is zero
            if (i == 0) {
                bb.assign(x[i].s(0), xi.s(0));
                bb.assign(x[i].s(1), fph.zero());
            } else {
                bb.assign(x[i], xi);
            }
            if (i != i_other) {
                bb.assign(x[i_other], xi);
            }
        } else {
            // for i == 0 bi.s(1) must be zero
            auto bi = bb.declare_assign(fph.type(2), "bi", xi);
            bb.assign(bi, init_vector(fph.type(2), {-bi.s(1), bi.s(0)}));
            if (i == 0) {
                bb.add(add_into(x[i].s(1), bi.s(1)));
            } else {
                bb.add(add_into(x[i], bi));
            }
            if (i != i_other) {
                bb.add(subtract_from(x[i_other], bi));
                bb.assign(x[i_other],
                          init_vector(fph.type(2), {x[i_other].s(0), -x[i_other].s(1)}));
            }
        }
    }
}

void load_store_slm(block_builder &bb, precision_helper fph, expr X,
                    std::array<std::size_t, 3> const &stride, expr X1, std::size_t M, expr K,
                    std::size_t Mb, std::size_t N, std::size_t Kb, std::size_t k_stride,
                    std::size_t k_offset, bool store, char const *callback = nullptr) {
    auto m_local = bb.declare_assign(generic_size(), "m_local", get_local_id(0));
    auto k_local = bb.declare_assign(generic_size(), "k_local", get_local_id(1));
    auto m_group = bb.declare_assign(generic_size(), "m_group", get_group_id(0));
    auto X_indexer =
        tensor_indexer<expr, 3, layout::col_major>({1, 1, 1}, {stride[0], stride[1], stride[2]});
    auto X1_indexer = tensor_indexer<expr, 3, layout::col_major>({Mb, N, Kb});
    auto k_first = bb.declare_assign(generic_size(), "k_first", get_group_id(1) * k_stride * Kb);

    auto i = var("i");
    if (!store && k_offset == 1) {
        auto k_maybe_not_written =
            bb.declare_assign(generic_ulong(), "k_maybe_not_written", K % Kb);
        bb.add(for_loop_builder(declaration_assignment(generic_uint(), i, k_local), i < N,
                                add_into(i, Kb))
                   .body([&](block_builder &bb) {
                       bb.assign(X1[X1_indexer(m_local, i, k_maybe_not_written)], fph.zero());
                   })
                   .get_product());
        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
    }

    auto mnk = var("mnk");
    auto X_offset = var("X_offset");
    auto const load_store = [&](block_builder &bb, expr x1, expr n_in, expr k_in) {
        auto X_idx = X_offset + X_indexer(0, n_in, k_in);
        if (callback) {
            bb.assign(mnk[1], n_in);
            bb.assign(mnk[2], k_first + k_in);
            if (store) {
                bb.add(call_external(callback, {X, X_idx, x1, mnk}));
            } else {
                bb.assign(x1, call_external(callback, {X, X_idx, mnk}));
            }
        } else {
            auto x = X[X_idx];
            if (store) {
                bb.assign(x, x1);
            } else {
                bb.assign(x1, x);
            }
        }
    };

    // ceil((K-k_first-k_offset)/2)
    auto kb = bb.declare_assign(generic_uint(), "kb", (K - k_first - k_offset - 1) / k_stride + 1);
    bb.assign(kb, select(Kb, kb, kb < Kb));
    auto make_load = [&](expr mb) {
        auto range_check =
            if_selection_builder(m_local < mb && k_local < kb).then([&](block_builder &bb) {
                auto base_idx =
                    bb.declare_assign(generic_uint(), "base_idx", m_local + k_local * mb);
                auto m_in = bb.declare_assign(generic_uint(), "m_in", base_idx % mb);

                bb.declare_assign(generic_size(), X_offset,
                                  X_indexer(m_group * Mb + m_in, 0, k_first));
                if (callback) {
                    bb.declare(generic_uint(3), mnk);
                    bb.assign(mnk[0], m_group * Mb + m_in);
                }
                for (std::size_t n_local = 0; n_local < N; ++n_local) {
                    auto idx =
                        bb.declare_assign(generic_uint(), "idx", base_idx + n_local * (mb * kb));
                    auto n_in = idx / mb % N;
                    auto k_in = idx / (mb * N);
                    auto x1 = X1[X1_indexer(m_in, n_in, k_in)];
                    load_store(bb, x1, n_in, k_offset + k_stride * k_in);
                }
            });
        return range_check.get_product();
    };
    if (M % Mb != 0) {
        auto mb = bb.declare_assign(generic_uint(), "mb", M - m_group * Mb);
        bb.assign(mb, select(Mb, mb, mb < Mb));
        auto check_mb = if_selection_builder(mb == Mb)
                            .then([&](block_builder &bb) { bb.add(make_load(Mb)); })
                            .otherwise([&](block_builder &bb) { bb.add(make_load(mb)); });
        bb.add(check_mb.get_product());
    } else {
        bb.add(make_load(Mb));
    }
}

void load_store_register(
    block_builder &bb, expr X1, expr x, std::size_t Mb, std::size_t N, std::size_t Kb, bool store,
    int component = -1,
    std::function<std::size_t(std::size_t)> P = [](std::size_t i) { return i; }) {
    auto X1_indexer = tensor_indexer<expr, 3, layout::col_major>({Mb, N, Kb});
    for (std::size_t j1 = 0; j1 < N; ++j1) {
        expr x1 = X1[X1_indexer(get_local_id(0), j1, get_local_id(1))];
        expr x2 = x[P(j1)];
        if (component >= 0) {
            x2 = x2.s(component);
        }
        if (store) {
            std::swap(x1, x2);
        }
        bb.assign(x2, x1);
    }
}

void generate_small_batch_fft(std::ostream &os, std::string name,
                              small_batch_configuration const &cfg) {
    auto N = cfg.N;
    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    auto N_in = cfg.type == transform_type::c2r ? N / 2 + 1 : N;
    auto N_out = cfg.type == transform_type::r2c ? N / 2 + 1 : N;
    auto N_slm = is_real ? N / 2 + 1 : N;
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

    auto fb = function_builder{std::move(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(generic_ulong(), K);
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Kb), 1));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));

    auto xy_ty = data_type(array_of(fph.type(2), N));

    // load in SLM from global memory, load transposed in registers from SLM
    fb.body([&](block_builder &bb) {
        auto kk = bb.declare_assign(generic_size(), "kk", get_global_id(1));
        auto mm = bb.declare_assign(generic_size(), "mm", get_global_id(0));

        auto X1 = bb.declare(array_of(fph.type(2, address_space::local_t), cfg.Kb * N_slm * cfg.Mb),
                             "X1");
        auto X1_in =
            bb.declare_assign(pointer_to(slm_in_ty), "X1_in", cast(pointer_to(slm_in_ty), X1));

        auto const load = [&](block_builder &bb, int stride, int component) {
            load_store_slm(bb, fph, in, cfg.istride, X1_in, cfg.M, K, cfg.Mb, N_in, cfg.Kb, stride,
                           component, false, cfg.load_function);
        };

        auto x = bb.declare(xy_ty, "x");
        if (cfg.type == transform_type::c2r) {
            load(bb, 2, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            c2r_pre(bb, fph, X1_in, x, cfg.Mb, N, cfg.Kb, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load(bb, 2, 1);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            c2r_pre(bb, fph, X1_in, x, cfg.Mb, N, cfg.Kb, 1);
        } else if (cfg.type == transform_type::r2c) {
            load(bb, 2, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load_store_register(bb, X1_in, x, cfg.Mb, N_in, cfg.Kb, false, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load(bb, 2, 1);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load_store_register(bb, X1_in, x, cfg.Mb, N_in, cfg.Kb, false, 1);
        } else {
            load(bb, 1, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load_store_register(bb, X1_in, x, cfg.Mb, N_in, cfg.Kb, false);
        }

        generate_fft::basic_inplace(bb, cfg.fp, cfg.direction, factorization, x);
        auto P = unscrambler(factorization);

        auto X1_out =
            bb.declare_assign(pointer_to(slm_out_ty), "X1_out", cast(pointer_to(slm_out_ty), X1));

        bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        auto const store = [&](block_builder &bb, int stride, int component) {
            load_store_slm(bb, fph, out, cfg.ostride, X1_out, cfg.M, K, cfg.Mb, N_out, cfg.Kb,
                           stride, component, true, cfg.store_function);
        };
        if (cfg.type == transform_type::r2c) {
            r2c_post(bb, fph, X1_out, x, cfg.Mb, N, cfg.Kb, 0, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 2, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            r2c_post(bb, fph, X1_out, x, cfg.Mb, N, cfg.Kb, 1, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 2, 1);
        } else if (cfg.type == transform_type::c2r) {
            load_store_register(bb, X1_out, x, cfg.Mb, N_out, cfg.Kb, true, 0, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 2, 0);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            load_store_register(bb, X1_out, x, cfg.Mb, N_out, cfg.Kb, true, 1, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 2, 1);
        } else {
            load_store_register(bb, X1_out, x, cfg.Mb, N_out, cfg.Kb, true, -1, P);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            store(bb, 1, 0);
        }
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

} // namespace bbfft
