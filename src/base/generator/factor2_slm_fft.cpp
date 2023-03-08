// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"
#include "bbfft/detail/generator_impl.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "generator/accessor.hpp"
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

factor2_slm_configuration configure_factor2_slm_fft(configuration const &cfg, device_info info) {
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
    bool external_buffer = max_slm_Mb == 0;
    std::size_t max_Mb = external_buffer ? max_compute_Mb : std::min(max_compute_Mb, max_slm_Mb);
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
        external_buffer,             // external_buffer
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
    oss << "eb" << external_buffer;
    oss << "_in" << inplace_unsupported;
    if (load_function) {
        oss << "_" << load_function;
    }
    if (store_function) {
        oss << "_" << store_function;
    }
    return oss.str();
}

class global_tensor {
  public:
    global_tensor(std::array<expr, 3u> stride, precision_helper fph, bool is_real)
        : indexer_({}, std::move(stride)), fph_(fph), is_real_(is_real) {}
    virtual ~global_tensor() {}

    auto &indexer() const { return indexer_; }
    auto fph() const { return fph_; }
    bool is_real() const { return is_real_; }
    auto type() const {
        return is_real_ ? fph_.type(address_space::global_t)
                        : fph_.type(2, address_space::global_t);
    }

    virtual auto subflat(block_builder &bb, expr mm, expr kk) const
        -> std::unique_ptr<accessor<1u>> = 0;
    virtual auto sub2d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<2u>> = 0;
    virtual auto access4d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<4u>> = 0;

  private:
    tensor_indexer<expr, 3u, layout::col_major> indexer_;
    precision_helper fph_;
    bool is_real_;
};

class global_tensor_callback : public global_tensor {
  public:
    global_tensor_callback(expr X, std::array<expr, 3u> stride, precision_helper fph,
                           bool is_real = false, char const *load = nullptr,
                           char const *store = nullptr)
        : global_tensor(std::move(stride), fph, is_real), X_(std::move(X)), load_(load),
          store_(store) {}

    auto subflat(block_builder &bb, expr mm, expr kk) const
        -> std::unique_ptr<accessor<1u>> override {
        return sub<1u>(bb, mm, kk, {this->indexer().stride(1)},
                       [mm, kk](std::array<expr, 1u> const &idx) -> expr {
                           return init_vector(generic_uint(3), {mm, idx[0], kk});
                       });
    }

    auto sub2d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<2u>> override {
        auto stride1 = this->indexer().stride(1);
        return sub<2u>(
            bb, mm, kk, {stride1, stride1 * N_stride},
            [mm, N_stride, kk](std::array<expr, 2u> const &idx) -> expr {
                return init_vector(generic_uint(3), {mm, idx[0] + idx[1] * N_stride, kk});
            });
    }

    auto access4d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<4u>> override {
        auto stride = this->indexer().stride();
        return sub<4u>(bb, mm, kk, {stride[0], stride[1], stride[1] * N_stride, stride[2]},
                       [mm, N_stride, kk](std::array<expr, 4u> const &idx) -> expr {
                           return init_vector(
                               generic_uint(3),
                               {mm + idx[0], idx[1] + idx[2] * N_stride, kk + idx[3]});
                       });
    }

  private:
    template <unsigned int D>
    auto sub(block_builder &bb, expr mm, expr kk, std::array<expr, D> stride,
             typename callback_accessor<D>::mnk_function mnk) const
        -> std::unique_ptr<accessor<D>> {
        auto offset = bb.declare_assign(generic_size(), "offset", this->indexer()(mm, 0, kk));
        return std::make_unique<callback_accessor<D>>(X_, std::array<expr, D>{}, std::move(stride),
                                                      offset, std::move(mnk), load_, store_);
    }

    expr X_;
    char const *load_ = nullptr;
    char const *store_ = nullptr;
};

class global_tensor_array : public global_tensor {
  public:
    global_tensor_array(expr X, std::array<expr, 3u> stride, precision_helper fph,
                        bool is_real = false)
        : global_tensor(std::move(stride), fph, is_real), X_(X) {}

    auto subflat(block_builder &bb, expr mm, expr kk) const
        -> std::unique_ptr<accessor<1u>> override {
        return sub<1u>(bb, mm, kk, {this->indexer().stride(1)});
    }

    auto sub2d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<2u>> override {
        auto stride1 = this->indexer().stride(1);
        return sub<2u>(bb, mm, kk, {stride1, stride1 * N_stride});
    }

    auto access4d(block_builder &bb, expr mm, expr kk, std::size_t N_stride) const
        -> std::unique_ptr<accessor<4u>> override {
        auto stride = this->indexer().stride();
        return sub<4u>(bb, mm, kk, {stride[0], stride[1], stride[1] * N_stride, stride[2]});
    }

  private:
    template <unsigned int D>
    auto sub(block_builder &bb, expr mm, expr kk, std::array<expr, D> stride) const
        -> std::unique_ptr<accessor<D>> {
        return std::make_unique<array_accessor<D>>(
            bb.declare_assign(pointer_to(this->type()), "sub", X_ + this->indexer()(mm, 0, kk)),
            std::array<expr, D>{}, std::move(stride));
    }

    expr X_;
};

auto make_global_tensor(expr X, std::array<std::size_t, 3u> stride, precision_helper fph,
                        bool is_real, char const *load, char const *store)
    -> std::unique_ptr<global_tensor> {
    std::array<expr, 3u> estride = {stride[0], stride[1], stride[2]};
    if (load || store) {
        return std::make_unique<global_tensor_callback>(std::move(X), std::move(estride), fph,
                                                        is_real, load, store);
    }
    return std::make_unique<global_tensor_array>(std::move(X), std::move(estride), fph, is_real);
}

class local_tensor : public array_accessor<3u> {
  public:
    local_tensor(expr X, std::array<expr, 3u> shape, data_type ty)
        : array_accessor<3u>(std::move(X), std::move(shape)), ty_(std::move(ty)) {}

    auto subflat(block_builder &bb) const {
        return array_accessor<1u>(sub(bb, 0), {}, {this->indexer().shape(0)});
    }

    auto sub1d(block_builder &bb, expr n2, std::size_t N_stride) const {
        auto shape0 = this->indexer().shape(0);
        return array_accessor<1u>(sub(bb, n2), {}, {shape0 * N_stride});
    }

    auto sub2d(block_builder &bb, std::size_t N_stride) const {
        auto shape0 = this->indexer().shape(0);
        return array_accessor<2u>(sub(bb, 0), {}, {shape0, shape0 * N_stride});
    }

    auto access4d(block_builder &, std::size_t N_stride) const {
        auto shape = this->indexer().shape();
        return array_accessor<4u>(this->x(), {shape[0], N_stride, shape[1] / N_stride, shape[2]});
    }

  private:
    auto sub(block_builder &bb, expr n2) const -> expr {
        return bb.declare_assign(pointer_to(ty_), "X1_sub",
                                 this->x() + this->indexer()(get_local_id(0), n2, get_local_id(2)));
    }

    data_type ty_;
};

void load_store_global(
    block_builder &bb, global_tensor const &X, accessor<1u> const &x, expr mm, std::size_t M,
    expr kk, expr K, expr j1, std::size_t N_loop, std::size_t N_stride, bool store,
    std::function<std::size_t(std::size_t)> P = [](std::size_t i) { return i; }) {
    if (X.is_real()) {
        auto load_store = [&](block_builder &bb, std::unique_ptr<accessor<2u>> sub, int component) {
            for (std::size_t j2 = 0; j2 < N_loop; ++j2) {
                if (store) {
                    bb.add(sub->store({j1, j2}, x(P(j2)).s(component)));
                } else {
                    bb.assign(x({P(j2)}).s(component), (*sub)(j1, j2));
                    if (component == 0) {
                        bb.assign(x({P(j2)}).s(1), X.fph().zero());
                    }
                }
            }
        };
        bb.add(if_selection_builder(mm < M)
                   .then([&](block_builder &bb) {
                       bb.add(if_selection_builder(2 * kk < K)
                                  .then([&](block_builder &bb) {
                                      load_store(bb, X.sub2d(bb, mm, 2 * kk, N_stride), 0);
                                  })
                                  .get_product());
                       bb.add(if_selection_builder(2 * kk + 1 < K)
                                  .then([&](block_builder &bb) {
                                      load_store(bb, X.sub2d(bb, mm, 2 * kk + 1, N_stride), 1);
                                  })
                                  .get_product());
                   })
                   .get_product());
    } else {
        auto range_check = if_selection_builder(mm < M && kk < K).then([&](block_builder &bb) {
            auto sub = X.sub2d(bb, mm, kk, N_stride);
            for (std::size_t j2 = 0; j2 < N_loop; ++j2) {
                if (store) {
                    bb.add(sub->store({j1, j2}, x(P(j2))));
                } else {
                    bb.add(x.store({P(j2)}, (*sub)(j1, j2)));
                }
            }
        });
        bb.add(range_check.get_product());
    }
}

/*template <typename T>
void load_store_global_slm(block_builder &bb, global_tensor<T> const &X, local_tensor<T> X1, expr K,
                           std::size_t Mb, std::size_t Nb, std::size_t Kb, std::size_t N1,
std::size_t N2, bool is_X_real, bool store, bool transpose) { auto k_local =
bb.declare_assign(generic_size(), "k_local", get_local_id(2)); auto n_local =
bb.declare_assign(generic_size(), "n_local", get_local_id(1)); auto m_local =
bb.declare_assign(generic_size(), "m_local", get_local_id(0)); auto k_group =
bb.declare_assign(generic_size(), "k_group", get_group_id(2)); auto m_group =
bb.declare_assign(generic_size(), "m_group", get_group_id(0)); short N_loop = N1 * N2 / Nb; auto
N1_in = N1; if (is_X_real) { N1_in *= 2;
    }
    auto kb = bb.declare_assign(generic_uint(), "kb", K - k_group * Kb);
    bb.assign(kb, select(Kb, kb, kb < Kb));
    auto range_check = if_selection_builder(k_local < kb).then([&](block_builder &bb) {
        auto X_sub = X.access4d(bb, m_group * Mb, k_group * Kb, N1_in);
        auto X1_sub = X1.access4d(bb, transpose ? N2 : N1);
        auto base_idx = bb.declare_assign(generic_uint(), "base_idx",
                                          m_local + n_local * Mb + k_local * (Mb * Nb));
        auto m_in = bb.declare_assign(generic_uint(), "m_in", base_idx % Mb);
        auto m = bb.declare_assign(generic_uint(), "m",
                                   is_X_real ? (base_idx % Mb) / Mb + m_in * 2 : base_idx % Mb);
        for (short i = 0; i < N_loop; ++i) {
            auto idx = bb.declare_assign(generic_uint(), "idx", base_idx + i * (Mb * Nb * kb));
            auto j1_in = (idx / Mb) % N1_in;
            auto j2_in = (idx / (Mb * N1_in)) % N2;
            auto k_in = idx / (Mb * N1_in * N2);
            auto j1 = (idx / Mb) % N1;
            auto j2 = (idx / (Mb * N1)) % N2;
            if (transpose) {
                std::swap(j1, j2);
            }
            auto k = idx / (Mb * N1 * N2);

            if (store) {
                bb.add(X_sub->store({m_in, j1_in, j2_in, k_in}, X1_sub({m, j1, j2, k})));
            } else {
                bb.add(X1_sub.store({m, j1, j2, k}, (*X_sub)({m_in, j1_in, j2_in, k_in})));
            }
        }
    });
    bb.add(range_check.get_product());
}*/

void load_store_slm(
    block_builder &bb, local_tensor const &X, expr x, expr j1, std::size_t N_loop,
    std::size_t N_stride, bool store, bool transpose,
    std::function<std::size_t(std::size_t)> P = [](std::size_t i) { return i; }) {
    auto sub = X.sub2d(bb, N_stride);
    for (std::size_t i = 0; i < N_loop; ++i) {
        expr x1 = transpose ? sub({i, j1}) : sub({j1, i});
        if (store) {
            bb.assign(x1, x[P(i)]);
        } else {
            bb.assign(x[P(i)], x1);
        }
    }
}

void r2c_post_i(block_builder &bb, precision_helper fph, expr i, std::size_t N,
                accessor<1u> const &x, accessor<1u> const &ya, accessor<1u> const &yb) {
    expr i_other = (N - i) % N;
    auto number_type = fph.type(2);
    var y1 = bb.declare_assign(number_type, "yi", x(i));
    var y2 = bb.declare_assign(number_type, "yN_i", x(i_other));
    bb.assign(y2, init_vector(number_type, {y2.s(0), -y2.s(1)}));
    var tmp = bb.declare_assign(number_type, "tmp", (y2 + y1) / fph.constant(2.0));
    bb.add(ya.store({i}, tmp));
    if (yb.store({i}, tmp)) {
        bb.assign(tmp, (y2 - y1) / fph.constant(2.0));
        bb.assign(tmp, init_vector(number_type, {-tmp.s(1), tmp.s(0)}));
        bb.add(yb.store({i}, tmp));
    }
}

void c2r_pre_i(block_builder &bb, precision_helper fph, expr i, std::size_t N1, std::size_t N2,
               accessor<1u> const &xa, accessor<1u> const &xb, accessor<1u> const &y) {
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
    bb.add(y.store({i_out}, ai + bi));
    bb.add(if_selection_builder(i_other < N1 * N2)
               .then([&](block_builder &bb) {
                   auto tmp = bb.declare_assign(number_type, "tmp", ai - bi);
                   bb.add(y.store({i_other_out}, init_vector(number_type, {tmp.s(0), -tmp.s(1)})));
               })
               .get_product());
}

void generate_factor2_slm_fft(std::ostream &os, factor2_slm_configuration const &cfg,
                              std::string_view name) {
    std::size_t N1 = cfg.N1;
    std::size_t N2 = cfg.N2;
    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    std::size_t N_slm = is_real ? N1 * N2 + 1 : N1 * N2;

    auto in = var("in");
    auto out = var("out");
    auto twiddle = var("twiddle");
    auto K = var("K");
    auto X1 = var("X1");

    auto X_in = make_global_tensor(in, cfg.istride, cfg.fp, cfg.type == transform_type::r2c,
                                   cfg.load_function, nullptr);
    auto X_out = make_global_tensor(out, cfg.ostride, cfg.fp, cfg.type == transform_type::c2r,
                                    nullptr, cfg.store_function);

    auto fph = precision_helper{cfg.fp};
    auto in_ty = cfg.type == transform_type::r2c ? fph.type(address_space::global_t)
                                                 : fph.type(2, address_space::global_t);
    auto out_ty = cfg.type == transform_type::c2r ? fph.type(address_space::global_t)
                                                  : fph.type(2, address_space::global_t);

    auto slm_shape = cfg.external_buffer ? std::array<expr, 3>{cfg.M, N_slm, K}
                                         : std::array<expr, 3>{cfg.Mb, N_slm, cfg.Kb};
    auto slm_ty = cfg.external_buffer ? fph.type(2, address_space::global_t)
                                      : fph.type(2, address_space::local_t);
    auto X_slm = local_tensor(X1, slm_shape, slm_ty);

    auto fft_inplace = &generate_fft::basic_inplace;

    auto fb = kernel_builder{name.empty() ? cfg.identifier() : std::string(name)};
    fb.argument(pointer_to(in_ty), in);
    fb.argument(pointer_to(out_ty), out);
    fb.argument(pointer_to(fph.type(2, address_space::constant_t)), twiddle);
    fb.argument(generic_ulong(), K);
    if (cfg.external_buffer) {
        fb.argument(pointer_to(slm_ty), X1);
    }
    fb.attribute(reqd_work_group_size(static_cast<int>(cfg.Mb), static_cast<int>(cfg.Nb),
                                      static_cast<int>(cfg.Kb)));
    fb.attribute(intel_reqd_sub_group_size(static_cast<int>(cfg.sgs)));
    fb.body([&](block_builder &bb) {
        if (!cfg.external_buffer) {
            bb.declare(array_of(slm_ty, cfg.Kb * N_slm * cfg.Mb), X1);
        } else {
            auto slm_offset = tensor_indexer<expr, 3, layout::col_major>(slm_shape);
            bb.add(add_into(X1, slm_offset(get_group_id(0) * cfg.Mb, 0, get_group_id(2) * cfg.Kb)));
        }
        auto kk = bb.declare_assign(generic_size(), "kk", get_global_id(2));
        auto mm = bb.declare_assign(generic_size(), "mm", get_global_id(0));
        auto n_local = bb.declare_assign(generic_size(), "n_local", get_local_id(1));

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
        if (cfg.type == transform_type::c2r) {
            auto load_fac2 = [&](block_builder &bb, std::unique_ptr<accessor<1u>> in_sub_b) {
                auto in_sub_a = X_in->subflat(bb, mm, 2 * kk);
                auto slm_sub = X_slm.subflat(bb);
                auto i = var("i");
                bb.add(for_loop_builder(declaration_assignment(generic_short(), i, n_local),
                                        i <= N1 * N2 / 2, add_into(i, cfg.Nb))
                           .body([&](block_builder &bb) {
                               c2r_pre_i(bb, cfg.fp, i, N1, N2, *in_sub_a, *in_sub_b, slm_sub);
                           })
                           .get_product());
            };
            bb.add(if_selection_builder(mm < cfg.M && 2 * kk + 1 < K)
                       .then([&](block_builder &bb) {
                           load_fac2(bb, X_in->subflat(bb, mm, 2 * kk + 1));
                       })
                       .otherwise([&](block_builder &bb) {
                           bb.add(if_selection_builder(mm < cfg.M && 2 * kk < K)
                                      .then([&](block_builder &bb) {
                                          load_fac2(bb,
                                                    std::make_unique<zero_accessor<1u>>(cfg.fp));
                                      })
                                      .get_product());
                       })
                       .get_product());
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            auto compute_fac1 = [&](block_builder &bb) {
                auto x = bb.declare(data_type(array_of(fph.type(2), N2)), "x");
                load_store_slm(bb, X_slm, x, j1, N2, N2, false, true);
                var tw_n2 = var("tw_n2");
                bb.declare_assign(pointer_to(fph.type(2, address_space::constant_t)), tw_n2,
                                  twiddle + j1 * N2);
                auto factor = trial_division(N2);
                fft_inplace(bb, cfg.fp, cfg.direction, factor, x, tw_n2);
                load_store_slm(bb, X_slm, x, j1, N2, N2, true, true, unscrambler(factor));
            };
            parallel_n2(bb, j1, N1, compute_fac1);
        } else {
            // load_store_global_slm<T>(bb, *X_in, X_slm, K, cfg.Mb, cfg.Nb, cfg.Kb,
            // N1, N2, cfg.type == transform_type::r2c, false, true);
            auto compute_fac1 = [&](block_builder &bb) {
                auto x = bb.declare(data_type(array_of(fph.type(2), N2)), "x");
                load_store_global(bb, *X_in, array_accessor<1u>(x, {}), mm, cfg.M, kk, K, j1, N2,
                                  N1, false);
                // load_store_slm(bb, X_slm, x, j1, N2, N2, false, true);
                var tw_n2 = var("tw_n2");
                bb.declare_assign(pointer_to(fph.type(2, address_space::constant_t)), tw_n2,
                                  twiddle + j1 * N2);
                auto factor = trial_division(N2);
                fft_inplace(bb, cfg.fp, cfg.direction, factor, x, tw_n2);
                load_store_slm(bb, X_slm, x, j1, N2, N2, true, true, unscrambler(factor));
            };
            parallel_n2(bb, j1, N1, compute_fac1);
        }
        if (cfg.external_buffer) {
            bb.add(barrier(cl_mem_fence_flags::CLK_GLOBAL_MEM_FENCE));
        } else {
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
        }
        auto n2 = bb.declare(generic_short(), "n2");
        if (cfg.type == transform_type::r2c) {
            auto compute_fac2 = [&](block_builder &bb) {
                auto x = bb.declare(data_type(array_of(fph.type(2), N1)), "x");
                load_store_slm(bb, X_slm, x, n2, N1, N2, false, false);
                auto factor = trial_division(N1);
                fft_inplace(bb, cfg.fp, cfg.direction, factor, x, nullptr);
                load_store_slm(bb, X_slm, x, n2, N1, N2, true, false, unscrambler(factor));
            };
            parallel_n2(bb, n2, N2, compute_fac2);
            bb.add(barrier(cl_mem_fence_flags::CLK_LOCAL_MEM_FENCE));
            auto store_fac = [&](block_builder &bb, std::unique_ptr<accessor<1u>> out_sub_b) {
                auto slm_sub = X_slm.subflat(bb);
                auto out_sub_a = X_out->subflat(bb, mm, 2 * kk);
                bb.add(for_loop_builder(assignment(n2, n_local), n2 <= N1 * N2 / 2,
                                        add_into(n2, cfg.Nb))
                           .body([&](block_builder &bb) {
                               r2c_post_i(bb, cfg.fp, n2, N1 * N2, slm_sub, *out_sub_a, *out_sub_b);
                           })
                           .get_product());
            };
            bb.add(if_selection_builder(mm < cfg.M && 2 * kk + 1 < K)
                       .then([&](block_builder &bb) {
                           store_fac(bb, X_out->subflat(bb, mm, 2 * kk + 1));
                       })
                       .otherwise([&](block_builder &bb) {
                           bb.add(if_selection_builder(mm < cfg.M && 2 * kk < K)
                                      .then([&](block_builder &bb) {
                                          store_fac(bb,
                                                    std::make_unique<zero_accessor<1u>>(cfg.fp));
                                      })
                                      .get_product());
                       })
                       .get_product());
        } else {
            auto fused_compute_store_fac2 = [&](block_builder &bb) {
                auto x = bb.declare(data_type(array_of(fph.type(2), N1)), "x");
                load_store_slm(bb, X_slm, x, n2, N1, N2, false, false);
                auto factor = trial_division(N1);
                fft_inplace(bb, cfg.fp, cfg.direction, factor, x, nullptr);
                load_store_global(bb, *X_out, array_accessor<1u>(x, {}), mm, cfg.M, kk, K, n2, N1,
                                  N2, true, unscrambler(factor));
            };
            parallel_n2(bb, n2, N2, fused_compute_store_fac2);
        }
    });

    auto f = fb.get_product();
    make_names_unique(f);
    unsafe_simplify(f);

    generate_opencl(os, f);
}

} // namespace bbfft
