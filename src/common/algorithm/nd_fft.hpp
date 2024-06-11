// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ND_FFT_20220602_HPP
#define ND_FFT_20220602_HPP

#include "algorithm/factor2_slm_fft.hpp"
#include "algorithm_1d.hpp"
#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/plan_impl.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/jit_cache.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace bbfft {

template <typename Api> class nd_fft_base : public Api::plan_type {
  public:
    using buffer = typename Api::buffer_type;
    using event = typename Api::event_type;

    nd_fft_base(configuration const &cfg, Api api, jit_cache *cache)
        : api_(std::move(api)), dim_(cfg.dim) {
        if (cfg.callbacks) {
            throw bad_configuration("User modules are unsuported for FFT dimension > 1.");
        }

        auto compare_strides = [](std::array<std::size_t, max_tensor_dim> const &s1,
                                  std::array<std::size_t, max_tensor_dim> const &s2, unsigned dim) {
            bool equal = true;
            for (unsigned d = 0; d < dim + 2; ++d) {
                equal = equal && (s1[d] == s2[d]);
            }
            return equal;
        };
        auto is_default_stride = [&compare_strides](configuration const &cfg, bool inplace) {
            auto const def_istride = default_istride(cfg.dim, cfg.shape, cfg.type, inplace);
            auto const def_ostride = default_ostride(cfg.dim, cfg.shape, cfg.type, inplace);
            return compare_strides(cfg.istride, def_istride, cfg.dim) &&
                   compare_strides(cfg.ostride, def_ostride, cfg.dim);
        };
        if (!is_default_stride(cfg, false) && !is_default_stride(cfg, true)) {
            throw bad_configuration("Only default tensor layouts are supported for the nd_fft.");
        }
        bool inplace_layout = is_default_stride(cfg, true);

        bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
        auto Nd_complex = [&](unsigned d) {
            return d == 0 && is_real ? cfg.shape[1] / 2 + 1 : cfg.shape[d + 1];
        };
        auto Nd_real = [&](unsigned d) {
            return d == 0 && is_real && inplace_layout ? 2 * (cfg.shape[1] / 2 + 1)
                                                       : cfg.shape[d + 1];
        };

        std::size_t N = 1;
        for (unsigned d = 0; d < dim_; ++d) {
            N *= Nd_complex(d);
        }

        std::array<configuration, max_fft_dim> cfg1d = {};
        std::size_t M = cfg.shape[0];
        std::size_t K = N * cfg.shape[dim_ + 1];
        std::array<std::size_t, max_tensor_dim> shape = {};
        std::array<std::size_t, max_tensor_dim> istride = {};
        std::array<std::size_t, max_tensor_dim> ostride = {};
        for (unsigned d = 0; d < dim_; ++d) {
            std::size_t Nd = cfg.shape[d + 1];
            std::size_t Ndc = Nd_complex(d);
            K /= Ndc;
            shape[0] = M;
            shape[1] = Nd;
            shape[2] = K;
            auto type = d == 0 ? cfg.type : transform_type::c2c;
            istride[0] = 1;
            istride[1] = M;
            istride[2] = M * Nd_real(d);
            ostride[0] = 1;
            ostride[1] = M;
            ostride[2] = M * Ndc;
            cfg1d[d] = {1, shape, cfg.fp, cfg.dir, type, istride, ostride};
            M *= Ndc;
        }
        if (cfg.type == transform_type::c2r) {
            for (unsigned d = 0; d < dim_; ++d) {
                auto &c = cfg1d[dim_ - 1 - d];
                std::swap(c.istride, c.ostride);
                plans_[d] = select_1d_fft_algorithm<Api>(c, api_, cache);
            }
        } else {
            for (unsigned d = 0; d < dim_; ++d) {
                plans_[d] = select_1d_fft_algorithm<Api>(cfg1d[d], api_, cache);
            }
        }

        std::size_t bytes_per_real = static_cast<std::size_t>(cfg.fp);
        std::size_t bytes_per_complex = 2 * bytes_per_real;
        auto ibytes = cfg.type == transform_type::r2c ? bytes_per_real : bytes_per_complex;
        auto obytes = cfg.type == transform_type::c2r ? bytes_per_real : bytes_per_complex;
        auto isize = cfg.istride[dim_ + 1] * cfg.shape[dim_ + 1] * ibytes;
        auto osize = cfg.ostride[dim_ + 1] * cfg.shape[dim_ + 1] * obytes;
        // if the input buffer is larger than the output buffer than temporaries are larger than the
        // output buffer and we cannot reuse the output buffer for temporaries
        if (isize > osize) {
            tmp_ = api_.create_device_buffer(isize);
        }
    }

    ~nd_fft_base() {
        if (tmp_) {
            api_.release_buffer(tmp_);
        }
    }

    nd_fft_base(nd_fft_base const &) = delete;
    nd_fft_base(nd_fft_base &&) = delete;
    nd_fft_base &operator=(nd_fft_base const &) = delete;
    nd_fft_base &operator=(nd_fft_base &&) = delete;

  protected:
    Api api_;
    unsigned dim_;
    std::array<std::shared_ptr<typename Api::plan_type>, max_fft_dim> plans_;
    buffer tmp_ = nullptr;
};

template <typename Api, typename PlanImplT = typename Api::plan_type> class nd_fft;

template <typename Api>
class nd_fft<Api, detail::plan_impl<typename Api::event_type>> : public nd_fft_base<Api> {
  public:
    using nd_fft_base<Api>::nd_fft_base;
    using event = typename Api::event_type;

    auto execute(mem const &in, mem const &out, std::vector<event> const &dep_events)
        -> event override {
        auto tmp = this->tmp_ ? mem(this->tmp_) : out;
        event e = this->plans_[0]->execute(in, tmp, dep_events);
        for (unsigned d = 1; d < this->dim_ - 1; ++d) {
            auto next_e = this->plans_[d]->execute(tmp, tmp, std::vector<event>{e});
            this->api_.release_event(e);
            e = std::move(next_e);
        }
        auto last_e = this->plans_[this->dim_ - 1]->execute(tmp, out, std::vector<event>{e});
        this->api_.release_event(std::move(e));
        return last_e;
    }
};

template <typename Api>
class nd_fft<Api, detail::plan_unmanaged_event_impl<typename Api::event_type>>
    : public nd_fft_base<Api> {
  public:
    using nd_fft_base<Api>::nd_fft_base;
    using event = typename Api::event_type;

    void execute(mem const &in, mem const &out, event signal_event, std::uint32_t num_dep_events,
                 event *dep_events) override {
        auto tmp = this->tmp_ ? mem(this->tmp_) : out;
        auto e = this->api_.get_internal_event();
        this->plans_[0]->execute(in, tmp, e, num_dep_events, dep_events);
        for (unsigned d = 1; d < this->dim_ - 1; ++d) {
            auto next_e = this->api_.get_internal_event();
            this->plans_[d]->execute(tmp, tmp, next_e, 1, &e);
            this->api_.append_reset_event(e);
            e = std::move(next_e);
        }
        this->plans_[this->dim_ - 1]->execute(tmp, out, signal_event, 1, &e);
        this->api_.append_reset_event(e);
    }
};

} // namespace bbfft

#endif // ND_FFT_20220602_HPP
