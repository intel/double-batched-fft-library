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

#include <cstddef>
#include <memory>
#include <vector>

namespace bbfft {

template <typename Api> class nd_fft : public detail::plan_impl<typename Api::event_type> {
  public:
    using event = typename Api::event_type;
    using buffer = typename Api::buffer_type;

    nd_fft(configuration const &cfg, Api api) : api_(std::move(api)), dim_(cfg.dim) {
        if (cfg.callbacks) {
            throw bad_configuration("User modules are unsuported for FFT dimension > 1.");
        }

        bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
        auto Nd_complex = [&](unsigned d) {
            return d == 0 && is_real ? cfg.shape[1] / 2 + 1 : cfg.shape[d + 1];
        };
        auto Nd_real = [&](unsigned d) {
            return d == 0 && is_real ? 2 * (cfg.shape[1] / 2 + 1) : cfg.shape[d + 1];
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
                plans_[d] = select_1d_fft_algorithm<Api>(c, api_);
            }
        } else {
            for (unsigned d = 0; d < dim_; ++d) {
                plans_[d] = select_1d_fft_algorithm<Api>(cfg1d[d], api_);
            }
        }
    }

    auto execute(void const *in, void *out, std::vector<event> const &dep_events)
        -> event override {
        if (in != out) {
            throw bad_configuration("Out-of-place is unsupported for FFT dimension > 1.");
        }
        event e = plans_[0]->execute(in, out, dep_events);
        for (unsigned d = 1; d < dim_; ++d) {
            auto next_e = plans_[d]->execute(out, out, std::vector<event>{e});
            api_.release_event(e);
            e = next_e;
        }
        return e;
    }

  private:
    Api api_;
    unsigned dim_;
    std::array<std::shared_ptr<detail::plan_impl<event>>, max_fft_dim> plans_;
};

} // namespace bbfft

#endif // ND_FFT_20220602_HPP
