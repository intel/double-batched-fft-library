// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FACTOR2_SLM_FFT_20220413_HPP
#define FACTOR2_SLM_FFT_20220413_HPP

#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/plan_impl.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/generator.hpp"
#include "bbfft/jit_cache.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string_view>
#include <vector>

namespace bbfft {

template <typename Api> class factor2_slm_fft : public detail::plan_impl<typename Api::event_type> {
  public:
    using event = typename Api::event_type;
    using buffer = typename Api::buffer_type;
    using kernel_bundle = typename Api::kernel_bundle_type;
    using kernel = typename Api::kernel_type;

    factor2_slm_fft(configuration const &cfg, Api api, jit_cache *cache)
        : api_(std::move(api)), p_(setup(cfg, cache)), k_(p_.create_kernel(identifier_)) {}

    ~factor2_slm_fft() {
        if (X1_) {
            api_.release_buffer(X1_);
        }
        api_.release_buffer(twiddle_);
    }

    auto execute(void const *in, void *out, std::vector<event> const &dep_events)
        -> event override {
        if (in == out && inplace_unsupported_) {
            throw bad_configuration("The plan does not support in-place transform on the current "
                                    "device. Please use the out-of-place transform.");
        }
        return api_.launch_kernel(k_, gws_, lws_, dep_events, [&](auto &h) {
            h.set_arg(0, in);
            h.set_arg(1, out);
            h.set_arg(2, twiddle_);
            h.set_arg(3, K_);
            if (X1_) {
                h.set_arg(4, X1_);
            }
        });
    }

  private:
    template <typename T> void create_twiddle(int direction, int N1, int N2) {
        constexpr double tau = 6.28318530717958647693;
        auto twiddle = std::vector<T>(2 * (N1 * N2));
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                auto arg = direction * tau / (N1 * N2) * i * j;
                twiddle[2 * (j + N2 * i)] = std::cos(arg);
                twiddle[2 * (j + N2 * i) + 1] = std::sin(arg);
            }
        }
        twiddle_ = api_.create_twiddle_table(twiddle);
    }

    kernel_bundle setup(configuration const &cfg, jit_cache *cache) {
        std::stringstream ss;
        if (cfg.callbacks) {
            ss << std::string_view(cfg.callbacks.data, cfg.callbacks.length) << std::endl;
        }
        bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
        auto f2c = configure_factor2_slm_fft(cfg, api_.info());

        uint64_t M = cfg.shape[0];
        K_ = cfg.shape[2];

        switch (cfg.fp) {
        case precision::f32:
            create_twiddle<float>(static_cast<int>(cfg.dir), f2c.N1, f2c.N2);
            break;
        case precision::f64:
            create_twiddle<double>(static_cast<int>(cfg.dir), f2c.N1, f2c.N2);
            break;
        }

        if (f2c.external_buffer) {
            std::size_t bytes_per_complex = 2 * static_cast<std::size_t>(cfg.fp);
            X1_ = api_.create_device_buffer(M * f2c.N1 * f2c.N2 * K_ * bytes_per_complex);
        }

        std::size_t Mg = (f2c.M - 1) / f2c.Mb + 1;
        uint64_t Kng = is_real ? (K_ - 1) / 2 + 1 : K_;
        std::size_t Kg = (Kng - 1) / f2c.Kb + 1;
        gws_ = std::array<std::size_t, 3>{Mg * f2c.Mb, f2c.Nb, Kg * f2c.Kb};
        lws_ = std::array<std::size_t, 3>{f2c.Mb, f2c.Nb, f2c.Kb};
        inplace_unsupported_ = f2c.inplace_unsupported;
        identifier_ = f2c.identifier();

        auto const make_cache_key = [this](factor2_slm_configuration const &f2c) {
            jit_cache_key key = {};
            static_assert(sizeof(key.cfg) >= sizeof(f2c));
            std::memcpy(&key.cfg[0], &f2c, sizeof(f2c));
            key.device_id = api_.device_id();
            return key;
        };

        bool use_cache = cache && !cfg.callbacks;
        if (use_cache) {
            auto [ptr, size] = cache->get_binary(make_cache_key(f2c));
            if (ptr && size > 0) {
                return api_.build_kernel_bundle(ptr, size);
            }
        }

        generate_factor2_slm_fft(ss, f2c);

        auto bundle = api_.build_kernel_bundle(ss.str());
        if (use_cache) {
            cache->store_binary(make_cache_key(f2c), bundle.get_binary());
        }

        return bundle;
    }

    Api api_;
    std::array<std::size_t, 3> gws_;
    std::array<std::size_t, 3> lws_;
    bool inplace_unsupported_;
    std::string identifier_;
    kernel_bundle p_;
    kernel k_;
    uint64_t K_;
    buffer twiddle_;
    buffer X1_ = nullptr;
}; // namespace bbfft

} // namespace bbfft

#endif // FACTOR2_SLM_FFT_20220413_HPP
