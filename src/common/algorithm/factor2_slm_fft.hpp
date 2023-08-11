// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FACTOR2_SLM_FFT_20220413_HPP
#define FACTOR2_SLM_FFT_20220413_HPP

#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/generator_impl.hpp"
#include "bbfft/detail/plan_impl.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

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

template <typename Api> class factor2_slm_fft_base : public Api::plan_type {
  public:
    using buffer = typename Api::buffer_type;
    using kernel_bundle = typename Api::kernel_bundle_type;
    using kernel = typename Api::kernel_type;

    factor2_slm_fft_base(configuration const &cfg, Api api, jit_cache *cache)
        : api_(std::move(api)), module_(setup(cfg, cache)),
          bundle_(api_.make_kernel_bundle(module_.get())),
          k_(api_.create_kernel(bundle_, identifier_)) {}

    ~factor2_slm_fft_base() {
        api_.release_kernel(k_);
        api_.release_buffer(twiddle_);
    }

    factor2_slm_fft_base(factor2_slm_fft_base const &) = delete;
    factor2_slm_fft_base(factor2_slm_fft_base &&) = delete;
    factor2_slm_fft_base &operator=(factor2_slm_fft_base const &) = delete;
    factor2_slm_fft_base &operator=(factor2_slm_fft_base &&) = delete;

  protected:
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

    auto setup(configuration const &cfg, jit_cache *cache) -> shared_handle<module_handle_t> {
        std::stringstream ss;
        if (cfg.callbacks) {
            ss << std::string_view(cfg.callbacks.data, cfg.callbacks.length) << std::endl;
        }
        bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
        auto f2c = configure_factor2_slm_fft(cfg, api_.info());

        K_ = cfg.shape[2];

        switch (cfg.fp) {
        case precision::f32:
            create_twiddle<float>(static_cast<int>(cfg.dir), f2c.N1, f2c.N2);
            break;
        case precision::f64:
            create_twiddle<double>(static_cast<int>(cfg.dir), f2c.N1, f2c.N2);
            break;
        }

        std::size_t Mg = (f2c.M - 1) / f2c.Mb + 1;
        uint64_t Kng = is_real ? (K_ - 1) / 2 + 1 : K_;
        std::size_t Kg = (Kng - 1) / f2c.Kb + 1;
        gws_ = std::array<std::size_t, 3>{Mg * f2c.Mb, f2c.Nb, Kg * f2c.Kb};
        lws_ = std::array<std::size_t, 3>{f2c.Mb, f2c.Nb, f2c.Kb};
        inplace_unsupported_ = f2c.inplace_unsupported;
        identifier_ = f2c.identifier();

        auto const make_cache_key = [this]() {
            return jit_cache_key{identifier_, api_.device_id()};
        };

        if (cache) {
            auto bundle = cache->get(make_cache_key());
            if (bundle) {
                return bundle;
            }
        }

        generate_factor2_slm_fft(ss, f2c);

        auto mod = api_.build_module(ss.str());
        if (cache) {
            cache->store(make_cache_key(), mod);
        }

        return mod;
    }

    Api api_;
    std::array<std::size_t, 3> gws_;
    std::array<std::size_t, 3> lws_;
    bool inplace_unsupported_;
    std::string identifier_;
    shared_handle<module_handle_t> module_;
    kernel_bundle bundle_;
    kernel k_;
    uint64_t K_;
    buffer twiddle_;
};

template <typename Api, typename PlanImplT = typename Api::plan_type> class factor2_slm_fft;

template <typename Api>
class factor2_slm_fft<Api, detail::plan_impl<typename Api::event_type>>
    : public factor2_slm_fft_base<Api> {
  public:
    using factor2_slm_fft_base<Api>::factor2_slm_fft_base;
    using event = typename Api::event_type;

    auto execute(void const *in, void *out, std::vector<event> const &dep_events)
        -> event override {
        if (in == out && this->inplace_unsupported_) {
            throw bad_configuration("The plan does not support in-place transform on the current "
                                    "device. Please use the out-of-place transform.");
        }
        return this->api_.launch_kernel(this->k_, this->gws_, this->lws_, dep_events, [&](auto &h) {
            h.set_arg(0, in);
            h.set_arg(1, out);
            h.set_arg(2, this->twiddle_);
            h.set_arg(3, this->K_);
        });
    }
};

template <typename Api>
class factor2_slm_fft<Api, detail::plan_unmanaged_event_impl<typename Api::event_type>>
    : public factor2_slm_fft_base<Api> {
  public:
    using factor2_slm_fft_base<Api>::factor2_slm_fft_base;
    using event = typename Api::event_type;

    void execute(void const *in, void *out, event signal_event, std::uint32_t num_dep_events,
                 event *dep_events) override {
        if (in == out && this->inplace_unsupported_) {
            throw bad_configuration("The plan does not support in-place transform on the current "
                                    "device. Please use the out-of-place transform.");
        }
        this->api_.launch_kernel(this->k_, this->gws_, this->lws_, signal_event, num_dep_events,
                                 dep_events, [&](auto &h) {
                                     h.set_arg(0, in);
                                     h.set_arg(1, out);
                                     h.set_arg(2, this->twiddle_);
                                     h.set_arg(3, this->K_);
                                 });
    }
};

} // namespace bbfft

#endif // FACTOR2_SLM_FFT_20220413_HPP
