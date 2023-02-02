// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SMALL_BATCH_FFT_20220413_HPP
#define SMALL_BATCH_FFT_20220413_HPP

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
#include <sstream>
#include <string_view>
#include <vector>

namespace bbfft {

template <typename Api> class small_batch_fft : public detail::plan_impl<typename Api::event_type> {
  public:
    using event = typename Api::event_type;
    using kernel_bundle = typename Api::kernel_bundle_type;
    using kernel = typename Api::kernel_type;

    small_batch_fft(configuration const &cfg, Api api, jit_cache *cache)
        : api_(std::move(api)), p_(setup(cfg, cache)), k_(api_.create_kernel(p_, identifier_)) {}
    ~small_batch_fft() {
        api_.release_kernel(k_);
        api_.release_kernel_bundle(p_);
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
            h.set_arg(2, K_);
        });
    }

  private:
    kernel_bundle setup(configuration const &cfg, jit_cache *cache) {
        std::stringstream ss;
        if (cfg.callbacks) {
            ss << std::string_view(cfg.callbacks.data, cfg.callbacks.length) << std::endl;
        }
        auto sbc = configure_small_batch_fft(cfg, api_.info());

        K_ = cfg.shape[2];

        std::size_t Mg = (sbc.M - 1) / sbc.Mb + 1;
        bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
        uint64_t Kng = is_real ? (K_ - 1) / 2 + 1 : K_;
        std::size_t Kg = (Kng - 1) / sbc.Kb + 1;
        gws_ = std::array<std::size_t, 3>{Mg * sbc.Mb, Kg * sbc.Kb, 1};
        lws_ = std::array<std::size_t, 3>{sbc.Mb, sbc.Kb, 1};
        inplace_unsupported_ = sbc.inplace_unsupported;
        identifier_ = sbc.identifier();

        auto const make_cache_key = [this]() {
            return jit_cache_key{identifier_, api_.device_id()};
        };

        bool use_cache = cache && !cfg.callbacks;
        if (use_cache) {
            auto [ptr, size] = cache->get_binary(make_cache_key());
            if (ptr && size > 0) {
                return api_.build_kernel_bundle(ptr, size);
            }
        }

        generate_small_batch_fft(ss, sbc);

        auto bundle = api_.build_kernel_bundle(ss.str());
        if (use_cache) {
            cache->store_binary(make_cache_key(), api_.get_native_binary(bundle));
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
}; // namespace bbfft

} // namespace bbfft

#endif // SMALL_BATCH_FFT_20220413_HPP
