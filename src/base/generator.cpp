// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
#include "algorithm.hpp"
#include "dummy_api.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/jit_cache.hpp"

#include <ostream>

namespace bbfft {

std::vector<std::string>
generate_fft_kernels(std::ostream &os, std::vector<configuration> const &cfgs, device_info info) {
    auto api = dummy_api(std::move(info), &os);
    jit_cache_all cache;
    for (auto const &cfg : cfgs) {
        select_fft_algorithm(cfg, api, &cache);
    }
    return cache.kernel_names();
}

} // namespace bbfft
