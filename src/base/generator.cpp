// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
#include "algorithm.hpp"
#include "dummy_api.hpp"

#include "bbfft/cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"

#include <ostream>

namespace bbfft {

void generate_fft_kernels(std::ostream &os, std::vector<configuration> const &cfgs,
                          device_info info) {
    auto api = dummy_api(std::move(info), &os);
    cache_all ch;
    for (auto const &cfg : cfgs) {
        select_fft_algorithm(cfg, api, &ch);
    }
}

} // namespace bbfft
