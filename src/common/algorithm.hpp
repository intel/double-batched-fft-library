// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ALGORITHM_20220602_HPP
#define ALGORITHM_20220602_HPP

#include "algorithm/nd_fft.hpp"
#include "algorithm_1d.hpp"
#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/plan_impl.hpp"

#include <memory>
#include <string>
#include <utility>

namespace bbfft {

template <typename Api>
std::shared_ptr<detail::plan_impl<typename Api::event_type>>
select_fft_algorithm(configuration const &cfg, Api api) {
    if (cfg.dim < 1 || cfg.dim > max_fft_dim) {
        throw bad_configuration("Unsupported FFT dimension: " + std::to_string(cfg.dim));
    }
    if (cfg.dim == 1) {
        return select_1d_fft_algorithm<Api>(cfg, std::move(api));
    }
    return std::make_shared<nd_fft<Api>>(cfg, std::move(api));
}

} // namespace bbfft

#endif // ALGORITHM_20220602_HPP
