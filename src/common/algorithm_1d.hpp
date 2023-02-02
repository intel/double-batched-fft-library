// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ALGORITHM_1D_20220602_HPP
#define ALGORITHM_1D_20220602_HPP

#include "algorithm/factor2_slm_fft.hpp"
#include "algorithm/small_batch_fft.hpp"
#include "bbfft/cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/detail/plan_impl.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace bbfft {

template <typename Api>
std::shared_ptr<detail::plan_impl<typename Api::event_type>>
select_1d_fft_algorithm(configuration const &cfg, Api api, cache *ch) {
    auto info = api.info();
    int sgs = info.min_subgroup_size();
    auto reg_space = info.register_space();
    std::size_t N = cfg.shape[1];
    auto required_reg_space_for_small_batch = 2 * static_cast<int>(cfg.fp) * N * sgs;

    if (required_reg_space_for_small_batch >= reg_space / 2) {
        return std::make_shared<factor2_slm_fft<Api>>(cfg, std::move(api), ch);
    }
    return std::make_shared<small_batch_fft<Api>>(cfg, std::move(api), ch);
}

} // namespace bbfft

#endif // ALGORITHM_1D_20220602_HPP
