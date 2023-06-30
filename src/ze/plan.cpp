// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/plan.hpp"
#include "algorithm.hpp"
#include "api.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/ze/make_plan.hpp"

#include <level_zero/ze_api.h>

namespace bbfft {

auto make_plan(configuration const &cfg, ze_command_list_handle_t queue,
               ze_context_handle_t context, ze_device_handle_t device, jit_cache *cache)
    -> level_zero_plan {
    return level_zero_plan(
        select_fft_algorithm<ze::api>(cfg, ze::api(queue, context, device), cache));
}

} // namespace bbfft

