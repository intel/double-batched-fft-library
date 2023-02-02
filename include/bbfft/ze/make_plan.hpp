// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_MAKE_PLAN_20221205_HPP
#define ZE_MAKE_PLAN_20221205_HPP

#include "bbfft/cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/export.hpp"
#include "bbfft/plan.hpp"

#include <level_zero/ze_api.h>

namespace bbfft {

/**
 * @brief Create a plan for the configuration
 *
 * @param cfg configuration
 * @param queue queue handle
 * @param context context handle
 * @param device device handle
 * @param ch optional kernel cache
 *
 * @return plan
 */
BBFFT_EXPORT auto make_plan(configuration const &cfg, ze_command_list_handle_t queue,
                            ze_context_handle_t context, ze_device_handle_t device,
                            cache *ch = nullptr) -> plan<ze_event_handle_t>;

} // namespace bbfft

#endif // ZE_MAKE_PLAN_20221205_HPP
