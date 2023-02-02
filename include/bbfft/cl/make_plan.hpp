// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_MAKE_PLAN_20221205_HPP
#define CL_MAKE_PLAN_20221205_HPP

#include "bbfft/cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/export.hpp"
#include "bbfft/plan.hpp"

#include <CL/cl.h>

namespace bbfft {

/**
 * @brief Create a plan for the configuration
 *
 * @param cfg configuration
 * @param queue queue handle
 * @param ch optional kernel cache
 *
 * @return plan
 */
BBFFT_EXPORT auto make_plan(configuration const &cfg, cl_command_queue queue, cache *ch = nullptr)
    -> plan<cl_event>;
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
BBFFT_EXPORT auto make_plan(configuration const &cfg, cl_command_queue queue, cl_context context,
                            cl_device_id device, cache *ch = nullptr) -> plan<cl_event>;

} // namespace bbfft

#endif // CL_MAKE_PLAN_20221205_HPP
