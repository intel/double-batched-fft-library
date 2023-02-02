// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_MAKE_PLAN_20221205_HPP
#define SYCL_MAKE_PLAN_20221205_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/export.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/plan.hpp"

#include <CL/sycl.hpp>

namespace bbfft {

/**
 * @brief Create a plan for the configuration
 *
 * @param cfg configuration
 * @param queue queue handle
 * @param cache optional kernel cache
 *
 * @return plan
 */
BBFFT_EXPORT auto make_plan(configuration const &cfg, ::sycl::queue queue,
                            jit_cache *cache = nullptr) -> plan<::sycl::event>;
/**
 * @brief Create a plan for the configuration
 *
 * @param cfg configuration
 * @param queue queue handle
 * @param context context handle
 * @param device device handle
 * @param cache optional kernel cache
 *
 * @return plan
 */
BBFFT_EXPORT auto make_plan(configuration const &cfg, ::sycl::queue queue, ::sycl::context context,
                            ::sycl::device device, jit_cache *cache = nullptr)
    -> plan<::sycl::event>;

} // namespace bbfft

#endif // SYCL_MAKE_PLAN_20221205_HPP
