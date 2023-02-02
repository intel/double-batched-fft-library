// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/plan.hpp"
#include "algorithm.hpp"
#include "api.hpp"
#include "bbfft/cache.hpp"
#include "bbfft/cl/make_plan.hpp"
#include "bbfft/configuration.hpp"

#include <CL/cl.h>

namespace bbfft {

auto make_plan(configuration const &cfg, cl_command_queue queue, cache *ch) -> plan<cl_event> {
    return plan<cl_event>(select_fft_algorithm<cl::api>(cfg, cl::api(queue), ch));
}

auto make_plan(configuration const &cfg, cl_command_queue queue, cl_context context,
               cl_device_id device, cache *ch) -> plan<cl_event> {
    return plan<cl_event>(select_fft_algorithm<cl::api>(cfg, cl::api(queue, context, device), ch));
}

} // namespace bbfft

