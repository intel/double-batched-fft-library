// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/plan.hpp"
#include "algorithm.hpp"
#include "api.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/sycl/make_plan.hpp"

#include <CL/sycl.hpp>
#include <utility>

namespace bbfft {

auto make_plan(configuration const &cfg, ::sycl::queue q, jit_cache *cache) -> sycl_plan {
    return make_plan(cfg, q, q.get_context(), q.get_device(), cache);
}

auto make_plan(configuration const &cfg, ::sycl::queue q, ::sycl::context c, ::sycl::device d,
               jit_cache *cache) -> sycl_plan {
    return sycl_plan(select_fft_algorithm<sycl::api>(
        cfg, sycl::api(std::move(q), std::move(c), std::move(d)), cache));
}

} // namespace bbfft

