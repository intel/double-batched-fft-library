// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/plan.hpp"
#include "algorithm.hpp"
#include "api.hpp"
#include "bbfft/cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"

#include <CL/sycl.hpp>
#include <utility>

namespace bbfft {

auto make_plan(configuration const &cfg, ::sycl::queue q, cache *ch) -> plan<::sycl::event> {
    return make_plan(cfg, q, q.get_context(), q.get_device(), ch);
}

auto make_plan(configuration const &cfg, ::sycl::queue q, ::sycl::context c, ::sycl::device d,
               cache *ch) -> plan<::sycl::event> {
    return plan<::sycl::event>(select_fft_algorithm<sycl::api>(
        cfg, sycl::api(std::move(q), std::move(c), std::move(d)), ch));
}

} // namespace bbfft

