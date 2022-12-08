// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_sycl.hpp"
#include "bbfft/sycl/make_plan.hpp"

auto test_bench_sycl::make_plan(bbfft::configuration const &cfg) const
    -> bbfft::plan<::sycl::event> {
    return bbfft::make_plan(cfg, queue_);
}
