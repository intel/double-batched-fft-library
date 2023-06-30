// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_sycl.hpp"

void test_bench_sycl::setup_plan(bbfft::configuration const &cfg) {
    plan_ = bbfft::make_plan(cfg, queue_);
}

void test_bench_sycl::run_plan(void const *in, void *out, std::uint32_t ntimes) {
    auto e = plan_.execute(in, out);
    for (std::uint32_t n = 1; n < ntimes; ++n) {
        e = plan_.execute(in, out, e);
    }
    e.wait();
}
