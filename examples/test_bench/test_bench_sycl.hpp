// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_SYCL_20221130_HPP
#define TEST_BENCH_SYCL_20221130_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/plan.hpp"
#include "bbfft/sycl/make_plan.hpp"

#include <CL/sycl.hpp>

class test_bench_sycl {
  public:
    inline test_bench_sycl() : plan_{}, queue_{::sycl::default_selector_v} {}

    test_bench_sycl(test_bench_sycl const &) = delete;
    test_bench_sycl &operator=(test_bench_sycl const &) = delete;

    inline void *malloc_device(size_t bytes) { return ::sycl::malloc_device(bytes, queue_); }

    template <typename T> T *malloc_device(size_t elements) {
        return ::sycl::malloc_device<T>(elements, queue_);
    }

    inline void memcpy_d2h(void *dest, const void *src, size_t bytes) {
        queue_.memcpy(dest, src, bytes).wait();
    }
    inline void memcpy_h2d(void *dest, const void *src, size_t bytes) {
        queue_.memcpy(dest, src, bytes).wait();
    }

    inline void free(void *ptr) { ::sycl::free(ptr, queue_); }

    inline auto device() const { return queue_.get_device(); }
    inline auto context() const { return queue_.get_context(); }
    inline auto queue() const { return queue_; }

    void setup_plan(bbfft::configuration const &cfg);
    void run_plan(void const *in, void *out, std::uint32_t ntimes = 1);

  private:
    bbfft::sycl_plan plan_;
    ::sycl::queue queue_;
};

#endif // TEST_BENCH_SYCL_20221130_HPP
