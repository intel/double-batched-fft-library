// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_SYCL_20221130_HPP
#define TEST_BENCH_SYCL_20221130_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/plan.hpp"

#include <CL/sycl.hpp>

class test_bench_sycl {
  public:
    inline test_bench_sycl() : queue_() {}

    test_bench_sycl(test_bench_sycl const &) = delete;
    test_bench_sycl &operator=(test_bench_sycl const &) = delete;

    inline void *malloc_device(size_t bytes) { return ::sycl::malloc_device(bytes, queue_); }

    template <typename T> T *malloc_device(size_t elements) {
        return ::sycl::malloc_device<T>(elements, queue_);
    }

    inline ::sycl::event memcpy(void *dest, const void *src, size_t bytes) {
        return queue_.memcpy(dest, src, bytes);
    }

    template <typename T>::sycl::event copy(T const *src, T *dest, size_t count) {
        return queue_.copy(src, dest, count);
    }

    inline void free(void *ptr) { ::sycl::free(ptr, queue_); }

    inline static void wait(::sycl::event e) { e.wait(); }
    inline static void release(::sycl::event) {}
    inline static void wait_and_release(::sycl::event e) { wait(std::move(e)); }

    inline auto device() const { return queue_.get_device(); }
    inline auto context() const { return queue_.get_context(); }
    inline auto queue() const { return queue_; }

    auto make_plan(bbfft::configuration const &cfg) const -> bbfft::plan<::sycl::event>;

  private:
    ::sycl::queue queue_;
};

#endif // TEST_BENCH_SYCL_20221130_HPP
