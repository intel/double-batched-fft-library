// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_OPENCL_20221207_HPP
#define TEST_BENCH_OPENCL_20221207_HPP

#include "bbfft/cl/make_plan.hpp"
#include "bbfft/cl/mem.hpp"
#include "bbfft/mem.hpp"
#include "bbfft/plan.hpp"

#include <CL/cl.h>

namespace bbfft {
struct configuration;
}

class test_bench_opencl {
  public:
    test_bench_opencl();
    ~test_bench_opencl();

    test_bench_opencl(test_bench_opencl const &) = delete;
    test_bench_opencl &operator=(test_bench_opencl const &) = delete;

    cl_mem malloc_device(size_t bytes);
    template <typename T> cl_mem malloc_device(size_t elements) {
        return malloc_device(elements * sizeof(T));
    }

    void memcpy_d2h(void *dest, const cl_mem src, size_t bytes);
    void memcpy_h2d(cl_mem dest, const void *src, size_t bytes);

    void free(cl_mem buf);

    inline auto device() const { return device_; }
    inline auto context() const { return context_; }
    inline auto queue() const { return queue_; }

    void setup_plan(bbfft::configuration const &cfg);
    void run_plan(bbfft::mem const &in, bbfft::mem const &out, std::uint32_t ntimes);

  private:
    bbfft::opencl_plan plan_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
};

#endif // TEST_BENCH_OPENCL_20221207_HPP
