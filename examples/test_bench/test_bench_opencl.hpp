// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_OPENCL_20221207_HPP
#define TEST_BENCH_OPENCL_20221207_HPP

#include "bbfft/cl/make_plan.hpp"
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

    void *malloc_device(size_t bytes);
    template <typename T> T *malloc_device(size_t elements) {
        return (T *)malloc_device(elements * sizeof(T));
    }

    void memcpy(void *dest, const void *src, size_t bytes);
    template <typename T> void copy(T const *src, T *dest, size_t count) {
        memcpy(dest, src, count * sizeof(T));
    }

    void free(void *ptr);

    inline auto device() const { return device_; }
    inline auto context() const { return context_; }
    inline auto queue() const { return queue_; }

    void setup_plan(bbfft::configuration const &cfg);
    void run_plan(void const *in, void *out, std::uint32_t ntimes);

  private:
    bbfft::opencl_plan plan_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    using clDeviceMemAllocINTEL_t = void *(*)(cl_context, cl_device_id, cl_bitfield const *, size_t,
                                              cl_uint, cl_int *);
    using clMemFreeINTEL_t = cl_int (*)(cl_context, void *);
    using clEnqueueMemcpyINTEL_t = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
                                              size_t, cl_uint, const cl_event *, cl_event *);
    clDeviceMemAllocINTEL_t clDeviceMemAllocINTEL;
    clMemFreeINTEL_t clMemFreeINTEL;
    clEnqueueMemcpyINTEL_t clEnqueueMemcpyINTEL;
};

#endif // TEST_BENCH_OPENCL_20221207_HPP
