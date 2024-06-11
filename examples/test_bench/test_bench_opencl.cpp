// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_opencl.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/configuration.hpp"

test_bench_opencl::test_bench_opencl() : plan_{} {
    cl_uint num_platforms = 1;
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(num_platforms, &platform, nullptr));
    cl_uint num_devices = 1;
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, &device_, nullptr));
    cl_int err;
    context_ = clCreateContext(nullptr, num_devices, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
    CL_CHECK(err);
}

test_bench_opencl::~test_bench_opencl() {
    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
}

cl_mem test_bench_opencl::malloc_device(size_t bytes) {
    cl_int err;
    cl_mem out = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CL_CHECK(err);
    return out;
}

void test_bench_opencl::memcpy_d2h(void *dest, const cl_mem src, size_t bytes) {
    CL_CHECK(clEnqueueReadBuffer(queue_, src, CL_TRUE, 0, bytes, dest, 0, nullptr, nullptr));
}
void test_bench_opencl::memcpy_h2d(cl_mem dest, const void *src, size_t bytes) {
    CL_CHECK(clEnqueueWriteBuffer(queue_, dest, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr));
}

void test_bench_opencl::free(cl_mem buf) { CL_CHECK(clReleaseMemObject(buf)); }

void test_bench_opencl::setup_plan(bbfft::configuration const &cfg) {
    plan_ = bbfft::make_plan(cfg, queue_, context_, device_);
}

void test_bench_opencl::run_plan(bbfft::mem const &in, bbfft::mem const &out,
                                 std::uint32_t ntimes) {
    auto event = plan_.execute(in, out);
    for (std::uint32_t n = 1; n < ntimes; ++n) {
        auto next_event = plan_.execute(in, out, event);
        CL_CHECK(clReleaseEvent(event));
        event = next_event;
    }
    CL_CHECK(clWaitForEvents(1, &event));
    CL_CHECK(clReleaseEvent(event));
}
