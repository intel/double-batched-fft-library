// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_opencl.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/cl/make_plan.hpp"
#include "bbfft/configuration.hpp"

test_bench_opencl::test_bench_opencl() {
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
    clDeviceMemAllocINTEL = (clDeviceMemAllocINTEL_t)clGetExtensionFunctionAddressForPlatform(
        platform, "clDeviceMemAllocINTEL");
    clMemFreeINTEL =
        (clMemFreeINTEL_t)clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL");
    clEnqueueMemcpyINTEL = (clEnqueueMemcpyINTEL_t)clGetExtensionFunctionAddressForPlatform(
        platform, "clEnqueueMemcpyINTEL");
    if (!clDeviceMemAllocINTEL || !clMemFreeINTEL || !clEnqueueMemcpyINTEL) {
        throw bbfft::cl::error("OpenCL unified shared memory extension unavailable",
                               CL_INVALID_COMMAND_QUEUE);
    }
}

test_bench_opencl::~test_bench_opencl() {
    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
}

void *test_bench_opencl::malloc_device(size_t bytes) {
    cl_int err;
    void *out = clDeviceMemAllocINTEL(context_, device_, nullptr, bytes, 0, &err);
    CL_CHECK(err);
    return out;
}

cl_event test_bench_opencl::memcpy(void *dest, const void *src, size_t bytes) {
    cl_event event;
    CL_CHECK(clEnqueueMemcpyINTEL(queue_, CL_FALSE, dest, src, bytes, 0, nullptr, &event));
    return event;
}

void test_bench_opencl::free(void *ptr) { CL_CHECK(clMemFreeINTEL(context_, ptr)); }

void test_bench_opencl::wait(cl_event e) { CL_CHECK(clWaitForEvents(1, &e)); }
void test_bench_opencl::release(cl_event e) { CL_CHECK(clReleaseEvent(e)); }
void test_bench_opencl::wait_and_release(cl_event e) {
    wait(e);
    release(e);
}

auto test_bench_opencl::make_plan(bbfft::configuration const &cfg) const -> bbfft::plan<cl_event> {
    return bbfft::make_plan(cfg, queue_, context_, device_);
}
