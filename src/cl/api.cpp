// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"
#include "bbfft/cl/device.hpp"
#include "bbfft/cl/online_compiler.hpp"

#include <CL/cl_ext.h>

namespace bbfft::cl {

api::api(cl_command_queue queue) : queue_(queue) {
    CL_CHECK(clRetainCommandQueue(queue_));

    CL_CHECK(
        clGetCommandQueueInfo(queue_, CL_QUEUE_CONTEXT, sizeof(cl_context), &context_, nullptr));
    CL_CHECK(clRetainContext(context_));

    CL_CHECK(
        clGetCommandQueueInfo(queue_, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_, nullptr));

    setup_extensions();
}
api::api(cl_command_queue queue, cl_context context, cl_device_id device)
    : queue_(queue), context_(context), device_(device) {
    CL_CHECK(clRetainCommandQueue(queue_));
    CL_CHECK(clRetainContext(context_));

    setup_extensions();
}
api::~api() {
    clReleaseContext(context_);
    clReleaseCommandQueue(queue_);
}

api::api(api const &other) { *this = other; }
void api::operator=(api const &other) {
    queue_ = other.queue_;
    CL_CHECK(clRetainCommandQueue(queue_));

    context_ = other.context_;
    CL_CHECK(clRetainContext(context_));

    device_ = other.device_;
    clSetKernelArgMemPointerINTEL_ = other.clSetKernelArgMemPointerINTEL_;
}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

api::kernel_bundle_type api::build_kernel_bundle(std::string const &source) {
    return ::bbfft::cl::build_kernel_bundle(source, context_, device_);
}
api::kernel_bundle_type api::build_kernel_bundle(uint8_t const *binary, std::size_t binary_size) {
    return ::bbfft::cl::build_kernel_bundle(binary, binary_size, context_, device_);
}
api::kernel_type api::create_kernel(kernel_bundle_type b, std::string const &name) {
    return ::bbfft::cl::create_kernel(b, name);
}
std::vector<uint8_t> api::get_native_binary(kernel_bundle_type b) {
    return ::bbfft::cl::get_native_binary(b, device_);
}

cl_mem api::create_device_buffer(std::size_t bytes) {
    cl_int err;
    cl_mem buf =
        clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, bytes, nullptr, &err);
    CL_CHECK(err);
    return buf;
}

void api::setup_extensions() {
    cl_platform_id plat;
    CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr));
    clSetKernelArgMemPointerINTEL_ =
        (clSetKernelArgMemPointerINTEL_t)clGetExtensionFunctionAddressForPlatform(
            plat, "clSetKernelArgMemPointerINTEL");
    if (clSetKernelArgMemPointerINTEL_ == nullptr) {
        throw cl::error("OpenCL unified shared memory extension unavailable",
                        CL_INVALID_COMMAND_QUEUE);
    }
}

} // namespace bbfft::cl
