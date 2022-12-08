// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"

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

device_info api::info() {
    auto info = device_info{};
    CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(info.max_work_group_size), &info.max_work_group_size, nullptr));

    CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sizeof(info.subgroup_sizes),
                             info.subgroup_sizes.data(), &info.num_subgroup_sizes));
    info.num_subgroup_sizes /= sizeof(info.subgroup_sizes[0]);

    cl_ulong local_mem_size;
    CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size),
                             &local_mem_size, nullptr));
    info.local_memory_size = local_mem_size;

    return info;
}

kernel_bundle api::build_kernel_bundle(std::string source) {
    return kernel_bundle(std::move(source), context_, device_);
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
