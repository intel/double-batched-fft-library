// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"
#include "bbfft/cl/device.hpp"
#include "bbfft/cl/online_compiler.hpp"
#include "bbfft/detail/cast.hpp"
#include "bbfft/detail/compiler_options.hpp"

#include <CL/cl_ext.h>

namespace bbfft::cl {

api::api(cl_command_queue queue) : queue_(queue) {
    CL_CHECK(clRetainCommandQueue(queue_));

    CL_CHECK(
        clGetCommandQueueInfo(queue_, CL_QUEUE_CONTEXT, sizeof(cl_context), &context_, nullptr));
    CL_CHECK(clRetainContext(context_));

    CL_CHECK(
        clGetCommandQueueInfo(queue_, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_, nullptr));

    setup_arg_handler();
}
api::api(cl_command_queue queue, cl_context context, cl_device_id device)
    : queue_(queue), context_(context), device_(device) {
    CL_CHECK(clRetainCommandQueue(queue_));
    CL_CHECK(clRetainContext(context_));

    setup_arg_handler();
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
    arg_handler_ = other.arg_handler_;
}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

auto api::build_module(std::string const &source) -> shared_handle<module_handle_t> {
    cl_program mod = ::bbfft::cl::build_kernel_bundle(
        source, context_, device_, detail::compiler_options, detail::required_extensions);
    return shared_handle<module_handle_t>(
        detail::cast<module_handle_t>(mod),
        [](module_handle_t mod) { clReleaseProgram(detail::cast<cl_program>(mod)); });
}
auto api::make_kernel_bundle(module_handle_t mod) -> kernel_bundle_type {
    return detail::cast<kernel_bundle_type>(mod);
}
auto api::create_kernel(kernel_bundle_type b, std::string const &name) -> kernel_type {
    return ::bbfft::cl::create_kernel(b, name);
}

auto api::launch_kernel(kernel_type &k, std::array<std::size_t, 3> global_work_size,
                        std::array<std::size_t, 3> local_work_size,
                        std::vector<cl_event> const &dep_events) -> cl_event {
    cl_event evt;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, k, 3, nullptr, global_work_size.data(),
                                    local_work_size.data(), dep_events.size(), dep_events.data(),
                                    &evt));
    return evt;
}

cl_mem api::create_device_buffer(std::size_t bytes) {
    cl_int err;
    cl_mem buf =
        clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, bytes, nullptr, &err);
    CL_CHECK(err);
    return buf;
}

void api::setup_arg_handler() {
    cl_platform_id plat;
    CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr));
    arg_handler_ = argument_handler(plat);
}

} // namespace bbfft::cl
