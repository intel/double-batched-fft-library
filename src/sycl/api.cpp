// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"
#include "bbfft/sycl/device.hpp"

namespace bbfft::sycl {

api::api(::sycl::queue queue)
    : queue_(std::move(queue)), context_(queue_.get_context()), device_(queue_.get_device()) {}

api::api(::sycl::queue queue, ::sycl::context context, ::sycl::device device)
    : queue_(std::move(queue)), context_(std::move(context)), device_(std::move(device)) {}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

kernel_bundle api::build_kernel_bundle(std::string source) {
    return kernel_bundle(std::move(source), context_, device_);
}
kernel_bundle api::build_kernel_bundle(uint8_t const *binary, std::size_t binary_size) {
    return kernel_bundle(binary, binary_size, context_, device_);
}

void *api::create_device_buffer(std::size_t bytes) {
    return malloc_device(bytes, device_, context_);
}

void *api::create_twiddle_table(void *twiddle_table, std::size_t bytes) {
    void *tw = malloc_device(bytes, device_, context_);
    queue_.memcpy(tw, twiddle_table, bytes).wait();
    return tw;
}

} // namespace bbfft::sycl
