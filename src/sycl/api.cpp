// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"

#include "bbfft/detail/compiler_options.hpp"
#include "bbfft/sycl/device.hpp"
#include "bbfft/sycl/online_compiler.hpp"

#include <CL/cl.h>
#include <level_zero/ze_api.h>

using ::sycl::backend;

namespace bbfft::sycl {

api::api(::sycl::queue queue)
    : queue_(std::move(queue)), context_(queue_.get_context()), device_(queue_.get_device()) {}

api::api(::sycl::queue queue, ::sycl::context context, ::sycl::device device)
    : queue_(std::move(queue)), context_(std::move(context)), device_(std::move(device)) {}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

auto api::build_module(std::string const &source) -> shared_handle<module_handle_t> {
    return ::bbfft::sycl::make_shared_handle(
        ::bbfft::sycl::build_native_module(source, context_, device_, detail::compiler_options,
                                           detail::required_extensions),
        queue_.get_backend());
}
auto api::make_kernel_bundle(module_handle_t mod) -> kernel_bundle_type {
    return ::bbfft::sycl::make_kernel_bundle(mod, true, context_);
}
auto api::create_kernel(kernel_bundle_type b, std::string const &name) -> kernel_type {
    return ::bbfft::sycl::create_kernel(std::move(b), name);
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
