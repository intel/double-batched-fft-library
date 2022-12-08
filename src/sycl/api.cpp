// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"

namespace bbfft::sycl {

api::api(::sycl::queue queue)
    : queue_(std::move(queue)), context_(queue_.get_context()), device_(queue_.get_device()) {}

api::api(::sycl::queue queue, ::sycl::context context, ::sycl::device device)
    : queue_(std::move(queue)), context_(std::move(context)), device_(std::move(device)) {}

device_info api::info() {
    auto info = device_info{};

    info.max_work_group_size = device_.get_info<::sycl::info::device::max_work_group_size>();

    auto sub_group_sizes = device_.get_info<::sycl::info::device::sub_group_sizes>();
    info.num_subgroup_sizes = std::min(sub_group_sizes.size(), info.subgroup_sizes.size());
    for (uint32_t i = 0; i < info.num_subgroup_sizes; ++i) {
        info.subgroup_sizes[i] = sub_group_sizes[i];
    }

    info.local_memory_size = device_.get_info<::sycl::info::device::local_mem_size>();

    return info;
}

kernel_bundle api::build_kernel_bundle(std::string source) {
    return kernel_bundle(std::move(source), context_, device_);
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
