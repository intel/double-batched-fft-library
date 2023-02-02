// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/sycl/device.hpp"
#include "bbfft/cl/device.hpp"
#include "bbfft/ze/device.hpp"

namespace bbfft {

auto get_device_info(::sycl::device device) -> device_info {
    auto info = device_info{};

    info.max_work_group_size = device.get_info<::sycl::info::device::max_work_group_size>();

    auto sub_group_sizes = device.get_info<::sycl::info::device::sub_group_sizes>();
    info.num_subgroup_sizes = std::min(sub_group_sizes.size(), info.subgroup_sizes.size());
    for (uint32_t i = 0; i < info.num_subgroup_sizes; ++i) {
        info.subgroup_sizes[i] = sub_group_sizes[i];
    }

    info.local_memory_size = device.get_info<::sycl::info::device::local_mem_size>();

    return info;
}
auto get_device_id(::sycl::device device) -> uint64_t {
    auto backend = device.get_backend();
    if (backend == ::sycl::backend::ext_oneapi_level_zero) {
        return get_device_id(
            ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::device>(device));
    }
    return get_device_id(::sycl::get_native<::sycl::backend::opencl, ::sycl::device>(device));
}

} // namespace bbfft

