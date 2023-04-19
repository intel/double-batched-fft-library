// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/sycl/device.hpp"
#include "bbfft/cl/device.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/ze/device.hpp"

#include <CL/cl.h>

namespace bbfft {

auto get_device_info(::sycl::device device) -> device_info {
    auto info = device_info{};

    info.max_work_group_size = device.get_info<::sycl::info::device::max_work_group_size>();
    info.subgroup_sizes = device.get_info<::sycl::info::device::sub_group_sizes>();
    info.local_memory_size = device.get_info<::sycl::info::device::local_mem_size>();

    auto type = device.get_info<::sycl::info::device::device_type>();
    info.type = device_type::custom;
    if (type == ::sycl::info::device_type::cpu) {
        info.type = device_type::cpu;
    } else if (type == ::sycl::info::device_type::gpu) {
        info.type = device_type::gpu;
    }

    return info;
}
auto get_device_id(::sycl::device device) -> uint64_t {
    auto backend = device.get_backend();
    if (backend == ::sycl::backend::ext_oneapi_level_zero) {
        return get_device_id(
            ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::device>(device));
    }
    auto native_device = ::sycl::get_native<::sycl::backend::opencl, ::sycl::device>(device);
    auto result = get_device_id(native_device);
    CL_CHECK(clReleaseDevice(native_device));
    return result;
}

} // namespace bbfft

