// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/cl/device.hpp"
#include "bbfft/cl/error.hpp"

#include <CL/cl_ext.h>
#include <type_traits>

namespace bbfft {

auto get_device_info(cl_device_id device) -> device_info {
    auto info = device_info{};
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(info.max_work_group_size), &info.max_work_group_size, nullptr));

    static_assert(std::is_same_v<decltype(info.subgroup_sizes)::value_type, std::size_t>);
    std::size_t subgroup_sizes_bytes = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr,
                             &subgroup_sizes_bytes));
    info.subgroup_sizes.resize(subgroup_sizes_bytes / sizeof(std::size_t));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SUB_GROUP_SIZES_INTEL, subgroup_sizes_bytes,
                             info.subgroup_sizes.data(), &subgroup_sizes_bytes));

    cl_ulong local_mem_size;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size),
                             &local_mem_size, nullptr));
    info.local_memory_size = local_mem_size;

    cl_device_type type;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr));
    info.type = device_type::custom;
    if (type == CL_DEVICE_TYPE_CPU) {
        info.type = device_type::cpu;
    } else if (type == CL_DEVICE_TYPE_GPU) {
        info.type = device_type::gpu;
    }

    return info;
}
auto get_device_id(cl_device_id device) -> uint64_t {
    cl_uint dev_id;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_ID_INTEL, sizeof(dev_id), &dev_id, nullptr));
    return dev_id;
}

} // namespace bbfft

