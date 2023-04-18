// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/ze/device.hpp"
#include "bbfft/ze/error.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace bbfft {

auto get_device_info(ze_device_handle_t device) -> device_info {
    auto info = device_info{};

    ze_device_properties_t p;
    ZE_CHECK(zeDeviceGetProperties(device, &p));

    ze_device_compute_properties_t p2;
    ZE_CHECK(zeDeviceGetComputeProperties(device, &p2));

    info.max_work_group_size = p2.maxTotalGroupSize;

    info.num_subgroup_sizes =
        std::min(std::size_t(p2.numSubGroupSizes), info.subgroup_sizes.size());
    for (uint32_t i = 0; i < info.num_subgroup_sizes; ++i) {
        info.subgroup_sizes[i] = p2.subGroupSizes[i];
    }

    info.local_memory_size = p2.maxSharedLocalMemory;

    info.type = device_type::custom;
    if (p.type == ZE_DEVICE_TYPE_CPU) {
        info.type = device_type::cpu;
    } else if (p.type == ZE_DEVICE_TYPE_GPU) {
        info.type = device_type::gpu;
    }

    return info;
}
auto get_device_id(ze_device_handle_t device) -> uint64_t {
    ze_device_properties_t props;
    ZE_CHECK(zeDeviceGetProperties(device, &props));
    return props.deviceId;
}

} // namespace bbfft

