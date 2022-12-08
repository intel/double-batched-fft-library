// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"

namespace bbfft::ze {

api::api(ze_command_list_handle_t command_list, ze_context_handle_t context,
         ze_device_handle_t device)
    : command_list_(command_list), context_(context), device_(device),
      pool_(std::make_shared<event_pool>(context_)) {}

device_info api::info() {
    auto info = device_info{};

    ze_device_compute_properties_t p2;
    ZE_CHECK(zeDeviceGetComputeProperties(device_, &p2));

    info.max_work_group_size = p2.maxTotalGroupSize;

    info.num_subgroup_sizes =
        std::min(std::size_t(p2.numSubGroupSizes), info.subgroup_sizes.size());
    for (uint32_t i = 0; i < info.num_subgroup_sizes; ++i) {
        info.subgroup_sizes[i] = p2.subGroupSizes[i];
    }

    info.local_memory_size = p2.maxSharedLocalMemory;

    return info;
}

kernel_bundle api::build_kernel_bundle(std::string source) {
    return kernel_bundle(std::move(source), context_, device_);
}

void *api::create_device_buffer(std::size_t bytes) {
    void *buf = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &buf));
    return buf;
}

void *api::create_twiddle_table(void *twiddle_table, std::size_t bytes) {
    void *tw = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ze_event_handle_t event = pool_->create_event();
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &tw));
    ZE_CHECK(
        zeCommandListAppendMemoryCopy(command_list_, tw, twiddle_table, bytes, event, 0, nullptr));
    ZE_CHECK(zeCommandListClose(command_list_));
    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    release_event(event);
    ZE_CHECK(zeCommandListReset(command_list_));
    return tw;
}

} // namespace bbfft::ze
