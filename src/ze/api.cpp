// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"
#include "bbfft/ze/device.hpp"

namespace bbfft::ze {

api::api(ze_command_list_handle_t command_list, ze_context_handle_t context,
         ze_device_handle_t device)
    : command_list_(command_list), context_(context), device_(device),
      pool_(std::make_shared<event_pool>(context_)) {}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

kernel_bundle api::build_kernel_bundle(std::string source) {
    return kernel_bundle(std::move(source), context_, device_);
}
kernel_bundle api::build_kernel_bundle(uint8_t const *binary, std::size_t binary_size) {
    return kernel_bundle(binary, binary_size, context_, device_);
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
