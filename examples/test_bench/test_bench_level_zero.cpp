// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_level_zero.hpp"
#include "bbfft/ze/error.hpp"
#include "bbfft/ze/make_plan.hpp"

#include <vector>

test_bench_level_zero::test_bench_level_zero() {
    ZE_CHECK(zeInit(0));
    uint32_t num_drivers = 1;
    ze_driver_handle_t driver;
    ZE_CHECK(zeDriverGet(&num_drivers, &driver));
    uint32_t num_devices = 1;
    ZE_CHECK(zeDeviceGet(driver, &num_devices, &device_));
    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(driver, &context_desc, &context_));
    // We take the first command queue group that supports compute
    auto ordinal = get_command_queue_group_ordinal(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    ze_command_queue_desc_t command_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                                 nullptr,
                                                 ordinal,
                                                 0,
                                                 0,
                                                 ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                                 ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &command_list_desc, &command_list_));
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
    ZE_CHECK(zeEventPoolCreate(context_, &event_pool_desc, 0, nullptr, &event_pool_));
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0,
                                  ZE_EVENT_SCOPE_FLAG_HOST

    };
    zeEventCreate(event_pool_, &event_desc, &event_);
}

test_bench_level_zero::~test_bench_level_zero() {
    zeEventDestroy(event_);
    zeEventPoolDestroy(event_pool_);
    zeCommandListDestroy(command_list_);
    zeContextDestroy(context_);
}

void *test_bench_level_zero::malloc_device(size_t bytes) {
    void *out = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &out));
    return out;
}

ze_event_handle_t test_bench_level_zero::memcpy(void *dest, const void *src, size_t bytes) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(command_list_, dest, src, bytes, event_, 0, nullptr));
    return event_;
}

void test_bench_level_zero::free(void *ptr) { ZE_CHECK(zeMemFree(context_, ptr)); }

void test_bench_level_zero::wait(ze_event_handle_t e) {
    ZE_CHECK(zeEventHostSynchronize(e, UINT64_MAX));
}
void test_bench_level_zero::release(ze_event_handle_t e) { ZE_CHECK(zeEventHostReset(e)); }
void test_bench_level_zero::wait_and_release(ze_event_handle_t e) {
    wait(e);
    release(e);
}

auto test_bench_level_zero::make_plan(bbfft::configuration const &cfg) const
    -> bbfft::plan<ze_event_handle_t> {
    return bbfft::make_plan(cfg, command_list_, context_, device_);
}

uint32_t test_bench_level_zero::get_command_queue_group_ordinal(
    ze_command_queue_group_property_flags_t flags) {
    uint32_t cmdqueue_group_count = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &cmdqueue_group_count, nullptr));
    auto cmdqueue_group_properties =
        std::vector<ze_command_queue_group_properties_t>(cmdqueue_group_count);
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &cmdqueue_group_count,
                                                    cmdqueue_group_properties.data()));

    uint32_t ordinal = cmdqueue_group_count;
    for (uint32_t i = 0; i < cmdqueue_group_count; ++i) {
        if ((~cmdqueue_group_properties[i].flags & flags) == 0) {
            ordinal = i;
            break;
        }
    }

    return ordinal;
}
