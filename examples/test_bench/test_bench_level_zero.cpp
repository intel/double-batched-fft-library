// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_bench_level_zero.hpp"
#include "bbfft/ze/error.hpp"

#include <vector>

test_bench_level_zero_base::test_bench_level_zero_base() {
    ZE_CHECK(zeInit(0));
    uint32_t num_drivers = 1;
    ze_driver_handle_t driver;
    ZE_CHECK(zeDriverGet(&num_drivers, &driver));
    uint32_t num_devices = 1;
    ZE_CHECK(zeDeviceGet(driver, &num_devices, &device_));
    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(driver, &context_desc, &context_));

    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            static_cast<uint32_t>(1 + events_.size())};
    ZE_CHECK(zeEventPoolCreate(context_, &event_pool_desc, 0, nullptr, &event_pool_));
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0,
                                  ZE_EVENT_SCOPE_FLAG_HOST

    };
    zeEventCreate(event_pool_, &event_desc, &host_event_);

    event_desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
    event_desc.wait = 0;
    for (std::uint32_t i = 0; i < events_.size(); ++i) {
        event_desc.index = i + 1;
        zeEventCreate(event_pool_, &event_desc, &events_[i]);
    }
}

test_bench_level_zero_base::~test_bench_level_zero_base() {
    for (auto &event : events_) {
        zeEventDestroy(event);
    }
    zeEventDestroy(host_event_);
    zeEventPoolDestroy(event_pool_);
    zeContextDestroy(context_);
}

void *test_bench_level_zero_base::malloc_device(size_t bytes) {
    void *out = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &out));
    return out;
}

void test_bench_level_zero_base::free(void *ptr) { ZE_CHECK(zeMemFree(context_, ptr)); }

uint32_t test_bench_level_zero_base::get_command_queue_group_ordinal(
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

test_bench_level_zero_immediate::test_bench_level_zero_immediate() : plan_{nullptr} {
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
}

test_bench_level_zero_immediate::~test_bench_level_zero_immediate() {
    plan_ = bbfft::level_zero_plan{};
    zeCommandListDestroy(command_list_);
}

void test_bench_level_zero_immediate::memcpy(void *dest, const void *src, size_t bytes) {
    ZE_CHECK(
        zeCommandListAppendMemoryCopy(command_list_, dest, src, bytes, host_event_, 0, nullptr));
    ZE_CHECK(zeEventHostSynchronize(host_event_, UINT64_MAX));
    ZE_CHECK(zeEventHostReset(host_event_));
}

void test_bench_level_zero_immediate::setup_plan(bbfft::configuration const &cfg) {
    plan_ = bbfft::make_plan(cfg, command_list_, context_, device_);
}

void test_bench_level_zero_immediate::run_plan(void const *in, void *out, std::uint32_t ntimes) {
    if (ntimes > 1) {
        auto wait_event = events_[0];
        plan_.execute(in, out, wait_event);
        for (std::uint32_t n = 1; n < ntimes - 1; ++n) {
            auto next_wait_event = events_[n % events_.size()];
            plan_.execute(in, out, next_wait_event, 1, &wait_event);
            ZE_CHECK(zeCommandListAppendEventReset(command_list_, wait_event));
            wait_event = next_wait_event;
        }
        plan_.execute(in, out, host_event_, 1, &wait_event);
    } else {
        plan_.execute(in, out, host_event_);
    }
    ZE_CHECK(zeEventHostSynchronize(host_event_, UINT64_MAX));
    ZE_CHECK(zeEventHostReset(host_event_));
}

test_bench_level_zero_regular::test_bench_level_zero_regular() : plan_{nullptr} {
    // We take the first command queue group that supports compute
    auto ordinal = get_command_queue_group_ordinal(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    ze_command_queue_desc_t queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                          nullptr,
                                          ordinal,
                                          0,
                                          0,
                                          ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                          ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ZE_CHECK(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_));
    ze_command_list_desc_t command_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                                ordinal, 0};
    ZE_CHECK(zeCommandListCreate(context_, device_, &command_list_desc, &command_list_));
    ZE_CHECK(zeCommandListCreate(context_, device_, &command_list_desc, &copy_command_list_));
}

test_bench_level_zero_regular::~test_bench_level_zero_regular() {
    plan_ = bbfft::level_zero_plan{};
    zeCommandListDestroy(copy_command_list_);
    zeCommandListDestroy(command_list_);
    zeCommandQueueDestroy(queue_);
}

void test_bench_level_zero_regular::memcpy(void *dest, const void *src, size_t bytes) {
    ZE_CHECK(zeCommandListReset(copy_command_list_));
    ZE_CHECK(
        zeCommandListAppendMemoryCopy(copy_command_list_, dest, src, bytes, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListClose(copy_command_list_));
    ZE_CHECK(zeCommandQueueExecuteCommandLists(queue_, 1, &copy_command_list_, nullptr));
    ZE_CHECK(zeCommandQueueSynchronize(queue_, UINT64_MAX));
}

void test_bench_level_zero_regular::setup_plan(bbfft::configuration const &cfg) {
    plan_ = bbfft::make_plan(cfg, command_list_, context_, device_);
}

void test_bench_level_zero_regular::run_plan(void const *in, void *out, std::uint32_t ntimes) {
    if (in != in_ || out != out_ || ntimes != ntimes_) {
        ZE_CHECK(zeCommandListReset(command_list_));
        if (ntimes > 1) {
            auto wait_event = events_[0];
            plan_.execute(in, out, wait_event);
            for (std::uint32_t n = 1; n < ntimes - 1; ++n) {
                auto next_wait_event = events_[n % events_.size()];
                plan_.execute(in, out, next_wait_event, 1, &wait_event);
                ZE_CHECK(zeCommandListAppendEventReset(command_list_, wait_event));
                wait_event = next_wait_event;
            }
            plan_.execute(in, out, host_event_, 1, &wait_event);
        } else {
            plan_.execute(in, out, host_event_);
        }
        ZE_CHECK(zeCommandListClose(command_list_));
        in_ = in;
        out_ = out;
        ntimes_ = ntimes;
    }
    ZE_CHECK(zeCommandQueueExecuteCommandLists(queue_, 1, &command_list_, nullptr));
    ZE_CHECK(zeEventHostSynchronize(host_event_, UINT64_MAX));
    ZE_CHECK(zeEventHostReset(host_event_));
}

