// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "api.hpp"
#include "bbfft/detail/cast.hpp"
#include "bbfft/detail/compiler_options.hpp"
#include "bbfft/ze/device.hpp"
#include "bbfft/ze/online_compiler.hpp"

namespace bbfft::ze {

api::api(ze_command_list_handle_t command_list, ze_context_handle_t context,
         ze_device_handle_t device)
    : command_list_(command_list), context_(context), device_(device),
      pool_(std::make_shared<event_pool>(context_, max_num_events_we_are_ever_going_to_need)) {}

device_info api::info() { return get_device_info(device_); }

uint64_t api::device_id() { return get_device_id(device_); }

auto api::build_module(std::string const &source) -> shared_handle<module_handle_t> {
    ze_module_handle_t mod = ::bbfft::ze::build_kernel_bundle(
        source, context_, device_, detail::compiler_options, detail::required_extensions);
    return shared_handle<module_handle_t>(
        detail::cast<module_handle_t>(mod),
        [](module_handle_t mod) { zeModuleDestroy(detail::cast<ze_module_handle_t>(mod)); });
}
auto api::make_kernel_bundle(module_handle_t mod) -> kernel_bundle_type {
    return detail::cast<ze_module_handle_t>(mod);
}
auto api::create_kernel(kernel_bundle_type b, std::string const &name) -> kernel_type {
    return ::bbfft::ze::create_kernel(b, name);
}

void api::launch_kernel(kernel_type &k, std::array<std::size_t, 3> global_work_size,
                        std::array<std::size_t, 3> local_work_size, ze_event_handle_t signal_event,
                        uint32_t num_wait_events, ze_event_handle_t *wait_events) {
    ze_group_count_t launch_args;
    // FIXME: Must be divisible (or ceil)
    launch_args.groupCountX = global_work_size[0] / local_work_size[0];
    launch_args.groupCountY = global_work_size[1] / local_work_size[1];
    launch_args.groupCountZ = global_work_size[2] / local_work_size[2];
    ZE_CHECK(zeCommandListAppendLaunchKernel(command_list_, k, &launch_args, signal_event,
                                             num_wait_events, wait_events));
}

void *api::create_device_buffer(std::size_t bytes) {
    void *buf = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &buf));
    return buf;
}

void *api::create_twiddle_table(void *twiddle_table, std::size_t bytes) {
    ze_command_queue_desc_t command_list_desc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t tmp_queue;
    ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &command_list_desc, &tmp_queue));

    void *tw = nullptr;
    ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                  0, 0};
    ZE_CHECK(zeMemAllocDevice(context_, &device_mem_desc, bytes, 0, device_, &tw));
    auto event = pool_->get_event();
    ZE_CHECK(zeCommandListAppendMemoryCopy(tmp_queue, tw, twiddle_table, bytes, event, 0, nullptr));
    ZE_CHECK(zeEventHostSynchronize(event, UINT64_MAX));
    ZE_CHECK(zeEventHostReset(event));

    ZE_CHECK(zeCommandListDestroy(tmp_queue));
    return tw;
}

} // namespace bbfft::ze
