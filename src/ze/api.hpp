// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_API_20220413_HPP
#define ZE_API_20220413_HPP

#include "argument_handler.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/ze/error.hpp"
#include "event_pool.hpp"
#include "kernel.hpp"
#include "kernel_bundle.hpp"

#include <array>
#include <cstdint>
#include <level_zero/ze_api.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace bbfft::ze {

class api {
  public:
    using event_type = ze_event_handle_t;
    using buffer_type = void *;
    using kernel_bundle_type = kernel_bundle;
    using kernel_type = kernel;

    api(ze_command_list_handle_t, ze_context_handle_t context, ze_device_handle_t device);

    device_info info();

    kernel_bundle build_kernel_bundle(std::string source);

    template <typename T>
    ze_event_handle_t launch_kernel(kernel &k, std::array<std::size_t, 3> global_work_size,
                                    std::array<std::size_t, 3> local_work_size,
                                    std::vector<ze_event_handle_t> const &dep_events, T set_args) {
        auto k_native = k.get_native();
        auto handler = argument_handler(k_native);
        set_args(handler);
        ze_event_handle_t event = pool_->create_event();
        ze_group_count_t launch_args;
        // FIXME: Must be divisible (or ceil)
        launch_args.groupCountX = global_work_size[0] / local_work_size[0];
        launch_args.groupCountY = global_work_size[1] / local_work_size[1];
        launch_args.groupCountZ = global_work_size[2] / local_work_size[2];
        ZE_CHECK(zeCommandListAppendLaunchKernel(
            command_list_, k_native, &launch_args, event, dep_events.size(),
            const_cast<ze_event_handle_t *>(dep_events.data())));
        return event;
    }

    void *create_device_buffer(std::size_t bytes);
    template <typename T> void *create_device_buffer(std::size_t num_T) {
        return create_device_buffer(num_T * sizeof(T));
    }

    void *create_twiddle_table(void *twiddle_table, std::size_t bytes);
    template <typename T> void *create_twiddle_table(std::vector<T> &twiddle_table) {
        return create_twiddle_table(twiddle_table.data(), twiddle_table.size() * sizeof(T));
    }

    static void release_event(ze_event_handle_t e) { zeEventDestroy(e); }
    void release_buffer(void *ptr) { zeMemFree(context_, ptr); }

  private:
    ze_command_list_handle_t command_list_;
    ze_context_handle_t context_;
    ze_device_handle_t device_;
    std::shared_ptr<event_pool> pool_;
};

} // namespace bbfft::ze

#endif // ZE_API_20220413_HPP
