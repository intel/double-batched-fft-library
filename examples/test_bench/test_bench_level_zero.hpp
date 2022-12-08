// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_LEVEL_ZERO_20221207_HPP
#define TEST_BENCH_LEVEL_ZERO_20221207_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/plan.hpp"

#include <level_zero/ze_api.h>

class test_bench_level_zero {
  public:
    test_bench_level_zero();
    ~test_bench_level_zero();

    test_bench_level_zero(test_bench_level_zero const &) = delete;
    test_bench_level_zero &operator=(test_bench_level_zero const &) = delete;

    void *malloc_device(size_t bytes);
    template <typename T> T *malloc_device(size_t elements) {
        return (T *)malloc_device(elements * sizeof(T));
    }

    ze_event_handle_t memcpy(void *dest, const void *src, size_t bytes);
    template <typename T> ze_event_handle_t copy(T const *src, T *dest, size_t count) {
        return memcpy(dest, src, count * sizeof(T));
    }

    void free(void *ptr);

    static void wait(ze_event_handle_t e);
    static void release(ze_event_handle_t e);
    static void wait_and_release(ze_event_handle_t e);

    inline auto device() const { return device_; }
    inline auto context() const { return context_; }
    inline auto queue() const { return command_list_; }

    auto make_plan(bbfft::configuration const &cfg) const -> bbfft::plan<ze_event_handle_t>;

  private:
    uint32_t get_command_queue_group_ordinal(ze_command_queue_group_property_flags_t flags);

    ze_device_handle_t device_;
    ze_context_handle_t context_;
    ze_command_list_handle_t command_list_;
    ze_event_pool_handle_t event_pool_;
    ze_event_handle_t event_;
};

#endif // TEST_BENCH_LEVEL_ZERO_20221207_HPP
