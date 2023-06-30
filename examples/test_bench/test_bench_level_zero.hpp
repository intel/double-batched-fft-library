// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_BENCH_LEVEL_ZERO_20221207_HPP
#define TEST_BENCH_LEVEL_ZERO_20221207_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/plan.hpp"
#include "bbfft/ze/make_plan.hpp"

#include <level_zero/ze_api.h>

class test_bench_level_zero_base {
  public:
    test_bench_level_zero_base();
    virtual ~test_bench_level_zero_base();

    test_bench_level_zero_base(test_bench_level_zero_base const &) = delete;
    test_bench_level_zero_base &operator=(test_bench_level_zero_base const &) = delete;

    void *malloc_device(size_t bytes);
    template <typename T> T *malloc_device(size_t elements) {
        return (T *)malloc_device(elements * sizeof(T));
    }

    virtual void memcpy(void *dest, const void *src, size_t bytes) = 0;
    template <typename T> void copy(T const *src, T *dest, size_t count) {
        memcpy(dest, src, count * sizeof(T));
    }

    void free(void *ptr);

    inline auto device() const { return device_; }
    inline auto context() const { return context_; }

  protected:
    uint32_t get_command_queue_group_ordinal(ze_command_queue_group_property_flags_t flags);

    ze_device_handle_t device_;
    ze_context_handle_t context_;
    ze_event_pool_handle_t event_pool_;
    ze_event_handle_t host_event_;
    std::array<ze_event_handle_t, 3u> events_;
};

class test_bench_level_zero_immediate : public test_bench_level_zero_base {
  public:
    test_bench_level_zero_immediate();
    ~test_bench_level_zero_immediate();

    void memcpy(void *dest, const void *src, size_t bytes);

    void setup_plan(bbfft::configuration const &cfg);
    void run_plan(void const *in, void *out, std::uint32_t ntimes = 1);

  private:
    bbfft::level_zero_plan plan_;
    ze_command_list_handle_t command_list_;
};

class test_bench_level_zero_regular : public test_bench_level_zero_base {
  public:
    test_bench_level_zero_regular();
    ~test_bench_level_zero_regular();

    void memcpy(void *dest, const void *src, size_t bytes);

    void setup_plan(bbfft::configuration const &cfg);
    void run_plan(void const *in, void *out, std::uint32_t ntimes = 1);

  private:
    bbfft::level_zero_plan plan_;
    ze_command_queue_handle_t queue_;
    ze_command_list_handle_t command_list_, copy_command_list_;
    void const *in_ = nullptr;
    void *out_ = nullptr;
    std::uint32_t ntimes_ = 0;
};

#endif // TEST_BENCH_LEVEL_ZERO_20221207_HPP
