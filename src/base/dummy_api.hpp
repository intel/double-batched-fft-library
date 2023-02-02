// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DUMMY_API_20230202_HPP
#define DUMMY_API_20230202_HPP

#include "bbfft/device_info.hpp"

#include <array>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace bbfft {

class dummy_api {
  public:
    using event_type = int;
    using buffer_type = void *;
    using kernel_bundle_type = int;
    using kernel_type = int;

    inline dummy_api(device_info info, std::ostream *os = nullptr)
        : info_(std::move(info)), os_(os) {}

    inline device_info info() { return info_; }
    inline uint64_t device_id() { return 0; }

    kernel_bundle_type build_kernel_bundle(std::string const &source) {
        if (os_) {
            *os_ << source;
        }
        return {};
    }
    inline kernel_bundle_type build_kernel_bundle(uint8_t const *, std::size_t) { return {}; }
    inline kernel_type create_kernel(kernel_bundle_type, std::string const &) { return {}; }
    inline std::vector<uint8_t> get_native_binary(kernel_bundle_type) { return {0}; }
    template <typename T>
    event_type launch_kernel(kernel_type &, std::array<std::size_t, 3>, std::array<std::size_t, 3>,
                             std::vector<event_type> const &, T) {
        return 0;
    }
    inline void barrier() {}

    inline buffer_type create_device_buffer(std::size_t) { return nullptr; }
    template <typename T> buffer_type create_device_buffer(std::size_t) { return nullptr; }

    inline buffer_type create_twiddle_table(void *, std::size_t) { return nullptr; }
    template <typename T> inline buffer_type create_twiddle_table(std::vector<T> &) {
        return nullptr;
    }

    inline static void release_event(event_type) {}
    inline static void release_buffer(buffer_type) {}
    inline static void release_kernel_bundle(kernel_bundle_type) {}
    inline static void release_kernel(kernel_type) {}

  private:
    device_info info_;
    std::ostream *os_;
};

} // namespace bbfft

#endif // DUMMY_API_20230202_HPP
