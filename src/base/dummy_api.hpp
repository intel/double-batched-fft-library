// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DUMMY_API_20230202_HPP
#define DUMMY_API_20230202_HPP

#include "bbfft/detail/plan_impl.hpp"
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
    using plan_type = detail::plan_impl<event_type>;
    using buffer_type = void *;
    using kernel_bundle_type = int;
    using kernel_type = int;

    inline dummy_api(device_info info, std::ostream *os = nullptr)
        : info_(std::move(info)), os_(os) {}

    inline device_info info() { return info_; }
    inline uint64_t device_id() { return 0; }

    inline auto build_module(std::string const &source) -> shared_handle<module_handle_t> {
        if (os_) {
            *os_ << source;
        }
        return {};
    }
    inline auto make_kernel_bundle(module_handle_t) -> kernel_bundle_type { return {}; };
    inline auto create_kernel(kernel_bundle_type, std::string const &) -> kernel_type { return {}; }
    template <typename T>
    event_type launch_kernel(kernel_type &, std::array<std::size_t, 3>, std::array<std::size_t, 3>,
                             std::vector<event_type> const &, T) {
        return 0;
    }

    inline buffer_type create_device_buffer(std::size_t) { return nullptr; }
    template <typename T> buffer_type create_device_buffer(std::size_t) { return nullptr; }

    inline buffer_type create_twiddle_table(void *, std::size_t) { return nullptr; }
    template <typename T> inline buffer_type create_twiddle_table(std::vector<T> &) {
        return nullptr;
    }

    inline static void release_event(event_type) {}
    inline static void release_buffer(buffer_type) {}
    inline static void release_kernel(kernel_type) {}

  private:
    device_info info_;
    std::ostream *os_;
};

} // namespace bbfft

#endif // DUMMY_API_20230202_HPP
