// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_API_20220413_HPP
#define SYCL_API_20220413_HPP

#include "argument_handler.hpp"

#include "bbfft/detail/plan_impl.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <CL/sycl.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace bbfft::sycl {

class api {
  public:
    using event_type = ::sycl::event;
    using plan_type = detail::plan_impl<event_type>;
    using buffer_type = void *;
    using kernel_bundle_type = ::sycl::kernel_bundle<::sycl::bundle_state::executable>;
    using kernel_type = ::sycl::kernel;

    api(::sycl::queue queue);
    api(::sycl::queue queue, ::sycl::context context, ::sycl::device device);

    device_info info();
    uint64_t device_id();

    auto build_module(std::string const &source) -> shared_handle<module_handle_t>;
    auto make_kernel_bundle(module_handle_t mod) -> kernel_bundle_type;
    auto create_kernel(kernel_bundle_type b, std::string const &name) -> kernel_type;

    inline auto arg_handler() const -> argument_handler const & { return *arg_handler_; }
    auto launch_kernel(::sycl::kernel &k, std::array<std::size_t, 3> global_work_size,
                       std::array<std::size_t, 3> local_work_size,
                       std::vector<::sycl::event> const &dep_events) -> ::sycl::event;

    void *create_device_buffer(std::size_t bytes);
    template <typename T> void *create_device_buffer(std::size_t num_T) {
        return create_device_buffer(num_T * sizeof(T));
    }

    void *create_twiddle_table(void *twiddle_table, std::size_t bytes);
    template <typename T> void *create_twiddle_table(std::vector<T> &twiddle_table) {
        return create_twiddle_table(twiddle_table.data(), twiddle_table.size() * sizeof(T));
    }

    inline void release_event(event_type) {}
    inline void release_buffer(buffer_type ptr) { free(ptr, context_); }
    inline void release_kernel(kernel_type) {}

  private:
    void setup_arg_handler();

    ::sycl::queue queue_;
    ::sycl::context context_;
    ::sycl::device device_;
    std::shared_ptr<argument_handler> arg_handler_;
};

} // namespace bbfft::sycl

#endif // SYCL_API_20220413_HPP
