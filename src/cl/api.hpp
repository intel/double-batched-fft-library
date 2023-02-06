// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_API_20220413_HPP
#define CL_API_20220413_HPP

#include "argument_handler.hpp"

#include "bbfft/cl/error.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <CL/cl.h>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace bbfft::cl {

class api {
  public:
    using event_type = cl_event;
    using buffer_type = cl_mem;
    using kernel_bundle_type = cl_program;
    using kernel_type = cl_kernel;

    api(cl_command_queue queue);
    api(cl_command_queue queue, cl_context context, cl_device_id device);
    ~api();

    api(api const &other);
    void operator=(api const &other);

    device_info info();
    uint64_t device_id();

    auto build_module(std::string const &source) -> shared_handle<module_handle_t>;
    auto make_kernel_bundle(module_handle_t mod) -> kernel_bundle_type;
    auto create_kernel(kernel_bundle_type b, std::string const &name) -> kernel_type;

    template <typename T>
    cl_event launch_kernel(kernel_type &k, std::array<std::size_t, 3> global_work_size,
                           std::array<std::size_t, 3> local_work_size,
                           std::vector<cl_event> const &dep_events, T set_args) {
        auto handler = argument_handler(k, clSetKernelArgMemPointerINTEL_);
        set_args(handler);
        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue_, k, 3, nullptr, global_work_size.data(),
                                        local_work_size.data(), dep_events.size(),
                                        dep_events.data(), &evt));
        return evt;
    }

    cl_mem create_device_buffer(std::size_t bytes);
    template <typename T> cl_mem create_device_buffer(std::size_t num_T) {
        return create_device_buffer(num_T * sizeof(T));
    }

    template <typename T> cl_mem create_twiddle_table(std::vector<T> &twiddle_table) {
        cl_int err;
        cl_mem tw = clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                   twiddle_table.size() * sizeof(T), twiddle_table.data(), &err);
        CL_CHECK(err);
        return tw;
    }

    inline void release_event(event_type e) { clReleaseEvent(e); }
    inline void release_buffer(buffer_type b) { clReleaseMemObject(b); }
    inline void release_kernel(kernel_type k) { clReleaseKernel(k); }

  private:
    void setup_extensions();

    cl_command_queue queue_;
    cl_context context_;
    cl_device_id device_;
    clSetKernelArgMemPointerINTEL_t clSetKernelArgMemPointerINTEL_;
};

} // namespace bbfft::cl

#endif // CL_API_20220413_HPP
