// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_API_20220413_HPP
#define CL_API_20220413_HPP

#include "argument_handler.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/device_info.hpp"
#include "kernel.hpp"
#include "kernel_bundle.hpp"

#include <CL/cl.h>
#include <array>
#include <string>
#include <type_traits>
#include <vector>

namespace bbfft::cl {

class api {
  public:
    using event_type = cl_event;
    using buffer_type = cl_mem;
    using kernel_bundle_type = kernel_bundle;
    using kernel_type = kernel;

    api(cl_command_queue queue);
    api(cl_command_queue queue, cl_context context, cl_device_id device);
    ~api();

    api(api const &other);
    void operator=(api const &other);

    device_info info();

    kernel_bundle build_kernel_bundle(std::string source);
    template <typename T>
    cl_event launch_kernel(kernel &k, std::array<std::size_t, 3> global_work_size,
                           std::array<std::size_t, 3> local_work_size,
                           std::vector<cl_event> const &dep_events, T set_args) {
        auto k_native = k.get_native();
        auto handler = argument_handler(k_native, clSetKernelArgMemPointerINTEL_);
        set_args(handler);
        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue_, k_native, 3, nullptr, global_work_size.data(),
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

    static void release_event(cl_event e) { clReleaseEvent(e); }
    static void release_buffer(cl_mem b) { clReleaseMemObject(b); }

  private:
    void setup_extensions();

    cl_command_queue queue_;
    cl_context context_;
    cl_device_id device_;
    clSetKernelArgMemPointerINTEL_t clSetKernelArgMemPointerINTEL_;
};

} // namespace bbfft::cl

#endif // CL_API_20220413_HPP
