// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_ARGUMENT_HANDLER_20240610_HPP
#define SYCL_ARGUMENT_HANDLER_20240610_HPP

#include "../cl/argument_handler.hpp"
#include "../ze/argument_handler.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/mem.hpp"
#include "bbfft/ze/error.hpp"

#include <CL/cl.h>
#include <cstddef>
#include <sycl/sycl.hpp>

namespace bbfft::sycl {

class argument_handler {
  public:
    virtual ~argument_handler() = default;
    virtual void set_arg(::sycl::kernel const &kernel, unsigned index, std::size_t size,
                         const void *value) const = 0;
    virtual void set_mem_arg(::sycl::kernel const &kernel, unsigned index, const void *value,
                             mem_type type) const = 0;
};

class argument_handler_cl : public argument_handler {
  public:
    inline argument_handler_cl(::sycl::platform const &plat)
        : cl_arg_(::sycl::get_native<::sycl::backend::opencl, ::sycl::platform>(plat)) {}

    inline void set_arg(::sycl::kernel const &kernel, unsigned index, std::size_t size,
                        void const *value) const override {
        auto native = ::sycl::get_native<::sycl::backend::opencl, ::sycl::kernel>(kernel);
        cl_arg_.set_arg(native, index, size, value);
        CL_CHECK(clReleaseKernel(native));
    }

    inline void set_mem_arg(::sycl::kernel const &kernel, unsigned index, const void *value,
                            mem_type type) const override {
        auto native_kernel = ::sycl::get_native<::sycl::backend::opencl, ::sycl::kernel>(kernel);
        cl_arg_.set_mem_arg(native_kernel, index, value, type);
        CL_CHECK(clReleaseKernel(native_kernel));
    }

  private:
    cl::argument_handler cl_arg_;
};

class argument_handler_ze : public argument_handler {
  public:
    inline void set_arg(::sycl::kernel const &kernel, unsigned index, std::size_t size,
                        void const *value) const override {
        auto native_kernel =
            ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::kernel>(kernel);
        ze::argument_handler::set_arg(native_kernel, index, size, value);
    }

    inline void set_mem_arg(::sycl::kernel const &kernel, unsigned index, const void *value,
                            mem_type type) const override {
        auto native_kernel =
            ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::kernel>(kernel);
        ze::argument_handler::set_mem_arg(native_kernel, index, value, type);
    }
};

} // namespace bbfft::sycl

#endif // SYCL_ARGUMENT_HANDLER_20240610_HPP
