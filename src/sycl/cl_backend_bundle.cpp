// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "cl_backend_bundle.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/cl/online_compiler.hpp"

#include <utility>

namespace bbfft::sycl {

cl_backend_bundle::cl_backend_bundle(std::string source, ::sycl::context context,
                                     ::sycl::device device)
    : context_(context) {
    auto native_context = ::sycl::get_native<::sycl::backend::opencl, ::sycl::context>(context);
    auto native_device = ::sycl::get_native<::sycl::backend::opencl, ::sycl::device>(device);
    backend_program_ = cl::build_kernel_bundle(std::move(source), native_context, native_device);
}

cl_backend_bundle::~cl_backend_bundle() { clReleaseProgram(backend_program_); }

::sycl::kernel cl_backend_bundle::create_kernel(std::string name) {
    auto k = cl::create_kernel(backend_program_, std::move(name));
    auto result = ::sycl::make_kernel<::sycl::backend::opencl>(k, context_);
    CL_CHECK(clReleaseKernel(k));
    return result;
}

} // namespace bbfft::sycl
