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
    backend_device_ = ::sycl::get_native<::sycl::backend::opencl, ::sycl::device>(device);
    backend_program_ = cl::build_kernel_bundle(std::move(source), native_context, backend_device_);
}
cl_backend_bundle::cl_backend_bundle(uint8_t const *binary, std::size_t binary_size,
                                     ::sycl::context context, ::sycl::device device)
    : context_(context) {
    auto native_context = ::sycl::get_native<::sycl::backend::opencl, ::sycl::context>(context);
    backend_device_ = ::sycl::get_native<::sycl::backend::opencl, ::sycl::device>(device);
    backend_program_ =
        cl::build_kernel_bundle(binary, binary_size, native_context, backend_device_);
}

cl_backend_bundle::~cl_backend_bundle() { clReleaseProgram(backend_program_); }

::sycl::kernel cl_backend_bundle::create_kernel(std::string name) {
    auto k = cl::create_kernel(backend_program_, std::move(name));
    auto result = ::sycl::make_kernel<::sycl::backend::opencl>(k, context_);
    CL_CHECK(clReleaseKernel(k));
    return result;
}

std::vector<uint8_t> cl_backend_bundle::get_binary() const {
    return cl::get_native_binary(backend_program_, backend_device_);
}

} // namespace bbfft::sycl
