// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_bundle.hpp"
#include "bbfft/cl/online_compiler.hpp"

namespace bbfft::cl {

kernel_bundle::kernel_bundle() : program_{} {}

kernel_bundle::kernel_bundle(std::string source, cl_context context, cl_device_id device) {
    program_ = cl::build_kernel_bundle(std::move(source), context, device);
}
kernel_bundle::~kernel_bundle() {
    if (program_) {
        clReleaseProgram(program_);
    }
}

kernel_bundle::kernel_bundle(kernel_bundle const &other) { *this = other; }
void kernel_bundle::operator=(kernel_bundle const &other) {
    program_ = other.program_;
    CL_CHECK(clRetainProgram(program_));
}

kernel kernel_bundle::create_kernel(std::string name) {
    cl_kernel k = cl::create_kernel(program_, std::move(name));
    auto result = kernel(k);
    CL_CHECK(clReleaseKernel(k));
    return result;
}

} // namespace bbfft::cl
