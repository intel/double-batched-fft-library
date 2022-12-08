// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel.hpp"

namespace bbfft::cl {

kernel::kernel() : kernel_{} {}

kernel::kernel(cl_kernel krnl) : kernel_(krnl) { CL_CHECK(clRetainKernel(kernel_)); }
kernel::~kernel() {
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
}

kernel::kernel(kernel const &other) { *this = other; }
void kernel::operator=(kernel const &other) {
    kernel_ = other.kernel_;
    CL_CHECK(clRetainKernel(kernel_));
}

} // namespace bbfft::cl
