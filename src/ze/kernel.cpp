// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel.hpp"

namespace bbfft::ze {

kernel::kernel() : kernel_{} {}

kernel::kernel(ze_kernel_handle_t krnl)
    : kernel_(shared_handle<ze_kernel_handle_t>(krnl, &delete_kernel)) {}

} // namespace bbfft::ze
