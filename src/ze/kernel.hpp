// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_KERNEL_20221128_HPP
#define ZE_KERNEL_20221128_HPP

#include "bbfft/ze/error.hpp"
#include "shared_handle.hpp"

#include <level_zero/ze_api.h>

namespace bbfft::ze {

class kernel {
  public:
    kernel();
    kernel(ze_kernel_handle_t krnl);

    inline ze_kernel_handle_t get_native() const { return kernel_.get(); }

  private:
    inline static void delete_kernel(ze_kernel_handle_t k) { zeKernelDestroy(k); }

    shared_handle<ze_kernel_handle_t> kernel_;
};

} // namespace bbfft::ze

#endif // ZE_KERNEL_20221128_HPP
