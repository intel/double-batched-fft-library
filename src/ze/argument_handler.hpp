// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_ARGUMENT_HANDLER_20221129_HPP
#define ZE_ARGUMENT_HANDLER_20221129_HPP

#include "bbfft/ze/error.hpp"

#include <level_zero/ze_api.h>

namespace bbfft::ze {

class argument_handler {
  public:
    argument_handler(ze_kernel_handle_t krnl) : kernel_(krnl) {}

    template <typename T> void set_arg(unsigned index, T &arg) {
        ZE_CHECK(zeKernelSetArgumentValue(kernel_, index, sizeof(T), &arg));
    }

  private:
    ze_kernel_handle_t kernel_;
};

} // namespace bbfft::ze

#endif // ZE_ARGUMENT_HANDLER_20221129_HPP
