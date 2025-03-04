// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_ARGUMENT_HANDLER_20221129_HPP
#define ZE_ARGUMENT_HANDLER_20221129_HPP

#include "bbfft/mem.hpp"
#include "bbfft/ze/error.hpp"

#include <level_zero/ze_api.h>

namespace bbfft::ze {

class argument_handler {
  public:
    inline static void set_arg(ze_kernel_handle_t kernel, unsigned index, std::size_t size,
                               const void *value) {
        ZE_CHECK(zeKernelSetArgumentValue(kernel, index, size, value));
    }

    inline static void set_mem_arg(ze_kernel_handle_t kernel, unsigned index, const void *value,
                                   mem_type) {
        set_arg(kernel, index, sizeof(value), &value);
    }
};

} // namespace bbfft::ze

#endif // ZE_ARGUMENT_HANDLER_20221129_HPP
