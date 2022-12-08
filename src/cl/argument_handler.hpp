// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ARGUMENT_HANDLER_20221129_HPP
#define CL_ARGUMENT_HANDLER_20221129_HPP

#include "bbfft/cl/error.hpp"

#include <CL/cl.h>
#include <type_traits>

namespace bbfft::cl {

using clSetKernelArgMemPointerINTEL_t = cl_int (*)(cl_kernel kernel, cl_uint arg_index,
                                                   const void *arg_value);

class argument_handler {
  public:
    argument_handler(cl_kernel krnl, clSetKernelArgMemPointerINTEL_t clSetKernelArgMemPointerINTEL)
        : kernel_(krnl), clSetKernelArgMemPointerINTEL_(clSetKernelArgMemPointerINTEL) {}

    template <typename T>
    std::enable_if_t<!std::is_pointer_v<std::decay_t<T>> || std::is_same_v<std::decay_t<T>, cl_mem>,
                     void>
    set_arg(unsigned index, T &arg) {
        CL_CHECK(clSetKernelArg(kernel_, index, sizeof(T), &arg));
    }

    template <typename T>
    std::enable_if_t<std::is_pointer_v<std::decay_t<T>> && !std::is_same_v<std::decay_t<T>, cl_mem>,
                     void>
    set_arg(unsigned index, T &arg) {
        CL_CHECK(clSetKernelArgMemPointerINTEL_(kernel_, index, arg));
    }

  private:
    cl_kernel kernel_;
    clSetKernelArgMemPointerINTEL_t clSetKernelArgMemPointerINTEL_;
};

} // namespace bbfft::cl

#endif // CL_ARGUMENT_HANDLER_20221129_HPP
