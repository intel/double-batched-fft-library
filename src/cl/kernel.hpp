// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_KERNEL_20221128_HPP
#define CL_KERNEL_20221128_HPP

#include "bbfft/cl/error.hpp"

#include <CL/cl.h>

namespace bbfft::cl {

class kernel {
  public:
    kernel();
    kernel(cl_kernel krnl);
    ~kernel();

    kernel(kernel const &other);
    void operator=(kernel const &other);

    inline cl_kernel get_native() const { return kernel_; }

  private:
    cl_kernel kernel_;
};

} // namespace bbfft::cl

#endif // CL_KERNEL_20221128_HPP
