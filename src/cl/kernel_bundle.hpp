// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_KERNEL_BUNDLE_20221208_HPP
#define CL_KERNEL_BUNDLE_20221208_HPP

#include "kernel.hpp"

#include <CL/cl.h>
#include <string>

namespace bbfft::cl {

class kernel_bundle {
  public:
    kernel_bundle();
    kernel_bundle(std::string source, cl_context context, cl_device_id device);
    ~kernel_bundle();

    kernel_bundle(kernel_bundle const &other);
    void operator=(kernel_bundle const &other);

    kernel create_kernel(std::string name);

    inline cl_program get_native() const { return program_; }

  private:
    cl_program program_;
};

} // namespace bbfft::cl

#endif // CL_KERNEL_BUNDLE_20221208_HPP
