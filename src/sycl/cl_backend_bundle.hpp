// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_CL_BACKEND_BUNDLE_20221129_HPP
#define SYCL_CL_BACKEND_BUNDLE_20221129_HPP

#include "backend_bundle.hpp"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <string>

namespace bbfft::sycl {

class cl_backend_bundle : public backend_bundle {
  public:
    cl_backend_bundle(std::string source, ::sycl::context context, ::sycl::device device);
    ~cl_backend_bundle();
    ::sycl::kernel create_kernel(std::string name) override;

  private:
    ::sycl::context context_;
    cl_program backend_program_;
};

} // namespace bbfft::sycl

#endif // SYCL_CL_BACKEND_BUNDLE_20221129_HPP
