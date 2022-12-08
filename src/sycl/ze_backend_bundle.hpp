// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_ZE_BACKEND_BUNDLE_20221129_HPP
#define SYCL_ZE_BACKEND_BUNDLE_20221129_HPP

#include "backend_bundle.hpp"

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <string>

namespace bbfft::sycl {

class ze_backend_bundle : public backend_bundle {
  public:
    ze_backend_bundle(std::string source, ::sycl::context context, ::sycl::device device);
    ::sycl::kernel create_kernel(std::string name) override;

  private:
    using bundle_type = ::sycl::kernel_bundle<::sycl::bundle_state::executable>;
    ::sycl::context context_;
    ze_module_handle_t native_module_;
    bundle_type bundle_;
};

} // namespace bbfft::sycl

#endif // SYCL_ZE_BACKEND_BUNDLE_20221129_HPP
