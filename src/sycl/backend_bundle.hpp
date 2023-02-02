// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_BACKEND_BUNDLE_20221129_HPP
#define SYCL_BACKEND_BUNDLE_20221129_HPP

#include <CL/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace bbfft::sycl {

class backend_bundle {
  public:
    virtual ~backend_bundle() {}
    virtual ::sycl::kernel create_kernel(std::string const &name) = 0;
    virtual std::vector<uint8_t> get_binary() const = 0;
};

} // namespace bbfft::sycl

#endif // SYCL_BACKEND_BUNDLE_20221129_HPP
