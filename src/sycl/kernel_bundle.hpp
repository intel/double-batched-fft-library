// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_KERNEL_BUNDLE_20221208_HPP
#define SYCL_KERNEL_BUNDLE_20221208_HPP

#include "backend_bundle.hpp"

#include <CL/sycl.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bbfft::sycl {

class kernel_bundle {
  public:
    kernel_bundle();
    kernel_bundle(std::string source, ::sycl::context context, ::sycl::device device);
    kernel_bundle(uint8_t const *binary, std::size_t binary_size, ::sycl::context context,
                  ::sycl::device device);
    ::sycl::kernel create_kernel(std::string name);
    std::vector<uint8_t> get_binary() const;

  private:
    std::unique_ptr<backend_bundle> backend_bundle_;
};

} // namespace bbfft::sycl

#endif // SYCL_KERNEL_BUNDLE_20221208_HPP
