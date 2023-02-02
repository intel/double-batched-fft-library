// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ze_backend_bundle.hpp"
#include "bbfft/ze/online_compiler.hpp"

#include <utility>

namespace bbfft::sycl {

ze_backend_bundle::ze_backend_bundle(std::string const &source, ::sycl::context context,
                                     ::sycl::device device)
    : context_(context),
      native_module_{ze::build_kernel_bundle(
          source,
          ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::context>(context),
          ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::device>(device))},
      bundle_(::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                                         ::sycl::bundle_state::executable>(
          {native_module_, ::sycl::ext::oneapi::level_zero::ownership::transfer}, context)) {}

ze_backend_bundle::ze_backend_bundle(uint8_t const *binary, std::size_t binary_size,
                                     ::sycl::context context, ::sycl::device device)
    : context_(context),
      native_module_{ze::build_kernel_bundle(
          binary, binary_size,
          ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::context>(context),
          ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero, ::sycl::device>(device))},
      bundle_(::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                                         ::sycl::bundle_state::executable>(
          {native_module_, ::sycl::ext::oneapi::level_zero::ownership::transfer}, context)) {}

::sycl::kernel ze_backend_bundle::create_kernel(std::string const &name) {
    auto k_native = ze::create_kernel(native_module_, name);
    return ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>({bundle_, k_native},
                                                                       context_);
}

std::vector<uint8_t> ze_backend_bundle::get_binary() const {
    return ze::get_native_binary(native_module_);
}

} // namespace bbfft::sycl
