// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_bundle.hpp"
#include "cl_backend_bundle.hpp"
#include "ze_backend_bundle.hpp"

#include <utility>

namespace bbfft::sycl {

kernel_bundle::kernel_bundle() : backend_bundle_{nullptr} {}

kernel_bundle::kernel_bundle(std::string source, ::sycl::context context, ::sycl::device device) {
    auto backend = context.get_backend();
    if (backend == ::sycl::backend::ext_oneapi_level_zero) {
        backend_bundle_ = std::make_unique<ze_backend_bundle>(std::move(source), std::move(context),
                                                              std::move(device));
    } else {
        backend_bundle_ = std::make_unique<cl_backend_bundle>(std::move(source), std::move(context),
                                                              std::move(device));
    }
}

::sycl::kernel kernel_bundle::create_kernel(std::string name) {
    return backend_bundle_->create_kernel(std::move(name));
}

} // namespace bbfft::sycl
