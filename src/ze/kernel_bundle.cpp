// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_bundle.hpp"
#include "bbfft/ze/error.hpp"
#include "bbfft/ze/online_compiler.hpp"

#include "ocloc_api.h"
#include <cstdio>

namespace bbfft::ze {

kernel_bundle::kernel_bundle() : module_{} {}

kernel_bundle::kernel_bundle(std::string source, ze_context_handle_t context,
                             ze_device_handle_t device) {
    auto mod = build_kernel_bundle(std::move(source), context, device);
    module_ = shared_handle<ze_module_handle_t>(mod, &delete_module);
}

kernel kernel_bundle::create_kernel(std::string name) {
    auto m = module_.get();
    return kernel(ze::create_kernel(m, std::move(name)));
}

} // namespace bbfft::ze
