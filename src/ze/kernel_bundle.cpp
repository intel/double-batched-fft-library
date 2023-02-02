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

kernel_bundle::kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                             ze_context_handle_t context, ze_device_handle_t device) {
    static_assert(sizeof(size_t) == sizeof(std::size_t));
    ze_module_handle_t mod;
    ze_module_desc_t module_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                    nullptr,
                                    ZE_MODULE_FORMAT_NATIVE,
                                    binary_size,
                                    binary,
                                    nullptr,
                                    nullptr};
    ze_result_t err;
    ze_module_build_log_handle_t build_log;
    err = zeModuleCreate(context, device, &module_desc, &mod, &build_log);
    if (err != ZE_RESULT_SUCCESS) {
        std::string log;
        std::size_t log_size;
        ZE_CHECK(zeModuleBuildLogGetString(build_log, &log_size, nullptr));
        log.resize(log_size);
        ZE_CHECK(zeModuleBuildLogGetString(build_log, &log_size, log.data()));
        char what[256];
        snprintf(what, sizeof(what), "zeModuleCreate returned %s (%d).\n",
                 ze::ze_result_to_string(err), err);
        throw ze::error(std::string(what) + log, err);
    }
    ZE_CHECK(zeModuleBuildLogDestroy(build_log));

    module_ = shared_handle<ze_module_handle_t>(mod, &delete_module);
}

kernel kernel_bundle::create_kernel(std::string name) {
    auto m = module_.get();
    return kernel(ze::create_kernel(m, std::move(name)));
}

std::vector<uint8_t> kernel_bundle::get_binary() const {
    return ze::get_native_binary(module_.get());
}

} // namespace bbfft::ze
