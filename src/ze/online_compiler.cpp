// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/ze/online_compiler.hpp"
#include "bbfft/ze/error.hpp"

#include "ocloc_api.h"
#include <cstdio>

namespace bbfft::ze {

void check_build_status(ze_module_build_log_handle_t build_log, ze_result_t err) {
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
}

ze_module_handle_t build_kernel_bundle(std::string const &source, ze_context_handle_t context,
                                       ze_device_handle_t device) {
    unsigned int num_args = 7;
    char const *argv[] = {
        "ocloc", "compile", "-internal_options", "-cl-ext=+cl_khr_fp64", "-spv_only",
        "-file", "fft.cl"};
    const uint32_t num_sources = 1;
    const uint8_t *data_sources = reinterpret_cast<const uint8_t *>(source.c_str());
    const uint64_t len_sources = source.size() + 1;
    char const *name_sources = argv[num_args - 1];
    uint32_t num_input_headers = 0;
    uint32_t num_outputs = 0;
    uint8_t **data_outputs = nullptr;
    uint64_t *len_outputs = nullptr;
    char **name_outputs = nullptr;
    oclocInvoke(num_args, argv, num_sources, &data_sources, &len_sources, &name_sources,
                num_input_headers, nullptr, nullptr, nullptr, &num_outputs, &data_outputs,
                &len_outputs, &name_outputs);

    ze_module_handle_t mod;
    ze_module_desc_t module_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                    nullptr,
                                    ZE_MODULE_FORMAT_IL_SPIRV,
                                    len_outputs[0],
                                    data_outputs[0],
                                    nullptr,
                                    nullptr};
    ze_result_t err;
    ze_module_build_log_handle_t build_log;
    err = zeModuleCreate(context, device, &module_desc, &mod, &build_log);
    check_build_status(build_log, err);
    ZE_CHECK(zeModuleBuildLogDestroy(build_log));

    oclocFreeOutput(&num_outputs, &data_outputs, &len_outputs, &name_outputs);

    return mod;
}

ze_module_handle_t build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
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
    check_build_status(build_log, err);
    ZE_CHECK(zeModuleBuildLogDestroy(build_log));

    return mod;
}

ze_kernel_handle_t create_kernel(ze_module_handle_t mod, std::string const &name) {
    char const *c_name = name.c_str();
    ze_kernel_desc_t kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, c_name};
    ze_kernel_handle_t krnl;
    ZE_CHECK(zeKernelCreate(mod, &kernel_desc, &krnl));
    return krnl;
}

std::vector<uint8_t> get_native_binary(ze_module_handle_t mod) {
    size_t size;
    ZE_CHECK(zeModuleGetNativeBinary(mod, &size, nullptr));
    auto result = std::vector<uint8_t>(size);
    ZE_CHECK(zeModuleGetNativeBinary(mod, &size, result.data()));
    return result;
}

} // namespace bbfft::ze
