// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/detail/cast.hpp"
#include "bbfft/ze/device.hpp"
#include "bbfft/ze/error.hpp"
#include "bbfft/ze/online_compiler.hpp"

#include "ocloc_api.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace bbfft::ze {

std::vector<uint8_t> compile_to_spirv_or_native(std::string const &source,
                                                std::string const &device_type,
                                                std::vector<std::string> const &options,
                                                std::vector<std::string> const &extensions) {
    auto format_ext_list = [](auto const &extensions) -> std::string {
        if (extensions.empty()) {
            return {};
        }
        auto oss = std::ostringstream{};
        auto ext_it = extensions.begin();
        oss << "-cl-ext=+" << *ext_it++;
        for (; ext_it != extensions.end(); ++ext_it) {
            oss << ",+" << *ext_it;
        }
        return oss.str();
    };
    bool spv_only = device_type.empty();
    unsigned int num_args = 2;
    constexpr unsigned int max_num_args = 10;
    char const *argv[max_num_args] = {"ocloc", "compile"};
    auto ext_list = format_ext_list(extensions);
    if (!ext_list.empty()) {
        argv[num_args++] = "-internal_options";
        argv[num_args++] = ext_list.c_str();
    }
    auto cl_options = std::string{};
    if (options.size() > 0) {
        auto it = options.cbegin();
        cl_options = *it++;
        for (; it != options.cend(); ++it) {
            cl_options += " ";
            cl_options += *it;
        }
        argv[num_args++] = "-options";
        argv[num_args++] = cl_options.c_str();
    }
    if (spv_only) {
        argv[num_args++] = "-spv_only";
    } else {
        argv[num_args++] = "-device";
        argv[num_args++] = device_type.c_str();
    }
    argv[num_args++] = "-file";
    argv[num_args++] = "fft.cl";

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

    auto const ends_with = [](char const *str, char const *ending) {
        auto lstr = strlen(str);
        auto lend = strlen(ending);
        if (lend > lstr) {
            return false;
        }
        return strncmp(str + (lstr - lend), ending, lend) == 0;
    };

    constexpr uint32_t invalid_index = std::numeric_limits<uint32_t>::max();
    uint32_t log_file = invalid_index;
    uint32_t bin_file = invalid_index;
    for (uint32_t o = 0; o < num_outputs; ++o) {
        if (strcmp(name_outputs[o], "stdout.log") == 0) {
            log_file = o;
        } else if (spv_only && ends_with(name_outputs[o], ".spv")) {
            bin_file = o;
        } else if (!spv_only &&
                   (ends_with(name_outputs[o], ".bin") || ends_with(name_outputs[o], ".ar"))) {
            bin_file = o;
        }
    }
    if (bin_file == invalid_index) {
        if (log_file != invalid_index) {
            char *log_ptr = reinterpret_cast<char *>(data_outputs[log_file]);
            auto log = std::string(log_ptr, len_outputs[log_file]);
            throw std::runtime_error("source compilation failed\n" + log);
        }
        throw std::runtime_error("source compilation failed (no log available)");
    }

    auto result = std::vector<uint8_t>(data_outputs[bin_file],
                                       data_outputs[bin_file] + len_outputs[bin_file]);
    oclocFreeOutput(&num_outputs, &data_outputs, &len_outputs, &name_outputs);
    return result;
}

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
                                       ze_device_handle_t device,
                                       std::vector<std::string> const &options,
                                       std::vector<std::string> const &extensions) {
    auto spirv = compile_to_spirv_or_native(source, "", options, extensions);
    return build_kernel_bundle(spirv.data(), spirv.size(), module_format::spirv, context, device);
}

ze_module_handle_t build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                                       module_format format, ze_context_handle_t context,
                                       ze_device_handle_t device) {
    static_assert(sizeof(size_t) == sizeof(std::size_t));
    ze_module_format_t zformat;
    switch (format) {
    case module_format::spirv:
        zformat = ZE_MODULE_FORMAT_IL_SPIRV;
        break;
    case module_format::native:
        zformat = ZE_MODULE_FORMAT_NATIVE;
        break;
    default:
        throw std::runtime_error("Unknown module format");
    }
    ze_module_handle_t mod;
    ze_module_desc_t module_desc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, zformat, binary_size, binary, nullptr, nullptr};
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

    /*char const *sched_hint = getenv("BBFFT_SCHEDULING_HINT");
    if (sched_hint) {
        ze_scheduling_hint_exp_flag_t flag;
        switch (atoi(sched_hint)) {
        case 0:
            flag = ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST;
            break;
        case 1:
            flag = ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN;
            break;
        case 2:
            flag = ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN;
            break;
        default:
            flag = ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN;
            break;
        }
        ze_scheduling_hint_exp_desc_t hint = {ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC, nullptr,
                                              flag};
        ZE_CHECK(zeKernelSchedulingHintExp(krnl, &hint));
    }*/
    return krnl;
}

std::vector<uint8_t> compile_to_spirv(std::string const &source,
                                      std::vector<std::string> const &options,
                                      std::vector<std::string> const &extensions) {
    return compile_to_spirv_or_native(source, "", options, extensions);
}

std::vector<uint8_t> compile_to_native(std::string const &source, std::string const &device_type,
                                       std::vector<std::string> const &options,
                                       std::vector<std::string> const &extensions) {
    if (device_type.empty()) {
        throw std::logic_error("compile_to_native: device_type must not be empty");
    }
    return compile_to_spirv_or_native(source, device_type, options, extensions);
}

aot_module create_aot_module(uint8_t const *binary, std::size_t binary_size, module_format format,
                             ze_context_handle_t context, ze_device_handle_t device) {

    aot_module amod;
    auto native_module = build_kernel_bundle(binary, binary_size, format, context, device);
    amod.mod = shared_handle<module_handle_t>(
        detail::cast<module_handle_t>(native_module),
        [](module_handle_t m) { zeModuleDestroy(detail::cast<ze_module_handle_t>(m)); });
    uint32_t count = 0;
    ZE_CHECK(zeModuleGetKernelNames(native_module, &count, nullptr));
    auto names = std::vector<const char *>(count);
    ZE_CHECK(zeModuleGetKernelNames(native_module, &count, names.data()));
    for (auto &name : names) {
        amod.kernel_names.insert(name);
    }
    amod.device_id = get_device_id(device);
    return amod;
}

} // namespace bbfft::ze
