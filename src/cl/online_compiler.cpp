// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/cl/online_compiler.hpp"
#include "bbfft/cl/device.hpp"
#include "bbfft/cl/error.hpp"

#include <cstdio>
#include <stdexcept>

namespace bbfft::cl {

void check_build_status(cl_program p, cl_int err, cl_device_id device) {
    if (err != CL_SUCCESS) {
        std::string log;
        std::size_t log_size;
        CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        log.resize(log_size);
        CL_CHECK(
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
        char what[256];
        snprintf(what, sizeof(what), "clBuildProgram returned %s (%d).\n",
                 cl::cl_status_to_string(err), err);
        throw cl::error(std::string(what) + log, err);
    }
}

cl_program build_kernel_bundle(std::string const &source, cl_context context, cl_device_id device) {
    char const *c_source = source.c_str();
    cl_int err;
    cl_program p = clCreateProgramWithSource(context, 1, &c_source, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(p, 0, nullptr, "-cl-mad-enable" /* -cl-intel-256-GRF-per-thread"*/,
                         nullptr, nullptr);
    check_build_status(p, err, device);
    return p;
}

cl_program build_kernel_bundle(uint8_t const *binary, std::size_t binary_size, module_format format,
                               cl_context context, cl_device_id device) {
    static_assert(sizeof(unsigned char) == sizeof(uint8_t));
    static_assert(sizeof(size_t) == sizeof(std::size_t));
    cl_int err;
    cl_program p;
    switch (format) {
    case module_format::spirv:
        p = clCreateProgramWithIL(context, binary, binary_size, &err);
        break;
    case module_format::native:
        p = clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary, nullptr, &err);
        break;
    default:
        throw std::runtime_error("Unknown module format");
    }
    CL_CHECK(err);
    err = clBuildProgram(p, 1, &device, "-cl-mad-enable", nullptr, nullptr);
    check_build_status(p, err, device);
    return p;
}

cl_kernel create_kernel(cl_program prog, std::string const &name) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name.c_str(), &err);
    CL_CHECK(err);
    return k;
}

aot_module create_aot_module(uint8_t const *binary, std::size_t binary_size, module_format format,
                             cl_context context, cl_device_id device) {

    aot_module amod;
    auto native_module = build_kernel_bundle(binary, binary_size, format, context, device);
    amod.mod =
        shared_handle<module_handle_t>(cast<module_handle_t>(native_module), [](module_handle_t m) {
            clReleaseProgram(cast<cl_program>(m));
        });
    size_t count = 0;
    CL_CHECK(clGetProgramInfo(native_module, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &count));
    auto names = std::vector<char>(count);
    CL_CHECK(clGetProgramInfo(native_module, CL_PROGRAM_KERNEL_NAMES, count, names.data(), &count));
    if (names.size() > 0) {
        char *begin = names.data();
        char *end = begin;
        // exclude terminating null char in names_end
        char *names_end = names.data() + names.size() - 1;
        for (; end != names_end; ++end) {
            if (*end == ';') {
                if (end - begin > 0) {
                    amod.kernel_names.insert(std::string(begin, end));
                }
                begin = end + 1;
            }
        }
        if (end - begin > 0) {
            amod.kernel_names.insert(std::string(begin, end));
        }
    }
    amod.device_id = get_device_id(device);
    return amod;
}

} // namespace bbfft::cl
