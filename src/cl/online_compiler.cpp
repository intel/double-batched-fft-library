// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/cl/online_compiler.hpp"
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

cl_program build_kernel_bundle(std::string source, cl_context context, cl_device_id device) {
    char const *c_source = source.c_str();
    cl_int err;
    cl_program p = clCreateProgramWithSource(context, 1, &c_source, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(p, 0, nullptr, "-cl-mad-enable" /* -cl-intel-256-GRF-per-thread"*/,
                         nullptr, nullptr);
    check_build_status(p, err, device);
    return p;
}

cl_program build_kernel_bundle(uint8_t const *binary, std::size_t binary_size, cl_context context,
                               cl_device_id device) {
    static_assert(sizeof(unsigned char) == sizeof(uint8_t));
    static_assert(sizeof(size_t) == sizeof(std::size_t));
    cl_int err;
    cl_program p =
        clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(p, 0, nullptr, "-cl-mad-enable", nullptr, nullptr);
    check_build_status(p, err, device);
    return p;
}

cl_kernel create_kernel(cl_program prog, std::string name) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name.c_str(), &err);
    CL_CHECK(err);
    return k;
}

std::vector<uint8_t> get_native_binary(cl_program p, cl_device_id device) {
    cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size,
                            void *param_value, size_t *param_value_size_ret);
    cl_uint num_devices;
    CL_CHECK(
        ::clGetProgramInfo(p, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, nullptr));

    auto devices = std::vector<cl_device_id>(num_devices);
    CL_CHECK(::clGetProgramInfo(p, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id),
                                devices.data(), nullptr));

    std::size_t num = 0;
    for (; num < devices.size() && devices[num] != device; ++num) {
    }
    if (num >= devices.size()) {
        throw std::runtime_error("get_native_binary: device not linked to OpenCL program");
    }

    auto sizes = std::vector<size_t>(num_devices);
    CL_CHECK(::clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t),
                                sizes.data(), nullptr));

    auto binaries = std::vector<unsigned char *>(num_devices);
    CL_CHECK(::clGetProgramInfo(p, CL_PROGRAM_BINARIES, num_devices * sizeof(unsigned char *),
                                binaries.data(), nullptr));

    return std::vector<uint8_t>(binaries[num], binaries[num] + sizes[num]);
}

} // namespace bbfft::cl
