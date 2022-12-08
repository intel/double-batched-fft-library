// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/cl/online_compiler.hpp"
#include "bbfft/cl/error.hpp"

#include <cstdio>

namespace bbfft::cl {

cl_program build_kernel_bundle(std::string source, cl_context context, cl_device_id device) {
    char const *c_source = source.c_str();
    cl_int err;
    cl_program p = clCreateProgramWithSource(context, 1, &c_source, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(p, 0, nullptr, "-cl-mad-enable" /* -cl-intel-256-GRF-per-thread"*/,
                         nullptr, nullptr);
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
    return p;
}

cl_kernel create_kernel(cl_program prog, std::string name) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name.c_str(), &err);
    CL_CHECK(err);
    return k;
}

} // namespace bbfft::cl
