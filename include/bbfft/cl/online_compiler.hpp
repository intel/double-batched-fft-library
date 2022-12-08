// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ONLINE_COMPILER_20221206_HPP
#define CL_ONLINE_COMPILER_20221206_HPP

#include "bbfft/export.hpp"

#include <CL/cl.h>
#include <string>

namespace bbfft::cl {

/**
 * @brief Compile OpenCL-C code to an OpenCL program
 *
 * @param source Source code
 * @param context OpenCL context
 * @param device OpenCL device
 *
 * @return OpenCL program
 */
BBFFT_EXPORT cl_program build_kernel_bundle(std::string source, cl_context context,
                                            cl_device_id device);

/**
 * @brief Create kernel from program
 *
 * @param prog OpenCL program
 * @param name Kernel name
 *
 * @return OpenCL kernel
 */
BBFFT_EXPORT cl_kernel create_kernel(cl_program prog, std::string name);

} // namespace bbfft::cl

#endif // CL_ONLINE_COMPILER_20221206_HPP
