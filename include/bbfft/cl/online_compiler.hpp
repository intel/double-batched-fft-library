// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ONLINE_COMPILER_20221206_HPP
#define CL_ONLINE_COMPILER_20221206_HPP

#include "bbfft/export.hpp"

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>

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
BBFFT_EXPORT cl_program build_kernel_bundle(std::string const &source, cl_context context,
                                            cl_device_id device);

/**
 * @brief Build OpenCL program from native binary
 *
 * @param binary Pointer to binary blob
 * @param binary_size Size of binary blob
 * @param context OpenCL context
 * @param device OpenCL device
 *
 * @return OpenCL program
 */
BBFFT_EXPORT cl_program build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                                            cl_context context, cl_device_id device);

/**
 * @brief Create kernel from program
 *
 * @param prog OpenCL program
 * @param name Kernel name
 *
 * @return OpenCL kernel
 */
BBFFT_EXPORT cl_kernel create_kernel(cl_program prog, std::string const &name);

/**
 * @brief Returns binary blob of program
 *
 * @param p program
 * @param device OpenCL device
 *
 * @return Vector of bytes
 */
BBFFT_EXPORT std::vector<uint8_t> get_native_binary(cl_program p, cl_device_id device);

} // namespace bbfft::cl

#endif // CL_ONLINE_COMPILER_20221206_HPP
