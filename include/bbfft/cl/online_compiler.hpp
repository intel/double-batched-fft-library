// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ONLINE_COMPILER_20221206_HPP
#define CL_ONLINE_COMPILER_20221206_HPP

#include "bbfft/aot_cache.hpp"
#include "bbfft/export.hpp"
#include "bbfft/module_format.hpp"

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>

namespace bbfft::cl {

/**
 * @brief Compile OpenCL-C code to an OpenCL program
 *
 * Compiler options are defined in the OpenCL standard:
 * https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#compiler-options
 *
 * @param source Source code
 * @param context OpenCL context
 * @param device OpenCL device
 * @param options List of compiler options
 * @param extensions List of OpenCL-C extensions
 *
 * @return OpenCL program
 */
BBFFT_EXPORT cl_program build_kernel_bundle(std::string const &source, cl_context context,
                                            cl_device_id device,
                                            std::vector<std::string> const &options = {},
                                            std::vector<std::string> const &extensions = {});

/**
 * @brief Build OpenCL program from native binary
 *
 * @param binary Pointer to binary blob
 * @param binary_size Size of binary blob
 * @param format Binary format
 * @param context OpenCL context
 * @param device OpenCL device
 *
 * @return OpenCL program
 */
BBFFT_EXPORT cl_program build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                                            module_format format, cl_context context,
                                            cl_device_id device);

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
 * @brief Build module for ahead-of-time kernel cache (aot_cache)
 *
 * @param binary Pointer to native device binary blob
 * @param binary_size Size of native device binary blob
 * @param format Binary format
 * @param context OpenCL context
 * @param device OpenCL device
 *
 * @return ahead-of-time module
 */
BBFFT_EXPORT aot_module create_aot_module(uint8_t const *binary, std::size_t binary_size,
                                          module_format format, cl_context context,
                                          cl_device_id device);

} // namespace bbfft::cl

#endif // CL_ONLINE_COMPILER_20221206_HPP
