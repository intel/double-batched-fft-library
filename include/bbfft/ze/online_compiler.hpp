// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_ONLINE_COMPILER_20221129_HPP
#define ZE_ONLINE_COMPILER_20221129_HPP

#include "bbfft/aot_cache.hpp"
#include "bbfft/export.hpp"
#include "bbfft/module_format.hpp"

#include <level_zero/ze_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace bbfft::ze {

/**
 * @brief Compile OpenCL-C code to a Level Zero module
 *
 * Compiler options are defined in the OpenCL standard:
 * https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#compiler-options
 *
 * @param source Source code
 * @param context Level Zero context
 * @param device Level Zero device
 * @param options List of compiler options
 *
 * @return Level Zero module
 */
BBFFT_EXPORT ze_module_handle_t build_kernel_bundle(std::string const &source,
                                                    ze_context_handle_t context,
                                                    ze_device_handle_t device,
                                                    std::vector<std::string> const &options = {});

/**
 * @brief Build Level Zero module from native binary
 *
 * @param binary Pointer to binary blob
 * @param binary_size Size of binary blob
 * @param format Binary format
 * @param context Level Zero context
 * @param device Level Zero device
 *
 * @return Level Zero module
 */
BBFFT_EXPORT ze_module_handle_t build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                                                    module_format format,
                                                    ze_context_handle_t context,
                                                    ze_device_handle_t device);

/**
 * @brief Create kernel from module
 *
 * @param mod Level Zero module
 * @param name Kernel name
 *
 * @return Level Zero kernel
 */
BBFFT_EXPORT ze_kernel_handle_t create_kernel(ze_module_handle_t mod, std::string const &name);

/**
 * @brief Takes OpenCL-C code and outputs SPIR-V
 *
 * @param source OpenCL-C source code
 * @param options List of compiler options
 *
 * @return binary
 */
BBFFT_EXPORT std::vector<uint8_t> compile_to_spirv(std::string const &source,
                                                   std::vector<std::string> const &options = {});
/**
 * @brief Takes OpenCL-C code and outputs the native device binary
 *
 * This function is a thin wrapper around ocloc
 *
 * @param source OpenCL-C source code
 * @param device_type Target device type; see ocloc compile --help for possible targets
 * @param options List of compiler options
 *
 * @return binary
 */
BBFFT_EXPORT std::vector<uint8_t> compile_to_native(std::string const &source,
                                                    std::string const &device_type,
                                                    std::vector<std::string> const &options = {});

/**
 * @brief Build module for ahead-of-time kernel cache (aot_cache)
 *
 * @param binary Pointer to native device binary blob
 * @param binary_size Size of native device binary blob
 * @param format Binary format
 * @param context Level Zero context
 * @param device Level Zero device
 *
 * @return ahead-of-time module
 */
BBFFT_EXPORT aot_module create_aot_module(uint8_t const *binary, std::size_t binary_size,
                                          module_format format, ze_context_handle_t context,
                                          ze_device_handle_t device);

} // namespace bbfft::ze

#endif // ZE_ONLINE_COMPILER_20221129_HPP
