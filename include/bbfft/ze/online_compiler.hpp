// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_ONLINE_COMPILER_20221129_HPP
#define ZE_ONLINE_COMPILER_20221129_HPP

#include "bbfft/export.hpp"

#include <level_zero/ze_api.h>
#include <string>

namespace bbfft::ze {

/**
 * @brief Compile OpenCL-C code to a Level Zero module
 *
 * @param source Source code
 * @param context Level Zero context
 * @param device Level Zero device
 *
 * @return Level Zero module
 */
BBFFT_EXPORT ze_module_handle_t build_kernel_bundle(std::string source, ze_context_handle_t context,
                                                    ze_device_handle_t device);

/**
 * @brief Create kernel from module
 *
 * @param mod Level Zero module
 * @param name Kernel name
 *
 * @return Level Zero kernel
 */
BBFFT_EXPORT ze_kernel_handle_t create_kernel(ze_module_handle_t mod, std::string name);

} // namespace bbfft::ze

#endif // ZE_ONLINE_COMPILER_20221129_HPP
