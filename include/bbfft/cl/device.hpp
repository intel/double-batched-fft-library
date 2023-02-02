// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_DEVICE_20230131_HPP
#define CL_DEVICE_20230131_HPP

#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <CL/cl.h>
#include <cstdint>

namespace bbfft {

/**
 * @brief Returns device info for device
 *
 * @param device device
 *
 * @return device_info
 */
BBFFT_EXPORT auto get_device_info(cl_device_id device) -> device_info;
/**
 * @brief Return device id for device
 *
 * @param device device
 *
 * @return device id
 */
BBFFT_EXPORT auto get_device_id(cl_device_id device) -> uint64_t;

} // namespace bbfft

#endif // CL_DEVICE_20230131_HPP
