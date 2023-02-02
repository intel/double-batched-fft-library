// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_DEVICE_20230131_HPP
#define ZE_DEVICE_20230131_HPP

#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <cstdint>
#include <level_zero/ze_api.h>

namespace bbfft {

/**
 * @brief Returns device info for device
 *
 * @param device device
 *
 * @return device_info
 */
BBFFT_EXPORT auto get_device_info(ze_device_handle_t device) -> device_info;
/**
 * @brief Return device id for device
 *
 * @param device device
 *
 * @return device id
 */
BBFFT_EXPORT auto get_device_id(ze_device_handle_t device) -> uint64_t;

} // namespace bbfft

#endif // ZE_DEVICE_20230131_HPP
