// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_DEVICE_20230131_HPP
#define SYCL_DEVICE_20230131_HPP

#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <CL/sycl.hpp>
#include <cstdint>

namespace bbfft {

/**
 * @brief Returns device info for device
 *
 * @param device device
 *
 * @return device_info
 */
BBFFT_EXPORT auto get_device_info(::sycl::device device) -> device_info;
/**
 * @brief Return device id for device
 *
 * @param device device
 *
 * @return device id
 */
BBFFT_EXPORT auto get_device_id(::sycl::device device) -> uint64_t;

} // namespace bbfft

#endif // SYCL_DEVICE_20230131_HPP
