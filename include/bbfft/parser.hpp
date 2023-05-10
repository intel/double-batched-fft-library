// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PARSER_20230419_HPP
#define PARSER_20230419_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <string_view>

namespace bbfft {

/**
 * @brief Parses an FFT descriptor
 *
 * See user manual for a description of the input format
 *
 * @param desc descriptor
 *
 * @return configuration
 */
BBFFT_EXPORT configuration parse_fft_descriptor(std::string_view desc);
/**
 * @brief Parses a device info descriptor
 *
 * See user manual for a description of the input format
 *
 * @param desc descriptor
 *
 * @return device info
 */
BBFFT_EXPORT device_info parse_device_info(std::string_view desc);

} // namespace bbfft

#endif // PARSER_20230419_HPP
