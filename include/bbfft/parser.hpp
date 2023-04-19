// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PARSER_20230419_HPP
#define PARSER_20230419_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <string_view>

namespace bbfft {

BBFFT_EXPORT configuration parse_fft_descriptor(std::string_view desc);
BBFFT_EXPORT device_info parse_device_info(std::string_view desc);

} // namespace bbfft

#endif // PARSER_20230419_HPP
