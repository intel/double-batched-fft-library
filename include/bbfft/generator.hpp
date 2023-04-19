// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENERATOR_20230202_HPP
#define GENERATOR_20230202_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace bbfft {

/**
 * @brief Generate FFT kernel code for configuration and device
 *
 * @param os Output stream (e.g. std::cout)
 * @param cfgs configurations
 * @param info Properties of target device
 *
 * @return kernel names
 */
BBFFT_EXPORT std::vector<std::string> generate_fft_kernels(std::ostream &os,
                                                           std::vector<configuration> const &cfgs,
                                                           device_info const &info);

} // namespace bbfft

#endif // GENERATOR_20230202_HPP
