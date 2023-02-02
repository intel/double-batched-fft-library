// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SMALL_BATCH_FFT_GENERATOR_20230202_HPP
#define SMALL_BATCH_FFT_GENERATOR_20230202_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/device_info.hpp"
#include "bbfft/export.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace bbfft {

/**
 * @brief Configuration for small batch FFT
 *
 * @attention Do not set values directly but use ::configure_small_batch_fft
 */
struct BBFFT_EXPORT small_batch_configuration {
    int direction;                       ///< -1 or +1
    std::size_t M;                       ///< M
    std::size_t Mb;                      ///< M block size (w.r.t. to reshaped data)
    std::size_t N;                       ///< Number of points in DFT
    std::size_t Kb;                      ///< K block size
    std::size_t sgs;                     ///< sub group size
    precision fp;                        ///< floating-point precision
    transform_type type;                 ///< transform type (c2c, r2c, c2r)
    std::array<std::size_t, 3u> istride; ///< stride of input tensor
    std::array<std::size_t, 3u> ostride; ///< stride of output tensor
    bool inplace_unsupported;            ///< true if inplace not available
    char const *load_function;           ///< user provided load callback name
    char const *store_function;          ///< user provided store callback name

    std::string identifier() const; ///< convert configuration to identification string
};
/**
 * @brief Configure small batch FFT algorithm
 *
 * @param cfg configuration
 * @param info Properties of target device
 *
 * @return small_batch_configuration
 */
BBFFT_EXPORT small_batch_configuration configure_small_batch_fft(configuration const &cfg,
                                                                 device_info info);
/**
 * @brief Generate OpenCL C code for small batch FFT algorithm
 *
 * @param os Output stream (e.g. std::cout)
 * @param cfg small batch configuration
 * @param name Override default kernel name
 */
BBFFT_EXPORT void generate_small_batch_fft(std::ostream &os, small_batch_configuration const &cfg,
                                           std::string_view name = {});

/**
 * @brief Configuration for two factor FFT
 *
 * @attention Do not set values directly but use ::configure_factor2_slm_fft
 */
struct BBFFT_EXPORT factor2_slm_configuration {
    int direction;                       ///< -1 or +1
    std::size_t M;                       ///< M
    std::size_t Mb;                      ///< M block size
    std::size_t N1;                      ///< First factor in N=N1*N2
    std::size_t N2;                      ///< Second factor in N=N1*N2
    std::size_t Nb;                      ///< Number of parallel FFTs in factor
    std::size_t Kb;                      ///< K block size
    std::size_t sgs;                     ///< sub group size
    precision fp;                        ///< floating-point precision
    transform_type type;                 ///< transform type (c2c, r2c, c2r)
    std::array<std::size_t, 3u> istride; ///< stride of input tensor
    std::array<std::size_t, 3u> ostride; ///< stride of output tensor
    bool external_buffer;                ///< use global memory buffer instead of slm
    bool inplace_unsupported;            ///< true if inplace not available
    char const *load_function;           ///< user provided load callback name
    char const *store_function;          ///< user provided store callback name

    std::string identifier() const; ///< convert configuration to identification string
};
/**
 * @brief Configure two factor FFT algorithm
 *
 * @param cfg configuration
 * @param info Properties of target device
 *
 * @return factor2_slm_configuration
 */
BBFFT_EXPORT factor2_slm_configuration configure_factor2_slm_fft(configuration const &cfg,
                                                                 device_info info);
/**
 * @brief Generate OpenCL C code for two factor FFT algorithm
 *
 * @param os Output stream (e.g. std::cout)
 * @param cfg small batch configuration
 * @param name Override default kernel name
 */
BBFFT_EXPORT void generate_factor2_slm_fft(std::ostream &os, factor2_slm_configuration const &cfg,
                                           std::string_view name = {});

} // namespace bbfft

#endif // SMALL_BATCH_FFT_GENERATOR_20230202_HPP
