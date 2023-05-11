// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CONFIGURATION_20220503_HPP
#define CONFIGURATION_20220503_HPP

#include "bbfft/export.hpp"
#include "bbfft/user_module.hpp"

#include <array>
#include <cstddef>
#include <iosfwd>

namespace bbfft {

/**
 * @brief Floating-point precision
 *
 * The value is the number of bytes needed to store one floating-point number,
 * e.g. static_cast<int>(precision::f32) = 4.
 */
enum class precision : int {
    f32 = 4, ///< 32-bit (single)
    f64 = 8  ///< 64-bit (double)
};

/**
 * @brief Convert C++ type to precision enum
 *
 * @tparam T float or double
 */
template <typename T> struct to_precision;
/**
 * @brief Specialization of to_precision for float
 */
template <> struct to_precision<float> {
    /**
     * @brief Precision corresponding to float.
     */
    static constexpr precision value = precision::f32;
};
/**
 * @brief Specialization of to_precision for double
 */
template <> struct to_precision<double> {
    /**
     * @brief Precision corresponding to double.
     */
    static constexpr precision value = precision::f64;
};
/**
 * @brief Convenience wrapper for to_precision
 *
 * Example: precision p = to_precision_v<double>;
 *
 * @tparam T float or double
 */
template <typename T> inline constexpr precision to_precision_v = to_precision<T>::value;

/**
 * @brief Sign in the exponential in the discrete Fourier transform.
 *
 * The convention is adopted that the forward transform as negative sign
 * and that the backward transform has positive sign.
 */
enum class direction : int {
    forward = -1, ///< Forward direction (-1)
    backward = 1  ///< Backward direction (+1)
};
/**
 * @brief Complex data and real data FFTs.
 *
 * The "standard" FFT is c2c, that is complex-valued input and complex-valued output.
 *
 * In practice, input data is often real-valued. Optimised plans for real-valued input or output
 * data can be selected with transform_type.
 */
enum class transform_type : int {
    c2c, ///< complex input, complex output
    r2c, ///< real input, complex output
    c2r  ///< complex input, real input
};
BBFFT_EXPORT char const *to_string(transform_type type); ///< Convert transform type to string

/**
 * @brief Maximum supported FFT dimension
 */
constexpr unsigned max_fft_dim = 3;
/**
 * @brief Maximum supported tensor dimension with batch indices
 */
constexpr unsigned max_tensor_dim = max_fft_dim + 2;

/**
 * @brief Compute the default strides of the input tensor.
 *
 * Let the shape be \f$(M,N_1,\dots,N_d,K)\f$, where d is the FFT dimension.
 * For **c2c** transforms the first two three entries of the stride array are given by
 * \f$s_0=1\f$ and \f$s_1=M\f$.
 * Further entries are computed with
 * \f[i=2,\dots,d+1: s_i = s_{i-1}N_{i-1}.\f]
 *
 * For out-of-place r2c transforms strides are the same as for c2c transforms.
 * For **in-place r2c** transforms we have to pad the first FFT mode such that the output data
 * fits into the input tensor. That is, we have \f$N_1' = 2 (\lfloor N_1/2\rfloor+1)\f$.
 * The stride computation goes according to the stride computation for c2c but with every instance
 * of \f$N_1\f$ replaced with \f$N_1'\f$.
 *
 * For **c2r** transform (both in-place and out-of-place) we expect that only \f$N_1''=\lfloor
 * N_1/2\rfloor+1\f$ complex numbers in the first FFT mode are stored (due to symmetry). The stride
 * computation goes according to the stride computation for c2c but with every instance of \f$N_1\f$
 * replaced with \f$N_1''\f$.
 *
 * @param dim FFT dimension
 * @param shape Input tensor Shape
 * @param type complex or real-valued FFT
 * @param inplace Is the transform in-place?
 *
 * @return Stride array
 */
BBFFT_EXPORT auto default_istride(unsigned dim,
                                  std::array<std::size_t, max_tensor_dim> const &shape,
                                  transform_type type, bool inplace)
    -> std::array<std::size_t, max_tensor_dim>;

/**
 * @brief Computes the default strides of the output tensor.
 *
 * "Reverses" the role of default_istride in the following manner:
 *
 * * **r2c:** Same as default_istride called for type = c2r.
 * * **c2r:** Same as default_istride called for type = r2c.
 * * **c2c:** Same as default_istride.
 *
 * @param dim FFT dimension
 * @param shape Output tensor Shape
 * @param type complex or real-valued FFT
 * @param inplace Is the transform in-place?
 *
 * @return Stride array
 */
BBFFT_EXPORT auto default_ostride(unsigned dim,
                                  std::array<std::size_t, max_tensor_dim> const &shape,
                                  transform_type type, bool inplace)
    -> std::array<std::size_t, max_tensor_dim>;

/**
 * @brief The unified configuration struct contains parameters for all plan types,
 *        including complex data, real data, and 1D to 3D FFTs.
 */
struct BBFFT_EXPORT configuration {
    unsigned dim; ///< FFT dimension (1,2, ..., max_fft_dim).
    std::array<std::size_t, max_tensor_dim>
        shape; /**< Shape of the \f$M \times N_1 \times \dots \times N_d \times K\f$ input tensor.
                * The FFT is taken over the N-modes, M and K are batch modes.
                * The **column-major** layout is assumed, therefore the M-mode varies fastest in
                * memory and the K-mode varies slowest in memory. Set shape array to {M, N_1, ...,
                * N_d, K}. */
    precision fp;                              ///< Floating-point precision
    direction dir = direction::forward;        ///< Forward or backward transform.
    transform_type type = transform_type::c2c; ///< Select complex data or real data FFTs.
    std::array<std::size_t, max_tensor_dim> istride = default_istride(
        dim, shape, type, true); /**< Stride used for address calculation of the input array.
                                  * For index \f$(m,n_1,...,n_d,k)\f$ the offset is calculated as
                                  * \f[m s_0 + \sum_{i=1}^dn_i s_i + k s_{d+1},\f]
                                  * where \f$s_i\f$ are the entries of *istride*.
                                  * The offset is measured in real if the input tensor is real
                                  * and measured in complex if the input tensor is complex.
                                  *
                                  * **Note:** \f$s_0\neq 1\f$ currently not supported. */
    std::array<std::size_t, max_tensor_dim> ostride = default_ostride(
        dim, shape, type, true); /**< Stride used for address calculation of the output array.
                                  * For index \f$(m,n_1,...,n_d,k)\f$ the offset is calculated as
                                  * \f[m s_0 + \sum_{i=1}^dn_i s_i + k s_{d+1},\f]
                                  * where \f$s_i\f$ are the entries of *ostride*.
                                  * The offset is measured in real if the output tensor is real
                                  * and measured in complex if the output tensor is complex.
                                  *
                                  * **Note:** \f$s_0\neq 1\f$ currently not supported. */
    user_module callbacks = {};  ///< User-provided load and store functions

    /**
     * @brief Compute and set strides from shape assuming the default data layout.
     *
     * See default_istride and default_ostride functions for a description of the default data
     * layout.
     *
     * @param inplace Set to true for in-place transform.
     */
    void set_strides_default(bool inplace);
    std::string to_string() const; ///< convert configuration to FFT descriptor
};

/**
 * @brief Output configuration as FFT descriptor
 *
 * @param os output stream
 * @param cfg configuration
 *
 * @return Reference to os
 */
BBFFT_EXPORT std::ostream &operator<<(std::ostream &os, configuration const &info);

} // namespace bbfft

#endif // CONFIGURATION_20220503_HPP
