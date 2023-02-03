// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_ONLINE_COMPILER_20230203_HPP
#define SYCL_ONLINE_COMPILER_20230203_HPP

#include "bbfft/export.hpp"

#include <CL/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace bbfft::sycl {

/**
 * @brief Compile OpenCL-C code
 *
 * @param source Source code
 * @param context context
 * @param device device
 *
 * @return Kernel bundle
 */
BBFFT_EXPORT auto build_kernel_bundle(std::string const &source, ::sycl::context context,
                                      ::sycl::device device)
    -> ::sycl::kernel_bundle<::sycl::bundle_state::executable>;

/**
 * @brief Build kernel bundle from native binary
 *
 * @param binary Pointer to binary blob
 * @param binary_size Size of binary blob
 * @param context context
 * @param device device
 *
 * @return Kernel bundle
 */
BBFFT_EXPORT auto build_kernel_bundle(uint8_t const *binary, std::size_t binary_size,
                                      ::sycl::context context, ::sycl::device device)
    -> ::sycl::kernel_bundle<::sycl::bundle_state::executable>;

/**
 * @brief Create kernel from bundle
 *
 * @param bundle kernel bundle
 * @param name Kernel name
 *
 * @return Kernel
 */
BBFFT_EXPORT auto create_kernel(::sycl::kernel_bundle<::sycl::bundle_state::executable> bundle,
                                std::string const &name) -> ::sycl::kernel;

/**
 * @brief Returns binary blob of bundle
 *
 * @param bundle kernel bundle
 * @param device device
 *
 * @return Vector of bytes
 */
BBFFT_EXPORT std::vector<uint8_t>
get_native_binary(::sycl::kernel_bundle<::sycl::bundle_state::executable> bundle,
                  ::sycl::device device);

} // namespace bbfft::sycl

#endif // SYCL_ONLINE_COMPILER_20230203_HPP
