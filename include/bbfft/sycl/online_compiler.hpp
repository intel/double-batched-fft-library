// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_ONLINE_COMPILER_20230203_HPP
#define SYCL_ONLINE_COMPILER_20230203_HPP

#include "bbfft/export.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <CL/sycl.hpp>
#include <cstdint>
#include <string>

namespace bbfft::sycl {

/**
 * @brief Build native module of SYCL back-end
 *
 * @param source OpenCL-C code
 * @param context context
 * @param device device
 *
 * @return Handle
 */
BBFFT_EXPORT auto build_native_module(std::string const &source, ::sycl::context context,
                                      ::sycl::device device) -> module_handle_t;

/**
 * @brief Build native module of SYCL back-end from native binary
 *
 * @param binary Pointer to binary blob
 * @param binary_size Size of binary blob
 * @param context context
 * @param device device
 *
 * @return Handle
 */
BBFFT_EXPORT auto build_native_module(uint8_t const *binary, std::size_t binary_size,
                                      ::sycl::context context, ::sycl::device device)
    -> module_handle_t;

/**
 * @brief Make shared handle from native handle
 *
 * @param mod native handle
 * @param be backend
 *
 * @return Shared native handle
 */
BBFFT_EXPORT auto make_shared_handle(module_handle_t mod, ::sycl::backend be)
    -> shared_handle<module_handle_t>;

/**
 * @brief Create kernel bundle from native module
 *
 * @param native_module Native module
 * @param keep_ownership False if ownership shall be passed to SYCL kernel bundle
 * @param context context
 *
 * @return Kernel bundle
 */
BBFFT_EXPORT auto make_kernel_bundle(module_handle_t native_module, bool keep_ownership,
                                     ::sycl::context context)
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

} // namespace bbfft::sycl

#endif // SYCL_ONLINE_COMPILER_20230203_HPP
