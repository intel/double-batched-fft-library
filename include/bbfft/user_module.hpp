// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef USER_MODULE_20220519_HPP
#define USER_MODULE_20220519_HPP

#include "bbfft/export.hpp"

#include <cstddef>

namespace bbfft {

/**
 * @brief Language of user module
 */
enum class kernel_language {
    opencl_c ///< OpenCL-C language

};

/**
 * @brief Definition of user callbacks
 */
struct BBFFT_EXPORT user_module {
    char const *data = nullptr;                           ///< Source code string
    std::size_t length = 0;                               ///< Length of source code string
    char const *load_function = nullptr;                  ///< Name of load function (C string)
    char const *store_function = nullptr;                 ///< Name of store function (C string)
    kernel_language language = kernel_language::opencl_c; ///< Language

    /**
     * @brief Checks if user module is set up properly
     *
     * @return Is user module valid?
     */
    explicit operator bool() const noexcept;
};

} // namespace bbfft

#endif // USER_MODULE_20220519_HPP
