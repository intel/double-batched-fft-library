// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MEM_20240610_HPP
#define MEM_20240610_HPP

#include <complex>
#include <type_traits>

namespace bbfft {

//! Memory type
enum class mem_type {
    buffer = 0x0,      ///< Buffer object (e.g. cl_mem)
    usm_pointer = 0x1, ///< Unified shared memory pointer
    svm_pointer = 0x2, ///< Shared virtual memory pointer
};

/**
 * @brief Type trait for checking whether a type is complex
 *
 * @tparam T type
 */
template <typename T> struct is_complex : public std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : public std::true_type {};
/**
 * @brief Helper template for is_complex
 *
 * @tparam T type
 */
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

/**
 * @brief Automatically determine memory type
 *
 * @tparam T type
 * @tparam Enable Used for enable_if
 */
template <typename T, typename Enable = void> struct auto_mem_type;

/**
 * @brief Check if type is a USM pointer
 *
 * @tparam T type
 * @param is_complex_v Checks if type is complex
 *
 * @return
 */
template <typename T>
constexpr bool usm_pointer_type =
    std::is_pointer_v<T> &&
    (std::is_fundamental_v<std::remove_pointer_t<T>> || is_complex_v<std::remove_pointer_t<T>>);

/**
 * @brief Auto mem type specialization for USM pointers
 *
 * @tparam T type
 */
template <typename T> struct auto_mem_type<T, std::enable_if_t<usm_pointer_type<T>>> {
    //! Memory type value
    constexpr static mem_type value = mem_type::usm_pointer;
};

/**
 * @brief Helper template for auto_mem_type
 *
 * @tparam T type
 */
template <typename T> inline constexpr auto auto_mem_type_v = auto_mem_type<T>::value;

//! Stores memory object and memory type
struct mem {
    /**
     * @brief ctor
     *
     * @tparam T type
     * @param value Memory object
     * @param type Memory object type
     */
    template <typename T>
    inline mem(T const value, mem_type type = auto_mem_type_v<T>) : value{value}, type{type} {}

    //! Memory object (either pointer or cl_mem)
    const void *value;
    //! Memory object type
    mem_type type;
};

} // namespace bbfft

#endif // MEM_20240610_HPP
