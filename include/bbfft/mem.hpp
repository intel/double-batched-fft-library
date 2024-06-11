// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MEM_20240610_HPP
#define MEM_20240610_HPP

#include <complex>
#include <type_traits>

namespace bbfft {

enum class mem_type {
    buffer = 0x0,      ///< Buffer object (e.g. cl_mem)
    usm_pointer = 0x1, ///< Unified shared memory pointer
    svm_pointer = 0x2, ///< Shared virtual memory pointer
};

template <typename T> struct is_complex : public std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : public std::true_type {};
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T, typename Enable = void> struct auto_mem_type;

template <typename T>
constexpr bool usm_pointer_type =
    std::is_pointer_v<T> &&
    (std::is_fundamental_v<std::remove_pointer_t<T>> || is_complex_v<std::remove_pointer_t<T>>);

template <typename T> struct auto_mem_type<T, std::enable_if_t<usm_pointer_type<T>>> {
    constexpr static mem_type value = mem_type::usm_pointer;
};

template <typename T> inline constexpr auto auto_mem_type_v = auto_mem_type<T>::value;

struct mem {
    template <typename T>
    inline mem(T const value, mem_type type = auto_mem_type_v<T>) : value{value}, type{type} {}

    const void *value;
    mem_type type;
};

} // namespace bbfft

#endif // MEM_20240610_HPP
