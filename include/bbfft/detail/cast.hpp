// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CAST_20230510_HPP
#define CAST_20230510_HPP

namespace bbfft::detail {

/**
 * @brief Hopefully safe cast for module handles
 *
 * @tparam To Target type
 * @tparam From Source type
 * @param v module handle with From type
 *
 * @return module handle with To type
 */
template <class To, class From> To cast(From v);
template <class To, class From> To cast(From v) {
    static_assert(sizeof(To) == sizeof(From));
    static_assert(alignof(To) == alignof(From));
    return reinterpret_cast<To>(v);
}

} // namespace bbfft::detail

#endif // CAST_20230510_HPP
