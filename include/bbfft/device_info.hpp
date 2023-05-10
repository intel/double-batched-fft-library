// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEVICE_INFO_20220413_HPP
#define DEVICE_INFO_20220413_HPP

#include "bbfft/export.hpp"

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace bbfft {

/**
 * @brief Device type
 */
enum class BBFFT_EXPORT device_type {
    gpu,   ///< GPU device
    cpu,   ///< CPU device
    custom ///< custom device
};

/**
 * @brief Parameters of target device
 */
struct BBFFT_EXPORT device_info {
    std::size_t max_work_group_size = 0;     ///< maximum number of work items in work group
    std::vector<std::size_t> subgroup_sizes; ///< supported sub group sizes
    std::size_t local_memory_size = 0;       ///< size of shared local memory
    device_type type = device_type::gpu;     ///< device type

    std::size_t min_subgroup_size() const;           ///< Minimum subgroup size
    std::size_t max_subgroup_size() const;           ///< Maximum subgroup size
    std::size_t register_space_min() const;          ///< Min size of register file in bytes
    std::size_t register_space_max() const;          ///< Max size of register file in bytes
    std::string to_string() const;                   ///< convert device info to string
    bool operator==(device_info const &other) const; ///< equality check
    bool operator!=(device_info const &other) const; ///< inequality check
};

/**
 * @brief Convert device type to string
 *
 * @param os output stream
 * @param type device type
 *
 * @return Reference to os
 */
BBFFT_EXPORT std::ostream &operator<<(std::ostream &os, device_type type);
/**
 * @brief Convert device info to string
 *
 * @param os output stream
 * @param info device info
 *
 * @return Reference to os
 */
BBFFT_EXPORT std::ostream &operator<<(std::ostream &os, device_info const &info);

} // namespace bbfft

#endif // DEVICE_INFO_20220413_HPP
