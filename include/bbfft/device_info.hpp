// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEVICE_INFO_20220413_HPP
#define DEVICE_INFO_20220413_HPP

#include "bbfft/export.hpp"

#include <array>
#include <cstddef>

namespace bbfft {

/**
 * @brief Parameters of target device
 */
struct BBFFT_EXPORT device_info {
    std::size_t max_work_group_size = 0;       ///< maximum number of work items in work group
    std::array<std::size_t, 5> subgroup_sizes; ///< supported sub group sizes
    std::size_t num_subgroup_sizes = 0;        ///< number of entries in subgroup_sizes
    std::size_t local_memory_size = 0;         ///< size of shared local memory

    std::size_t min_subgroup_size(); ///< Minimum subgroup size
    std::size_t max_subgroup_size(); ///< Maximum subgroup size
    std::size_t register_space();    ///< Size of register file in bytes
};

} // namespace bbfft

#endif // DEVICE_INFO_20220413_HPP
