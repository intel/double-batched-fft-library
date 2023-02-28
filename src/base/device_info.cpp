// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/device_info.hpp"

#include <algorithm>
#include <stdexcept>

namespace bbfft {

std::size_t device_info::min_subgroup_size() {
    std::size_t sgs = 8;
    if (num_subgroup_sizes) {
        sgs = *std::min_element(subgroup_sizes.begin(),
                                subgroup_sizes.begin() +
                                    std::min(num_subgroup_sizes, subgroup_sizes.size()));
    }
    return sgs;
}

std::size_t device_info::max_subgroup_size() {
    std::size_t sgs = 8;
    if (num_subgroup_sizes) {
        sgs = *std::max_element(subgroup_sizes.begin(),
                                subgroup_sizes.begin() +
                                    std::min(num_subgroup_sizes, subgroup_sizes.size()));
    }
    return sgs;
}

std::size_t device_info::register_space() {
    switch (type) {
    case device_type::cpu: {
        // Assume AVX512 for now
        constexpr std::size_t num_zmm = 32;
        constexpr std::size_t zmm_bytes = 64;
        return num_zmm * zmm_bytes;
    }
    case device_type::gpu: {
        constexpr std::size_t bytes_per_reg = 32u; // Number of bytes per register
        constexpr std::size_t num_regs = 256u;     // Number of registers (with large GRF)

        std::size_t sgs = min_subgroup_size();
        std::size_t scale_bytes_per_reg =
            std::max(std::size_t(1),
                     sgs / 8u); // Assume that register width scales with minimum sub-group size
        return scale_bytes_per_reg * bytes_per_reg * num_regs;
    }
    case device_type::custom:
        throw std::runtime_error("register_space unknown for custom device");
    }
    return 0;
}

} // namespace bbfft
