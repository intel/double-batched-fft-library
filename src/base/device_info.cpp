// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/device_info.hpp"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace bbfft {

std::size_t device_info::min_subgroup_size() const {
    std::size_t sgs = 8;
    if (!subgroup_sizes.empty()) {
        sgs = *std::min_element(subgroup_sizes.cbegin(), subgroup_sizes.cend());
    }
    return sgs;
}

std::size_t device_info::max_subgroup_size() const {
    std::size_t sgs = 8;
    if (!subgroup_sizes.empty()) {
        sgs = *std::max_element(subgroup_sizes.cbegin(), subgroup_sizes.cend());
    }
    return sgs;
}

std::size_t device_info::register_space() const {
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

std::string device_info::to_string() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}
bool device_info::operator==(device_info const &other) const {
    return max_work_group_size == other.max_work_group_size &&
           subgroup_sizes == other.subgroup_sizes && local_memory_size == other.local_memory_size &&
           type == other.type;
}
bool device_info::operator!=(device_info const &other) const { return !(*this == other); }

std::ostream &operator<<(std::ostream &os, device_type type) {
    switch (type) {
    case device_type::gpu:
        os << "gpu";
        break;
    case device_type::cpu:
        os << "cpu";
        break;
    default:
        os << "custom";
        break;
    }
    return os;
}
std::ostream &operator<<(std::ostream &os, device_info const &info) {
    os << "{" << info.max_work_group_size << ", {";
    if (!info.subgroup_sizes.empty()) {
        auto it = info.subgroup_sizes.cbegin();
        os << *it++;
        for (; it < info.subgroup_sizes.cend(); ++it) {
            os << ", " << *it;
        }
    }
    return os << "}, " << info.local_memory_size << ", " << info.type << "}";
}

} // namespace bbfft
