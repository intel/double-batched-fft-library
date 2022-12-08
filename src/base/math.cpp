// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "math.hpp"

namespace bbfft {

auto max_power_of_2_less_equal(std::size_t max_x) -> std::size_t {
    std::size_t x2 = 1;
    while (2 * x2 <= max_x) {
        x2 *= 2;
    }
    return x2;
}

auto min_power_of_2_greater_equal(std::size_t max_x) -> std::size_t {
    std::size_t x2 = 1;
    while (x2 < max_x) {
        x2 *= 2;
    }
    return x2;
}

} // namespace bbfft
