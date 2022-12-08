// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MATH_20221207_HPP
#define MATH_20221207_HPP

#include <cstddef>

namespace bbfft {

auto max_power_of_2_less_equal(std::size_t max_x) -> std::size_t;
auto min_power_of_2_greater_equal(std::size_t max_x) -> std::size_t;

} // namespace bbfft

#endif // MATH_20221207_HPP
