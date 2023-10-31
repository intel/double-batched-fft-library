// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MATH_20221207_HPP
#define MATH_20221207_HPP

#include <cstddef>
#include <utility>

namespace bbfft {

/**
 * @brief Computes base raised to the power of exp
 */
int ipow(int base, unsigned exp);

/**
 * @brief Returns floor(radicand^(1/index))
 *
 * Returns integer r with
 *
 * r^2 <= radicand < (r + 1)^2
 */
unsigned iroot(unsigned radicand, unsigned index);

/**
 * @brief Checks whether n is prime
 */
bool is_prime(unsigned n);

auto max_power_of_2_less_equal(std::size_t max_x) -> std::size_t;
auto min_power_of_2_greater_equal(std::size_t max_x) -> std::size_t;
template <class It, class T> constexpr T product(It it, It end, T initial_value) {
    for (; it != end; ++it) {
        initial_value = std::move(initial_value) * *it;
    }
    return initial_value;
}

} // namespace bbfft

#endif // MATH_20221207_HPP
