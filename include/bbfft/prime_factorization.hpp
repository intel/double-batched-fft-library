// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PRIME_FACTORIZATION_20240610_HPP
#define PRIME_FACTORIZATION_20240610_HPP

#include "bbfft/export.hpp"

#include <utility>
#include <vector>

namespace bbfft {

/**
 * @brief Prime factorization of integer n
 *
 * @param n
 *
 * @return List of prime factors
 */
BBFFT_EXPORT std::vector<int> trial_division(int n);

/**
 * @brief Factors "n" into "index" number of integers such that the product is n
 *
 * @return Vector of factors
 */
BBFFT_EXPORT std::vector<unsigned> factor(unsigned n, unsigned index);

} // namespace bbfft

#endif // PRIME_FACTORIZATION_20240610_HPP
