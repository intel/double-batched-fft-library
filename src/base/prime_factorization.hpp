// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef INTEGER_FACTORIZATION_20220407_HPP
#define INTEGER_FACTORIZATION_20220407_HPP

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
std::vector<int> trial_division(int n);

std::pair<unsigned, double> update_factor(unsigned n, unsigned index, double const target,
                                          unsigned *factors, unsigned *workspace);

/**
 * @brief Factors "n" into "index" number of integers such that the product is n
 *
 * @return Vector of factors
 */
std::vector<unsigned> factor(unsigned n, unsigned index);

} // namespace bbfft

#endif // INTEGER_FACTORIZATION_20220407_HPP
