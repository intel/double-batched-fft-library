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

/**
 * @brief Computes floor(sqrt(n))
 */
int isqrt(int n);

/**
 * @brief Factorizes integer n in 2 factors that are ideally equal
 *
 * Examples:
 * 32: returns [4,8]
 * 36: returns [6,6]
 * 11 (prime): return [1,11]
 *
 * @param n Integer to factorize
 *
 * @return [a,b] such that n=a*b and a <= b
 */
std::pair<int, int> factor2(int n);

} // namespace bbfft

#endif // INTEGER_FACTORIZATION_20220407_HPP
