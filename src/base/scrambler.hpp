// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SCRAMBLER_20220714_HPP
#define SCRAMBLER_20220714_HPP

#include <cstddef>
#include <vector>

namespace bbfft {

/**
 * @brief Scrambles indices to Cooley-Tukey order.
 */
class scrambler {
  public:
    scrambler(std::vector<int> factorization);
    /**
     * @brief Computes scrambled index.
     *
     * Given the factorization N = N_1 * ... * N_d the input index is bijectively mapped to
     * index = i_1 + i_2 * N_1 + ... + i_d * N_1 * ... * N_{d-1}
     * The algorithm returns the "bit reversed" index, i.e.
     * scrambled_index = i_d + i_{d-1} * N_d + ... + i_1 * N_d * ... * N_2
     *
     * If index is a multiple of N, e.g. index = k * N + r, k > 0, 0 <= r < N, then
     * scrambled_index = k * N + scrambled(r)
     * is returned.
     *
     * @param index Input index
     *
     * @return Scrambled index
     */
    std::size_t operator()(std::size_t index);

  private:
    std::vector<int> factorization_;
};

/**
 * @brief Unscrambles indices from Cooley-Tukey order.
 */
class unscrambler {
  public:
    unscrambler(std::vector<int> factorization);
    /**
     * @brief Unscrambles index.
     *
     * Implements the inverse of scrambler. That is,
     * p = scrambler(factorization)
     * P = unscramber(factorization)
     * p(P(index)) == index
     * P(p(index)) == index
     *
     * @param index Scrambled index
     *
     * @return Unscrambled index
     */
    std::size_t operator()(std::size_t index);

  private:
    std::vector<int> factorization_;
};

} // namespace bbfft

#endif // SCRAMBLER_20220714_HPP
