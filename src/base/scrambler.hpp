// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SCRAMBLER_20220714_HPP
#define SCRAMBLER_20220714_HPP

#include <cstddef>
#include <utility>
#include <vector>

namespace bbfft {

/**
 * @brief Scrambles indices to Cooley-Tukey order.
 */
template <typename T = int> class scrambler {
  public:
    scrambler(std::vector<int> factorization) : factorization_(std::move(factorization)) {}
    template <typename It>
    scrambler(It first, It last) : factorization_(std::move(first), std::move(last)) {}
    /**
     * @brief Computes scrambled index.
     *
     * Given the factorization N = N_1 * ... * N_d the input index is bijectively mapped to
     * index = i_1 + i_2 * N_1 + ... + i_d * N_1 * ... * N_{d-1}
     * The algorithm returns the "bit reversed" index, i.e.
     * scrambled_index = i_d + i_{d-1} * N_d + ... + i_1 * N_d * ... * N_2
     *
     * If index is a multiple of N and in0toN = false, e.g. index = k * N + r, k > 0, 0 <= r < N,
     * then scrambled_index = k * N + scrambled(r) is returned. If in0toN = true then the index must
     * be in [0, N)
     *
     * @param index Input index
     *
     * @return Scrambled index
     */
    T operator()(T index) const {
        if (factorization_.size() == 1 && in0toN_) {
            return index;
        }
        T result = T(0);
        int N = 1u;
        for (auto const &Ni : factorization_) {
            result = result * Ni + index % Ni;
            index = std::move(index) / Ni;
            N *= Ni;
        }
        return in0toN_ ? result : result + index * N;
    }

    void in0toN(bool in0toN) { in0toN_ = in0toN; }
    bool in0toN() { return in0toN_; }

  private:
    std::vector<int> factorization_;
    bool in0toN_ = false;
};

/**
 * @brief Unscrambles indices from Cooley-Tukey order.
 */
template <typename T = int> class unscrambler {
  public:
    unscrambler(std::vector<int> factorization) : factorization_(std::move(factorization)) {}
    template <typename It>
    unscrambler(It first, It last) : factorization_(std::move(first), std::move(last)) {}
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
    T operator()(T index) const {
        if (factorization_.size() == 1 && in0toN_) {
            return index;
        }
        T result = T(0);
        std::size_t N = 1u;
        for (auto it = factorization_.rbegin(); it != factorization_.rend(); ++it) {
            auto Ni = *it;
            result = result * Ni + index % Ni;
            index = std::move(index) / Ni;
            N *= Ni;
        }
        return in0toN_ ? result : result + index * N;
    }

    void in0toN(bool in0toN) { in0toN_ = in0toN; }
    bool in0toN() { return in0toN_; }

  private:
    std::vector<int> factorization_;
    bool in0toN_ = false;
};

} // namespace bbfft

#endif // SCRAMBLER_20220714_HPP
