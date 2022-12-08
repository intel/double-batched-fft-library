// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "scrambler.hpp"

namespace bbfft {

scrambler::scrambler(std::vector<int> factorization) : factorization_(std::move(factorization)) {}
std::size_t scrambler::operator()(std::size_t index) {
    auto L = factorization_.size();
    std::size_t result = 0u;
    std::size_t N = 1u;
    for (std::size_t i = 0; i < L; ++i) {
        auto Ni = factorization_[i];
        result = result * Ni + index % Ni;
        index /= Ni;
        N *= Ni;
    }
    return result + index * N;
}

unscrambler::unscrambler(std::vector<int> factorization)
    : factorization_(std::move(factorization)) {}
std::size_t unscrambler::operator()(std::size_t index) {
    auto L = factorization_.size();
    std::size_t result = 0u;
    std::size_t N = 1u;
    for (int i = L - 1; i >= 0; --i) {
        auto Ni = factorization_[i];
        result = result * Ni + index % Ni;
        index /= Ni;
        N *= Ni;
    }
    return result + index * N;
}

} // namespace bbfft
