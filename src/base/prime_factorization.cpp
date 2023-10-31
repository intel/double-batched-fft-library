// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "prime_factorization.hpp"
#include "math.hpp"

#include <algorithm>

namespace bbfft {

std::vector<int> trial_division(int n) {
    auto factorization = std::vector<int>{};
    int f = 2;
    while (n > 1) {
        if (n % f == 0) {
            factorization.emplace_back(f);
            n /= f;
        } else {
            ++f;
        }
    }
    return factorization;
}

unsigned update_factor(unsigned n, unsigned index, unsigned *factors) {
    if (index == 1) {
        return 1;
    }
    unsigned r = iroot(n, index);
    auto p = ipow(r, index - 1);
    std::fill(factors, factors + (index - 1), r);
    while (n % p) {
        --factors[0];
        if (n % factors[0] == 0) {
            p = update_factor(n / factors[0], index - 1, factors + 1);
            p *= factors[0];
        } else {
            p = p / (factors[0] + 1) * factors[0];
        }
    }
    return p;
}

std::vector<unsigned> factor(unsigned n, unsigned index) {
    if (n == 0) {
        return std::vector<unsigned>(index, 0);
    }
    if (index == 0) {
        return {};
    }
    auto result = std::vector<unsigned>(index);
    auto p = update_factor(n, index, result.data());
    result.back() = n / p;
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace bbfft
