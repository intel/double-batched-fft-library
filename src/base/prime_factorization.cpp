// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "prime_factorization.hpp"

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

int isqrt(int n) {
    int a = n;
    int b = (n + 1) / 2;
    while (b < a) {
        a = b;
        b = (a * a + n) / (2 * a);
    }
    return a;
}

std::pair<int, int> factor2(int n) {
    auto r = isqrt(n);
    while (n % r) {
        --r;
    }
    return std::make_pair(r, n / r);
}

} // namespace bbfft
