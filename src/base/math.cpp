// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "math.hpp"

namespace bbfft {

int ipow(int base, unsigned exp) {
    if (exp == 0) {
        return 1;
    }
    int result = base;
    while (--exp) {
        result *= base;
    }
    return result;
}

unsigned iroot(unsigned radicand, unsigned index) {
    // Checks ipow(m, index) <= radicand
    auto const check_smaller_equal = [&](unsigned m, unsigned index) {
        auto p = 1;
        while (index--) {
            p *= m;
            if (p > radicand) {
                return false;
            }
        }
        return true;
    };
    unsigned l = 0;
    unsigned r = radicand + 1;
    while (l != r - 1) {
        unsigned m = (l + r) / 2;
        if (check_smaller_equal(m, index)) {
            l = m;
        } else {
            r = m;
        }
    }
    return l;
}

bool is_prime(unsigned n) {
    if (n == 2) {
        return true;
    }
    if (n < 2 || n % 2 == 0) {
        return false;
    }
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

auto max_power_of_2_less_equal(std::size_t max_x) -> std::size_t {
    std::size_t x2 = 1;
    while (2 * x2 <= max_x) {
        x2 *= 2;
    }
    return x2;
}

auto min_power_of_2_greater_equal(std::size_t max_x) -> std::size_t {
    std::size_t x2 = 1;
    while (x2 < max_x) {
        x2 *= 2;
    }
    return x2;
}

} // namespace bbfft
