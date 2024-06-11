// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/prime_factorization.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

std::pair<unsigned, double> update_factor(unsigned n, unsigned index, double const target,
                                          unsigned *factors, unsigned *workspace) {
    if (index == 1) {
        factors[0] = n;
        auto const diff = target - n;
        return {n, diff * diff};
    }
    unsigned r = iroot(n, index);
    auto p = ipow(r, index);
    std::fill(factors, factors + index, r);
    double err_best = std::numeric_limits<double>::max();
    unsigned prod_best = 0;
    for (unsigned f0 = r; f0 > 0; --f0) {
        if (n % f0 == 0) {
            auto [prod, err] =
                update_factor(n / f0, index - 1, target, factors + 1, workspace + index);
            prod *= f0;
            if (n == prod) {
                auto const diff = target - f0;
                err += diff * diff;
                if (err < err_best) {
                    workspace[0] = f0;
                    for (int i = 1; i < index; ++i) {
                        workspace[i] = factors[i];
                        err_best = err;
                        prod_best = prod;
                    }
                }
            }
        }
    }
    for (int i = 0; i < index; ++i) {
        factors[i] = workspace[i];
    }
    return {prod_best, err_best};
}

std::vector<unsigned> factor(unsigned n, unsigned index) {
    if (n == 0) {
        return std::vector<unsigned>(index, 0);
    }
    if (index == 0) {
        return {};
    }
    auto factors = std::vector<unsigned>(index);
    auto workspace = std::vector<unsigned>(index * (index + 1) / 2);
    double const target = std::pow(n, 1. / index);
    auto [p, err] = update_factor(n, index, target, factors.data(), workspace.data());
    std::sort(factors.begin(), factors.end());
    return factors;
}

} // namespace bbfft
