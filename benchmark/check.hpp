// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CHECK_20220318_H
#define CHECK_20220318_H

#include "real_type.hpp"
#include "signal.hpp"
#include "tensor.hpp"

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>

template <typename T> struct tol {
    static constexpr auto value = 1.0e3 * std::numeric_limits<T>::epsilon();
};
template <typename T> inline constexpr auto tol_v = tol<T>::value;

namespace detail {
template <typename U, typename V, bool Inverse = false>
bool check(unsigned int N, tensor<V, 3u> data) {
    using real_t = typename real_type<U>::type;
    auto [K, NN, M] = data.shape();
    for (unsigned int k = 0; k < K; ++k) {
        for (unsigned int m = 0; m < M; ++m) {
            auto s = signal<U>(N, m + k, real_t(1.0) + k / real_t(K));
            for (unsigned int n = 0; n < NN; ++n) {
                V ref;
                if constexpr (Inverse) {
                    ref = s.x(n);
                } else {
                    ref = s.X(n);
                }
                auto err = std::abs(data(k, n, m) - ref);
                if (err > std::sqrt(N) * tol_v<real_t>) {
                    auto old_prec = std::cerr.precision();
                    std::cerr << std::setprecision(16) << "FFT error (" << m << ", " << n << ", "
                              << k << "): " << data(k, n, m) << " != " << ref << std::endl;
                    std::cerr << "data: ";
                    for (unsigned int n = 0; n < NN; ++n) {
                        std::cerr << data(k, n, m) << " ";
                    }
                    std::cerr << std::endl << std::setprecision(old_prec);
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace detail

template <typename U, typename V> bool check(tensor<U, 3u> x, tensor<V, 3u> X, bool inverse) {
    auto N = x.shape(1);
    return inverse ? detail::check<U, U, true>(N, x) : detail::check<U, V, false>(N, X);
}

#endif // CHECK_20220318_H
