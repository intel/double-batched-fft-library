// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FFT_20220517_HPP
#define FFT_20220517_HPP

#include "doctest/doctest.h"
#include <CL/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

constexpr double tau = 6.28318530717958647693;

template <typename T> double tol(std::size_t N) {
    return 1.0e2 * std::numeric_limits<T>::epsilon() * std::sqrt(N);
}

template <typename T, std::size_t D> double tol(std::array<std::size_t, D> N) {
    std::size_t NN = 1;
    for (auto n : N) {
        NN *= n;
    }
    return tol<T>(NN);
}

template <typename T> T periodic_delta(long z, long N) { return z % N == 0 ? T(1.0) : T(0.0); }

template <typename T> std::string to_string(T value) { return std::to_string(value); }

template <typename T, std::size_t D> std::string to_string(std::array<T, D> value) {
    std::string result = "{" + std::to_string(value[0]);
    for (std::size_t d = 1; d < D; ++d) {
        result += "," + std::to_string(value[d]);
    }
    result += "}";
    return result;
}

template <std::size_t D>
std::array<std::size_t, D> unflatten(std::size_t idx, std::array<std::size_t, D> const &shape) {
    std::array<std::size_t, D> unflattened_idx;
    for (std::size_t d = 0; d < D; ++d) {
        unflattened_idx[d] = idx % shape[d];
        idx /= shape[d];
    }
    return unflattened_idx;
}

template <typename F, std::size_t D, typename... T>
auto outer_product(F f, std::array<T, D>... arg) {
    auto result = f(arg[0]...);
    for (std::size_t d = 1; d < D; ++d) {
        result *= f(arg[d]...);
    }
    return result;
}

#define DOCTEST_TENSOR3_TEST(MM, NN, KK)                                                           \
    do {                                                                                           \
        for (auto kk : KK) {                                                                       \
            for (auto nn : NN) {                                                                   \
                for (auto mm : MM) {                                                               \
                    DOCTEST_SUBCASE(                                                               \
                        (to_string(mm) + "x" + to_string(nn) + "x" + to_string(kk)).c_str()) {     \
                        K = kk;                                                                    \
                        N = nn;                                                                    \
                        M = mm;                                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    } while (false)

#if defined(NO_DOUBLE_PRECISION)
#define TEST_PRECISIONS float
#else
#define TEST_PRECISIONS float, double
#endif

#endif // FFT_20220517_HPP
