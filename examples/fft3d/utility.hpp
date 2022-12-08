// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMMON_20220627_HPP
#define COMMON_20220627_HPP

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace std::chrono;

template <class T>
struct is_complex
    : std::integral_constant<
          bool, std::is_same_v<std::complex<float>, typename std::remove_cv_t<T>> ||
                    std::is_same_v<std::complex<double>, typename std::remove_cv_t<T>> ||
                    std::is_same_v<std::complex<long double>, typename std::remove_cv_t<T>>> {};
template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T> struct real_type { using type = T; };
template <typename T> struct real_type<std::complex<T>> { using type = T; };
template <typename T> using real_type_t = typename real_type<T>::type;

constexpr double tau = 6.28318530717958647693;
template <typename T> double tol(std::size_t N) {
    return 1.0e2 * std::numeric_limits<T>::epsilon() * std::sqrt(N);
}
template <typename T> T periodic_delta(long z, long N) { return z % N == 0 ? T(1.0) : T(0.0); }
template <typename T> T signal(long j, long N) {
    using RT = real_type_t<T>;
    RT arg = (RT(tau) / N) * j;
    if constexpr (is_complex_v<T>) {
        return {std::cos(arg) / RT(N), std::sin(arg) / RT(N)};
    } else {
        return std::cos(arg) / RT(N);
    }
}

template <typename T> auto reference_output(long k, long N) {
    if constexpr (is_complex_v<T>) {
        using RT = real_type_t<T>;
        return T{periodic_delta<RT>(k - 1, N), RT(0.0)};
    } else {
        return std::complex<T>{(periodic_delta<T>(k - 1, N) + periodic_delta<T>(k + 1, N)) / T(2.0),
                               T(0.0)};
    }
}

template <typename T> double bench(int Ntimes, T fun, bool verbose = false) {
    double min_time = std::numeric_limits<double>::max();
    for (int i = 0; i < Ntimes; ++i) {
        auto start = high_resolution_clock::now();
        fun();
        auto end = high_resolution_clock::now();
        duration<double> time = end - start;
        min_time = std::min(min_time, time.count());
        if (verbose) {
            std::cout << i << " " << time.count() << std::endl;
        }
    }
    return min_time;
}

template <typename T>
void init(T *x, std::size_t N1, std::size_t N2, std::size_t N3, bool inplace) {
    std::size_t N1_out = inplace && !is_complex_v<T> ? 2 * (N1 / 2 + 1) : N1;
    for (std::size_t j3 = 0; j3 < N3; ++j3) {
        for (std::size_t j2 = 0; j2 < N2; ++j2) {
            for (std::size_t j1 = 0; j1 < N1; ++j1) {
                x[j1 + j2 * N1_out + j3 * N1_out * N2] =
                    signal<T>(j1, N1) * signal<T>(j2, N2) * signal<T>(j3, N3);
            }
        }
    }
}

template <typename T> bool check(T const *x, std::size_t N1, std::size_t N2, std::size_t N3) {
    std::size_t N1_out = is_complex_v<T> ? N1 : N1 / 2 + 1;
    for (std::size_t j3 = 0; j3 < N3; ++j3) {
        for (std::size_t j2 = 0; j2 < N2; ++j2) {
            for (std::size_t j1 = 0; j1 < N1_out; ++j1) {
                auto res = reinterpret_cast<std::complex<real_type_t<T>> const *>(
                    x)[j1 + j2 * N1_out + j3 * N1_out * N2];
                auto ref = reference_output<T>(j1, N1) * reference_output<T>(j2, N2) *
                           reference_output<T>(j3, N3);
                auto err = std::abs(res - ref);
                if (err > tol<real_type_t<T>>(std::max(std::max(N1, N2), N3))) {
                    std::cerr << std::setprecision(16) << "FFT error (" << j1 << ", " << j2 << ", "
                              << j3 << "): " << res << " != " << ref << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

#endif // COMMON_20220627_HPP
