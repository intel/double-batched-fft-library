// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SIGNAL20220210_H
#define SIGNAL20220210_H

#include <cmath>

#ifdef CUDA
#define DEVICE __device__
#include <thrust/complex.h>
using thrust::complex;
#else
#define DEVICE
#include <complex>
using std::complex;
using std::cos;
using std::sin;
#endif

template <typename T> DEVICE T periodic_delta(int z, int N) { return z % N == 0 ? T(1.0) : T(0.0); }

template <typename T> class signal {
  private:
    unsigned int N;
    unsigned int m;
    T scale;

  public:
    DEVICE signal(unsigned int N, unsigned int m, T scale = T(1.0))
        : N(N), m(m % N), scale(scale) {}

    DEVICE T x(unsigned int n) {
        constexpr T tau = 6.28318530717958647693;
        return scale * cos((tau / N) * m * n);
    }
    DEVICE complex<T> X(unsigned int n) {
        T delta = periodic_delta<T>(n - m, N) + periodic_delta<T>(n + m, N);
        return {scale * (T(1.0) / T(2.0)) * delta, T(0.0)};
    }
};

template <typename T> class signal<complex<T>> {
  private:
    unsigned int N;
    unsigned int m;
    T scale;

  public:
    DEVICE signal(unsigned int N, unsigned int m, T scale = T(1.0))
        : N(N), m(m % N), scale(scale) {}

    DEVICE complex<T> x(unsigned int n) {
        constexpr T tau = 6.28318530717958647693;
        T arg = (tau / N) * m * n;
        return scale * complex<T>{cos(arg), sin(arg)};
    }
    DEVICE complex<T> X(unsigned int n) {
        T delta = periodic_delta<T>(n - m, N);
        return {scale * delta, T(0.0)};
    }
};

#endif // SIGNAL20220210_H
