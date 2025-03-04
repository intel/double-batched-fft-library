// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REFDFT_ANALYTIC_20240614_HPP
#define REFDFT_ANALYTIC_20240614_HPP

#include "refdft.hpp"

#include "bbfft/configuration.hpp"

#include <cstring>
#include <memory>

inline double periodic_delta(long n, long N) { return n % N == 0 ? 1.0 : 0.0; }
inline long mode(long m, long N) { return m % N; }
inline double arg(long n, long N, long m) {
    constexpr static double tau = 6.28318530717958647693;
    return (tau / N) * n * mode(m, N);
}

template <typename T> class refdft_analytic_c2c : public refdft {
  public:
    // constexpr static double delta = 0.1;
    constexpr static double delta = 0.0;

    refdft_analytic_c2c(long N, long first_mode, long last_mode)
        : N(N), first_mode(first_mode), last_mode(last_mode) {}

    void x(long n, long k, void *val) const override {
        auto sum = std::complex<double>(0.0, 0.0);
        for (long i = first_mode; i <= last_mode; ++i) {
            double a = arg(n, N, i + k);
            sum +=
                (1.0 + delta * (i - first_mode)) * std::complex<double>{std::cos(a), std::sin(a)};
        }
        auto s = std::complex<T>(sum);
        memcpy(val, &s, sizeof(s));
    }
    void X(long n, long k, void *val) const override {
        auto sum = std::complex<double>(0.0, 0.0);
        for (long i = first_mode; i <= last_mode; ++i) {
            sum += (1.0 + delta * (i - first_mode)) *
                   std::complex<double>{periodic_delta(n - mode(i + k, N), N), 0.0};
        }
        auto s = std::complex<T>(sum);
        memcpy(val, &s, sizeof(s));
    }
    double Linf() const override {
        double sum = 0.0;
        for (long i = first_mode; i <= last_mode; ++i) {
            sum += 1.0 + delta * (i - first_mode);
        }
        return sum;
    }

  private:
    long N, first_mode, last_mode;
};

template <typename T> class refdft_analytic_r2c : public refdft {
  public:
    constexpr static double delta = 0.1;

    refdft_analytic_r2c(long N, long first_mode, long last_mode)
        : N(N), first_mode(first_mode), last_mode(last_mode) {}

    void x(long n, long k, void *val) const override {
        double sum = 0.0;
        for (long i = first_mode; i <= last_mode; ++i) {
            double a = arg(n, N, i + k);
            sum += (1.0 + delta * (i - first_mode)) * std::cos(a);
        }
        auto s = T(sum);
        memcpy(val, &s, sizeof(s));
    }
    void X(long n, long k, void *val) const override {
        auto sum = std::complex<double>(0.0, 0.0);
        for (long i = first_mode; i <= last_mode; ++i) {
            auto md = mode(i + k, N);
            auto re = (periodic_delta(n - md, N) + periodic_delta(n + md, N)) / 2.0;
            sum += std::complex<double>{(1.0 + delta * (i - first_mode)) * re, 0.0};
        }
        auto s = std::complex<T>(sum);
        memcpy(val, &s, sizeof(s));
    }
    double Linf() const override {
        double sum = 0.0;
        for (long i = first_mode; i <= last_mode; ++i) {
            sum += 1.0 + delta * (i - first_mode);
        }
        return sum;
    }

  private:
    long N, first_mode, last_mode;
};

class refdft_analytic_factory : public refdft_factory {
  public:
    inline refdft_analytic_factory(long first_mode, long last_mode)
        : first_mode(first_mode), last_mode(last_mode) {}

    inline auto make_ref(bbfft::precision fp, bbfft::transform_type type, long N, long K) const
        -> std::unique_ptr<refdft> override {
        auto const make = [&](auto real_t) -> std::unique_ptr<refdft> {
            using T = decltype(real_t);
            if (type == bbfft::transform_type::r2c) {
                return std::make_unique<refdft_analytic_r2c<T>>(N, first_mode, last_mode);
            } else {
                return std::make_unique<refdft_analytic_c2c<T>>(N, first_mode, last_mode);
            }
        };
        if (fp == bbfft::precision::f32) {
            return make(float());
        } else if (fp == bbfft::precision::f64) {
            return make(double());
        }
        return nullptr;
    }

  private:
    long first_mode, last_mode;
};

#endif // REFDFT_ANALYTIC_20240614_HPP
