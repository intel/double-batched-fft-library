// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_signal.hpp"

#include "bbfft/tensor_indexer.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <ostream>

using namespace bbfft;

template <typename T> class epsilon {
  public:
    static double value() { return std::numeric_limits<T>::epsilon(); }
};

template <typename T> struct refdft_base {
    int sign;

    constexpr static T tau = static_cast<T>(6.28318530717958647693);
    double tol(std::size_t N) { return 5.0e2 * std::numeric_limits<T>::epsilon() * std::sqrt(N); }
    T periodic_delta(long z, long N) { return z % N == 0 ? T(1.0) : T(0.0); }
    long mode(long m, long N) { return m % N; }
    T arg(long j, long N, long m) { return -sign * (tau / N) * j * mode(m, N); }
};

template <typename T, transform_type TransformType> struct refdft;
template <typename T> struct refdft<T, transform_type::c2c> : refdft_base<T> {
    using in_t = std::complex<T>;
    using out_t = std::complex<T>;
    auto x(long j, long N, long m) -> in_t {
        T a = this->arg(j, N, m);
        return {std::cos(a), std::sin(a)};
    }
    auto X(long k, long N, long m) -> out_t {
        return {this->periodic_delta(k - this->mode(m, N), N), T(0.0)};
    }
};
template <typename T> struct refdft<T, transform_type::r2c> : public refdft_base<T> {
    using in_t = T;
    using out_t = std::complex<T>;
    auto x(long j, long N, long m) -> in_t { return std::cos(this->arg(j, N, m)); }
    auto X(long k, long N, long m) -> out_t {
        auto md = this->mode(m, N);
        return {(this->periodic_delta(k - md, N) + this->periodic_delta(k + md, N)) / T(2.0),
                T(0.0)};
    }
};

template <std::size_t D>
auto make_indexer(std::array<std::size_t, max_tensor_dim> const &shape,
                  std::array<std::size_t, max_tensor_dim> const &stride) {
    return bbfft::tensor_indexer<std::size_t, D, bbfft::layout::col_major>(
        bbfft::fit_array<D>(shape), bbfft::fit_array<D>(stride));
}

void test_signal_1d(void *x, configuration const &cfg, long first_mode) {
    auto const set_signal = [&](auto ref) {
        auto xv = static_cast<typename decltype(ref)::in_t *>(x);
        auto xi = make_indexer<3u>(cfg.shape, cfg.istride);
#pragma omp parallel
        for (std::size_t k = 0; k < cfg.shape[2]; ++k) {
            for (std::size_t n = 0; n < cfg.shape[1]; ++n) {
                for (std::size_t m = 0; m < cfg.shape[0]; ++m) {
                    xv[xi(m, n, k)] = ref.x(n, cfg.shape[1], first_mode + m + k);
                }
            }
        }
    };

    auto const set_signal_helper = [&](auto real_t) {
        using T = decltype(real_t);
        int sign = static_cast<int>(cfg.dir);
        if (cfg.type == transform_type::r2c) {
            set_signal(refdft<T, transform_type::r2c>{sign});
        } else {
            set_signal(refdft<T, transform_type::c2c>{sign});
        }
    };

    if (cfg.fp == precision::f32) {
        set_signal_helper(float(1.0));
    } else if (cfg.fp == precision::f64) {
        set_signal_helper(double(1.0));
    }
}

bool check_signal_1d(void *x, configuration const &cfg, long first_mode, std::ostream *os) {
    auto const check_signal = [&](auto ref) {
        constexpr std::size_t max_error = 10;
        using T = typename decltype(ref)::out_t;
        auto tolerance = ref.tol(cfg.shape[1]);
        auto xv = static_cast<T *>(x);

        auto x_shape = cfg.shape;
        if (cfg.type == transform_type::r2c) {
            x_shape[1] = x_shape[1] / 2 + 1;
        }

        bool ok = true;
        std::size_t num_err = 0;
        auto xi = make_indexer<3u>(x_shape, cfg.ostride);
#pragma omp parallel reduction(&& : ok) reduction(+ : num_err)
        for (std::size_t k = 0; k < x_shape[2]; ++k) {
            for (std::size_t n = 0; n < x_shape[1]; ++n) {
                for (std::size_t m = 0; m < x_shape[0]; ++m) {
                    auto val = xv[xi(m, n, k)];
                    auto refval = ref.X(n, cfg.shape[1], first_mode + m + k);
                    bool below_tol = std::abs(val - refval) <= tolerance;
                    ok = ok && below_tol;
                    if (os && !below_tol) {
                        if (num_err < max_error) {
                            *os << xi(m, n, k) << ": " << val << " != " << refval << std::endl;
                        }
                        ++num_err;
                        if (num_err == max_error) {
                            *os << "..." << std::endl;
                        }
                    }
                }
            }
        }
        if (num_err > max_error) {
            *os << num_err << " errors" << std::endl;
        }
        return ok;
    };
    auto const check_signal_helper = [&](auto real_t) {
        using T = decltype(real_t);
        bool ok = false;
        int sign = static_cast<int>(cfg.dir);
        if (cfg.type == transform_type::r2c) {
            ok = check_signal(refdft<T, transform_type::r2c>{sign});
        } else {
            ok = check_signal(refdft<T, transform_type::c2c>{sign});
        }
        return ok;
    };
    if (cfg.fp == precision::f32) {
        return check_signal_helper(float(1.0));
    } else if (cfg.fp == precision::f64) {
        return check_signal_helper(double(1.0));
    }
    return false;
}

