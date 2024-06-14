// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_signal.hpp"

#include "bbfft/tensor_indexer.hpp"

#include <limits>
#include <ostream>

using namespace bbfft;

template <typename T> class epsilon {
  public:
    static double value() { return std::numeric_limits<T>::epsilon(); }
};

template <std::size_t D>
auto make_indexer(std::array<std::size_t, max_tensor_dim> const &shape,
                  std::array<std::size_t, max_tensor_dim> const &stride) {
    return bbfft::tensor_indexer<std::size_t, D, bbfft::layout::col_major>(
        bbfft::fit_array<D>(shape), bbfft::fit_array<D>(stride));
}

test_bench_1d::test_bench_1d(bbfft::configuration const &cfg, refdft_factory const &factory)
    : cfg_(cfg), ref_{factory.make_ref(cfg_.fp, cfg_.type, cfg.shape[1], cfg.shape[2])} {}

void test_bench_1d::signal(void *x) const {
    auto const set_signal = [&](auto ref) {
        auto xv = static_cast<decltype(ref) *>(x);
        auto xi = make_indexer<3u>(cfg_.shape, cfg_.istride);
#pragma omp parallel for collapse(2)
        for (std::size_t k = 0; k < cfg_.shape[2]; ++k) {
            for (std::size_t n = 0; n < cfg_.shape[1]; ++n) {
                ref_->x(n, k, &xv[xi(0, n, k)]);
            }
        }
    };

    auto const set_signal_helper = [&](auto real_t) {
        using T = decltype(real_t);
        if (cfg_.type == transform_type::r2c) {
            set_signal(T());
        } else {
            set_signal(std::complex<T>());
        }
    };

    if (cfg_.fp == precision::f32) {
        set_signal_helper(float());
    } else if (cfg_.fp == precision::f64) {
        set_signal_helper(double());
    }
}

bool test_bench_1d::check(void *x, std::ostream *os) const {
    auto const check_signal = [&](auto ref) {
        using real_t = decltype(ref);
        using complex_t = std::complex<real_t>;
        const auto tolerance =
            10.0 * epsilon<decltype(ref)>::value() * std::sqrt(std::log2(cfg_.shape[1]));

        constexpr std::size_t max_error = 10;
        auto xv = static_cast<complex_t *>(x);

        auto x_shape = cfg_.shape;
        if (cfg_.type == transform_type::r2c) {
            x_shape[1] = x_shape[1] / 2 + 1;
        }

        auto const Linf = ref_->Linf();

        bool ok = true;
        std::size_t num_err = 0;
        auto xi = make_indexer<3u>(x_shape, cfg_.ostride);
#pragma omp parallel for collapse(2) reduction(&& : ok) reduction(+ : num_err)
        for (std::size_t k = 0; k < x_shape[2]; ++k) {
            for (std::size_t n = 0; n < x_shape[1]; ++n) {
                auto val = xv[xi(0, n, k)];
                auto refval = complex_t{};
                ref_->X(n, k, &refval);
                const auto err = std::abs(val - refval) / Linf;
                bool below_tol = err <= tolerance;
                ok = ok && below_tol;
                if (os && !below_tol) {
                    if (num_err < max_error) {
#pragma omp critical
                        *os << xi(0, n, k) << " (" << err << " > " << tolerance << "; " << Linf
                            << "): " << val << " != " << refval << std::endl;
                    }
                    ++num_err;
                    if (num_err == max_error) {
                        *os << "..." << std::endl;
                    }
                }
            }
        }
        if (num_err > max_error) {
            *os << num_err << " errors" << std::endl;
        }
        return ok;
    };

    if (cfg_.fp == precision::f32) {
        return check_signal(float());
    } else if (cfg_.fp == precision::f64) {
        return check_signal(double());
    }
    return false;
}

