// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MIXED_RADIX_FFT_20220428_HPP
#define MIXED_RADIX_FFT_20220428_HPP

#include "bbfft/configuration.hpp"
#include "clir/builder.hpp"
#include "clir/expr.hpp"
#include "clir/var.hpp"
#include "generator/utility.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace bbfft {

class complex_mul {
  public:
    complex_mul(precision_helper fph) : fph_(std::move(fph)) {}

    clir::expr operator()(clir::expr c, std::complex<double> const &w);
    clir::expr operator()(clir::expr c, clir::expr w);

    clir::expr pair_real(clir::expr x1, clir::expr x2, std::complex<double> const &w);
    clir::expr pair_imag(clir::expr x1, clir::expr x2, std::complex<double> const &w);

    inline auto fph() const { return fph_; }

  private:
    precision_helper fph_;
};

class basic_esum {
  public:
    basic_esum(precision_helper fph, int direction, int Nf, std::function<clir::expr(int)> x)
        : cmul_(std::move(fph)), direction_(direction), Nf_(Nf), x_(std::move(x)) {}
    clir::expr operator()(clir::block_builder &, int kf);

  protected:
    complex_mul cmul_;
    int direction_;
    int Nf_;
    std::function<clir::expr(int)> x_;
};

class pair_optimization_esum : public basic_esum {
  public:
    using basic_esum::basic_esum;
    clir::expr operator()(clir::block_builder &bb, int kf);

  private:
    struct product {
        std::pair<int, int> w_arg;
        int jf;

        bool operator<(product const &other) const {
            return w_arg < other.w_arg || (w_arg == other.w_arg && jf < other.jf);
        }
    };
    std::map<std::pair<product, product>, std::pair<clir::var, clir::var>> available_pairs_;
};

namespace generate_fft {

void basic(clir::block_builder &bb, precision fp, int direction, std::vector<int> factorization,
           clir::var &x, clir::var &y, clir::expr is_odd, clir::expr twiddle = nullptr);

void basic_inplace(clir::block_builder &bb, precision fp, int direction,
                   std::vector<int> factorization, clir::var x, clir::expr twiddle = nullptr);

void pair_optimization_inplace(clir::block_builder &bb, precision fp, int direction,
                               std::vector<int> factorization, clir::var x,
                               clir::expr twiddle = nullptr);

void basic_inplace_subgroup(clir::block_builder &bb, precision fp, int direction,
                            std::vector<int> factorization, clir::var x, clir::expr is_odd,
                            clir::expr twiddle = nullptr);

} // namespace generate_fft
} // namespace bbfft

#endif // MIXED_RADIX_FFT_20220428_HPP
