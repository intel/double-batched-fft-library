// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MIXED_RADIX_FFT_20220428_HPP
#define MIXED_RADIX_FFT_20220428_HPP

#include "bbfft/configuration.hpp"
#include "clir/builder.hpp"
#include "clir/expr.hpp"
#include "clir/var.hpp"

#include <cstddef>
#include <vector>

namespace bbfft {

namespace generate_fft {
void with_cse(clir::block_builder &bb, precision fp, int direction, std::vector<int> factorization,
              clir::var &x, clir::var &y, clir::expr is_odd);

void basic(clir::block_builder &bb, precision fp, int direction, std::vector<int> factorization,
           clir::var &x, clir::var &y, clir::expr is_odd, clir::expr twiddle = nullptr);

void basic_inplace(clir::block_builder &bb, precision fp, int direction,
                   std::vector<int> factorization, clir::var x, clir::expr twiddle = nullptr);

void basic_inplace_subgroup(clir::block_builder &bb, precision fp, int direction,
                            std::vector<int> factorization, clir::var x, clir::expr is_odd,
                            clir::expr twiddle = nullptr);
} // namespace generate_fft

} // namespace bbfft

#endif // MIXED_RADIX_FFT_20220428_HPP
