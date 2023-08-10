// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SNIPPET_20230803_HPP
#define SNIPPET_20230803_HPP

#include "tensor_view.hpp"
#include "utility.hpp"

#include "clir/builder.hpp"
#include "clir/expr.hpp"

#include <cstdint>
#include <functional>

namespace bbfft {

using permutation_fun = std::function<std::size_t(std::size_t)>;
inline std::size_t identity(std::size_t i) { return i; }

void copy_mbNkb_block_on_2D_grid(clir::block_builder &bb, tensor_view<3u> const &X_src,
                                 tensor_view<3u> const &X_dest, clir::expr mb, std::size_t N,
                                 clir::expr kb);

void copy_N_block_with_permutation(clir::block_builder &bb, tensor_view<1u> const &X_src,
                                   tensor_view<1u> const &X_dest, std::size_t N,
                                   permutation_fun src_perm = identity,
                                   permutation_fun dest_perm = identity);

void set_k_maybe_not_written_to_zero(clir::block_builder &bb, precision_helper fph,
                                     tensor_view<3u> const &X1, std::size_t N, clir::expr K,
                                     std::size_t Kb);

} // namespace bbfft

#endif // SNIPPET_20230803_HPP
