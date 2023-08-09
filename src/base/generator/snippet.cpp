// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "snippet.hpp"

#include "clir/builtin_function.hpp"
#include "clir/data_type.hpp"
#include "clir/visitor/to_imm.hpp"

#include <cstdint>
#include <variant>

using namespace clir;

namespace bbfft {

void copy_mbNkb_block_on_2D_grid(block_builder &bb, tensor_view<3u> const &X_src,
                                 tensor_view<3u> const &X_dest, expr mb, std::size_t N, expr kb) {
    auto const make_copy = [&](block_builder &bb, expr mb) {
        auto m_local = bb.declare_assign(generic_uint(), "m_local", get_local_id(0));
        auto k_local = bb.declare_assign(generic_uint(), "k_local", get_local_id(1));
        auto range_check =
            if_selection_builder(m_local < mb && k_local < kb).then([&](block_builder &bb) {
                auto base_idx =
                    bb.declare_assign(generic_uint(), "base_idx", m_local + k_local * mb);
                auto m_in = bb.declare_assign(generic_uint(), "m_in", base_idx % mb);

                for (std::size_t n_local = 0; n_local < N; ++n_local) {
                    auto idx =
                        bb.declare_assign(generic_uint(), "idx", base_idx + n_local * (mb * kb));
                    auto n_in = idx / mb % N;
                    auto k_in = idx / (mb * N);
                    bb.add(X_dest.store(X_src(m_in, n_in, k_in), m_in, n_in, k_in));
                }
            });
        bb.add(range_check.get_product());
    };

    auto i = get_imm(mb);
    if (std::holds_alternative<uint64_t>(i)) {
        make_copy(bb, std::get<uint64_t>(i));
    } else if (std::holds_alternative<int64_t>(i)) {
        make_copy(bb, std::get<int64_t>(i));
    } else {
        auto check_mb = if_selection_builder(mb == get_local_size(0))
                            .then([&](block_builder &bb) { make_copy(bb, get_local_size(0)); })
                            .otherwise([&](block_builder &bb) { make_copy(bb, mb); });
        bb.add(check_mb.get_product());
    }
}

void set_k_maybe_not_written_to_zero(block_builder &bb, precision_helper fph,
                                     tensor_view<3u> const &X1, std::size_t N, expr K,
                                     std::size_t Kb) {
    auto m_local = bb.declare_assign(generic_uint(), "m_local", get_local_id(0));
    auto k_local = bb.declare_assign(generic_uint(), "k_local", get_local_id(1));
    auto i = var("i");
    auto k_maybe_not_written = bb.declare_assign(generic_ulong(), "k_maybe_not_written", K % Kb);
    bb.add(
        for_loop_builder(declaration_assignment(generic_uint(), i, k_local), i < N, add_into(i, Kb))
            .body([&](block_builder &bb) {
                bb.add(X1.store(fph.zero(), m_local, i, k_maybe_not_written));
            })
            .get_product());
}

} // namespace bbfft
