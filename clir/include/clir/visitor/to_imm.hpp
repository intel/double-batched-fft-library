// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TO_IMM_20230807_HPP
#define TO_IMM_20230807_HPP

#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/internal/expr_node.hpp"

#include <cstdint>
#include <variant>

namespace clir {

class CLIR_EXPORT to_imm {
  public:
    using return_t = std::variant<std::monostate, int64_t, uint64_t, double>;

    /* Expr nodes */
    auto operator()(internal::expr_node &) -> return_t;
    auto operator()(internal::int_imm &i) -> return_t;
    auto operator()(internal::uint_imm &i) -> return_t;
    auto operator()(internal::float_imm &i) -> return_t;
};

CLIR_EXPORT auto get_imm(expr e) -> to_imm::return_t;

} // namespace clir

#endif // TO_IMM_20230807_HPP
