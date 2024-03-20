// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EQUAL_EXPR_20220408_HPP
#define EQUAL_EXPR_20220408_HPP

#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/internal/data_type_node.hpp"
#include "clir/internal/expr_node.hpp"

namespace clir {

class CLIR_EXPORT equal_expr {
  public:
    /* Data type nodes */
    bool operator()(internal::data_type_node &a, internal::data_type_node &b);
    bool operator()(internal::scalar_data_type &a, internal::scalar_data_type &b);
    bool operator()(internal::vector_data_type &a, internal::vector_data_type &b);
    bool operator()(internal::pointer &a, internal::pointer &b);
    bool operator()(internal::array &a, internal::array &b);

    /* Expr nodes */
    bool operator()(internal::expr_node &a, internal::expr_node &b);
    bool operator()(internal::variable &a, internal::variable &b);
    bool operator()(internal::int_imm &a, internal::int_imm &b);
    bool operator()(internal::uint_imm &a, internal::uint_imm &b);
    bool operator()(internal::float_imm &a, internal::float_imm &b);
    bool operator()(internal::cl_mem_fence_flags_imm &a, internal::cl_mem_fence_flags_imm &b);
    bool operator()(internal::memory_scope_imm &a, internal::memory_scope_imm &b);
    bool operator()(internal::memory_order_imm &a, internal::memory_order_imm &b);
    bool operator()(internal::unary_op &a, internal::unary_op &b);
    bool operator()(internal::binary_op &a, internal::binary_op &b);
    bool operator()(internal::ternary_op &a, internal::ternary_op &b);
    bool operator()(internal::access &a, internal::access &b);
    bool operator()(internal::call_builtin &a, internal::call_builtin &b);
    bool operator()(internal::cast &a, internal::cast &b);
};

CLIR_EXPORT bool is_equivalent(data_type a, data_type b);
CLIR_EXPORT bool is_equivalent(expr a, expr b);

} // namespace clir

#endif // EQUAL_EXPR_20220408_HPP
