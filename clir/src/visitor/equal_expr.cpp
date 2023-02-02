// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/equal_expr.hpp"
#include "clir/visit.hpp"

#include <algorithm>

namespace clir {

/* Data type nodes */
bool equal_expr::operator()(internal::scalar_data_type &a, internal::scalar_data_type &b) {
    return a.type() == b.type() && a.space() == b.space();
}
bool equal_expr::operator()(internal::vector_data_type &a, internal::vector_data_type &b) {
    return a.type() == b.type() && a.size() == b.size() && a.space() == b.space();
}
bool equal_expr::operator()(internal::pointer &a, internal::pointer &b) {
    return visit(*this, *a.ty(), *b.ty());
}
bool equal_expr::operator()(internal::array &a, internal::array &b) {
    return a.size() == b.size() && visit(*this, *a.ty(), *b.ty());
}

/* Expr nodes */
bool equal_expr::operator()(internal::expr_node &, internal::expr_node &) { return false; }
bool equal_expr::operator()(internal::variable &a, internal::variable &b) { return &a == &b; }
bool equal_expr::operator()(internal::int_imm &a, internal::int_imm &b) {
    return a.value() == b.value() && a.bits() == b.bits();
}
bool equal_expr::operator()(internal::uint_imm &a, internal::uint_imm &b) {
    return a.value() == b.value() && a.bits() == b.bits();
}
bool equal_expr::operator()(internal::float_imm &a, internal::float_imm &b) {
    return a.value() == b.value() && a.bits() == b.bits();
}
bool equal_expr::operator()(internal::cl_mem_fence_flags_imm &a,
                            internal::cl_mem_fence_flags_imm &b) {
    return a.value() == b.value();
}
bool equal_expr::operator()(internal::unary_op &a, internal::unary_op &b) {
    return a.op() == b.op() && visit(*this, *a.term(), *b.term());
}
bool equal_expr::operator()(internal::binary_op &a, internal::binary_op &b) {
    return a.op() == b.op() && visit(*this, *a.lhs(), *b.lhs()) && visit(*this, *a.rhs(), *b.rhs());
}
bool equal_expr::operator()(internal::access &a, internal::access &b) {
    return visit(*this, *a.field(), *b.field()) && visit(*this, *a.address(), *b.address());
}
bool equal_expr::operator()(internal::call_builtin &a, internal::call_builtin &b) {
    auto &aa = a.args();
    auto &ba = b.args();
    if (a.fn() != b.fn() || aa.size() != ba.size()) {
        return false;
    }
    bool result = true;
    for (auto ai = aa.begin(), bi = ba.begin(); ai != aa.end() && bi != ba.end(); ++ai, ++bi) {
        result = result && visit(*this, **ai, **bi);
    }
    return result;
}
bool equal_expr::operator()(internal::cast &a, internal::cast &b) {
    return visit(*this, *a.target_ty(), *b.target_ty()) && visit(*this, *a.term(), *b.term());
}

bool is_equivalent(data_type a, data_type b) { return visit(equal_expr{}, *a, *b); }
bool is_equivalent(expr a, expr b) { return visit(equal_expr{}, *a, *b); }

} // namespace clir
