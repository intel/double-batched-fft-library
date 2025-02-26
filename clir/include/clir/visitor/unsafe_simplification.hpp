// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UNSAFE_SIMPLIFICATION_20220408_HPP
#define UNSAFE_SIMPLIFICATION_20220408_HPP

#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/func.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/internal/program_node.hpp"
#include "clir/internal/stmt_node.hpp"
#include "clir/stmt.hpp"

namespace clir {

class CLIR_EXPORT prog;
class CLIR_EXPORT func;
class CLIR_EXPORT stmt;

class CLIR_EXPORT unsafe_simplification {
  public:
    /* Expr nodes */
    expr operator()(internal::variable &v);
    expr operator()(internal::int_imm &v);
    expr operator()(internal::uint_imm &v);
    expr operator()(internal::float_imm &v);
    expr operator()(internal::cl_mem_fence_flags_imm &v);
    expr operator()(internal::memory_scope_imm &v);
    expr operator()(internal::memory_order_imm &v);
    expr operator()(internal::string_imm &v);
    expr operator()(internal::unary_op &e);
    expr operator()(internal::binary_op &e);
    expr operator()(internal::ternary_op &e);
    expr operator()(internal::access &e);
    expr operator()(internal::call_builtin &fn);
    expr operator()(internal::call &fn);
    expr operator()(internal::cast &c);
    expr operator()(internal::swizzle &s);

    /* Stmt nodes */
    void operator()(internal::declaration &d);
    void operator()(internal::declaration_assignment &d);
    void operator()(internal::expression_statement &e);
    void operator()(internal::block &b);
    void operator()(internal::for_loop &loop);
    void operator()(internal::if_selection &is);
    void operator()(internal::while_loop &loop);

    /* Kernel nodes */
    void operator()(internal::prototype &proto);
    void operator()(internal::function &fn);
    void operator()(internal::global_declaration &d);

    /* Program nodes */
    void operator()(internal::program &prg);
};

CLIR_EXPORT expr unsafe_simplify(expr e);
CLIR_EXPORT void unsafe_simplify(stmt s);
CLIR_EXPORT void unsafe_simplify(func k);
CLIR_EXPORT void unsafe_simplify(prog p);

} // namespace clir

#endif // UNSAFE_SIMPLIFICATION_20220408_HPP
