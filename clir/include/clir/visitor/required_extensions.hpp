// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REQUIRED_EXTENSIONS_20240123_HPP
#define REQUIRED_EXTENSIONS_20240123_HPP

#include "clir/builtin_function.hpp"
#include "clir/export.hpp"
#include "clir/internal/data_type_node.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/internal/program_node.hpp"
#include "clir/internal/stmt_node.hpp"

#include <array>
#include <vector>

namespace clir {

class CLIR_EXPORT func;
class CLIR_EXPORT prog;

class CLIR_EXPORT required_extensions {
  public:
    /* Expr nodes */
    void operator()(internal::expr_node &);
    void operator()(internal::unary_op &op);
    void operator()(internal::binary_op &op);
    void operator()(internal::ternary_op &op);
    void operator()(internal::access &op);
    void operator()(internal::call_builtin &fn);
    void operator()(internal::call &fn);
    void operator()(internal::cast &op);
    void operator()(internal::swizzle &op);

    /* Stmt nodes */
    void operator()(internal::stmt_node &);
    void operator()(internal::declaration_assignment &d);
    void operator()(internal::expression_statement &e);
    void operator()(internal::block &b);
    void operator()(internal::for_loop &loop);
    void operator()(internal::if_selection &is);

    /* Kernel nodes */
    void operator()(internal::prototype &);
    void operator()(internal::function &fn);
    void operator()(internal::global_declaration &d);

    /* Program nodes */
    void operator()(internal::program &prg);

    auto extensions() const -> std::vector<extension>;

  private:
    std::array<bool, static_cast<int>(extension::unknown)> needs_ext_ = {};
};

CLIR_EXPORT auto get_required_extensions(func k) -> std::vector<extension>;
CLIR_EXPORT auto get_required_extensions(prog p) -> std::vector<extension>;

} // namespace clir

#endif // REQUIRED_EXTENSIONS_20240123_HPP
