// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UNIQUE_NAMES_20220405_HPP
#define UNIQUE_NAMES_20220405_HPP

#include "clir/export.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/internal/program_node.hpp"
#include "clir/internal/stmt_node.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clir {

namespace internal {
class CLIR_EXPORT expr_node;
}

class CLIR_EXPORT func;
class CLIR_EXPORT stmt;
class CLIR_EXPORT prog;

class CLIR_EXPORT unique_names {
  public:
    /* Expr nodes */
    void operator()(internal::expr_node &e);

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

  private:
    CLIR_NO_EXPORT void make_unique_name(var e);
    CLIR_NO_EXPORT void push_scope();
    CLIR_NO_EXPORT void pop_scope();

    std::vector<std::unordered_set<uintptr_t>> declared_vars_;
    std::vector<std::unordered_map<std::string, unsigned long>> name_counters_;
};

CLIR_EXPORT void make_names_unique(prog p);
CLIR_EXPORT void make_names_unique(func k);
CLIR_EXPORT void make_names_unique(stmt s);

} // namespace clir

#endif // UNIQUE_NAMES_20220405_HPP
