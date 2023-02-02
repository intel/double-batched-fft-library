// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/stmt.hpp"
#include "clir/expr.hpp"
#include "clir/internal/stmt_node.hpp"

#include <utility>

namespace clir {

stmt expression_statement(expr e) {
    return stmt(std::make_shared<internal::expression_statement>(std::move(e)));
}

} // namespace clir
