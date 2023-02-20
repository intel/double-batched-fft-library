// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef STMT_20220405_HPP
#define STMT_20220405_HPP

#include "clir/export.hpp"
#include "clir/handle.hpp"

#include <memory>

namespace clir {

namespace internal {
class stmt_node;
}

class expr;

class CLIR_EXPORT stmt : public handle<internal::stmt_node> {
  public:
    using handle<internal::stmt_node>::handle;
};

CLIR_EXPORT stmt expression_statement(expr e);

} // namespace clir

#endif // STMT_20220405_HPP
