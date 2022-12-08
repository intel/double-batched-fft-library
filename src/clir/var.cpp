// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/var.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/internal/stmt_node.hpp"

#include <memory>
#include <utility>

namespace clir {

var::var(std::string prefix) : e_(std::make_shared<internal::variable>(std::move(prefix))) {}

} // namespace clir
