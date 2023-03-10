// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/func.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/stmt.hpp"

#include <utility>

namespace clir {

func function(func prototype, stmt body) {
    return func(std::make_shared<internal::function>(std::move(prototype), std::move(body)));
}

} // namespace clir
