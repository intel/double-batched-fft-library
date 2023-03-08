// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FUNC_20220405_HPP
#define FUNC_20220405_HPP

#include "clir/export.hpp"
#include "clir/handle.hpp"

namespace clir {

namespace internal {
class function_node;
}

class CLIR_EXPORT func : public handle<internal::function_node> {
  public:
    using handle<internal::function_node>::handle;
};

} // namespace clir

#endif // FUNC_20220405_HPP
