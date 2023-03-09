// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PROG_20230309_HPP
#define PROG_20230309_HPP

#include "clir/export.hpp"
#include "clir/handle.hpp"

namespace clir {

namespace internal {
class program_node;
}

class CLIR_EXPORT prog : public handle<internal::program_node> {
  public:
    using handle<internal::program_node>::handle;
};

} // namespace clir

#endif // PROG_20230309_HPP
