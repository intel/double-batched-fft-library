// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ATTR_20220405_HPP
#define ATTR_20220405_HPP

#include "clir/export.hpp"
#include "clir/handle.hpp"

namespace clir {

namespace internal {
class CLIR_EXPORT attr_node;
} // namespace internal

class CLIR_EXPORT attr : public handle<internal::attr_node> {
  public:
    using handle<internal::attr_node>::handle;
};

} // namespace clir

#endif // ATTR_20220405_HPP
