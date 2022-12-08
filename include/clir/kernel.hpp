// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef KERNEL_20220405_HPP
#define KERNEL_20220405_HPP

#include "clir/export.hpp"
#include "clir/handle.hpp"

namespace clir {

namespace internal {
class kernel_node;
}

class CLIR_EXPORT kernel : public handle<internal::kernel_node> {
  public:
    using handle<internal::kernel_node>::handle;
};

} // namespace clir

#endif // KERNEL_20220405_HPP
