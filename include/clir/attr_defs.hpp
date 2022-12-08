// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ATTR_DEFS_20220405_HPP
#define ATTR_DEFS_20220405_HPP

#include "clir/attr.hpp"
#include "clir/internal/attr_node.hpp"

#include <utility>

#define CLIR_MAKE_ATTRIBUTE_FUNCTION(NAME)                                                         \
    template <typename... Ts> auto NAME(Ts &&...t) {                                               \
        return attr(                                                                               \
            std::make_shared<internal::attribute<internal::NAME>>(std::forward<Ts>(t)...));        \
    }

namespace clir {

CLIR_MAKE_ATTRIBUTE_FUNCTION(work_group_size_hint)
CLIR_MAKE_ATTRIBUTE_FUNCTION(reqd_work_group_size)
CLIR_MAKE_ATTRIBUTE_FUNCTION(intel_reqd_sub_group_size)
CLIR_MAKE_ATTRIBUTE_FUNCTION(opencl_unroll_hint)
CLIR_MAKE_ATTRIBUTE_FUNCTION(aligned)
CLIR_MAKE_ATTRIBUTE_FUNCTION(packed)
CLIR_MAKE_ATTRIBUTE_FUNCTION(endian)

} // namespace clir

#endif // ATTR_DEFS_20220405_HPP
