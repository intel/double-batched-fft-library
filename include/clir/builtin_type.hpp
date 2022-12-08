// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BUILTIN_TYPE_20220405_HPP
#define BUILTIN_TYPE_20220405_HPP

#include "clir/export.hpp"

#include <iosfwd>

namespace clir {

enum class builtin_type {
    bool_t,
    /* begin vector types */
    char_t,
    uchar_t,
    short_t,
    ushort_t,
    int_t,
    uint_t,
    long_t,
    ulong_t,
    float_t,
    double_t,
    /* end vector types */
    half_t,
    size_t,
    ptrdiff_t,
    intptr_t,
    uintptr_t,
    void_t,
    cl_mem_fence_flags_t
};

enum class cl_mem_fence_flags { CLK_GLOBAL_MEM_FENCE, CLK_LOCAL_MEM_FENCE, CLK_IMAGE_MEM_FENCE };
enum class address_space { generic_t, global_t, local_t, constant_t, private_t };
enum class endianess { device, host };

CLIR_EXPORT char const *to_string(builtin_type basic_data_type);
CLIR_EXPORT char const *to_string(cl_mem_fence_flags fence_flag);
CLIR_EXPORT char const *to_string(address_space as);
CLIR_EXPORT char const *to_string(endianess e);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, builtin_type sdt);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, cl_mem_fence_flags fence_flag);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, address_space as);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, endianess e);

} // namespace clir

#endif // BUILTIN_TYPE_20220405_HPP
