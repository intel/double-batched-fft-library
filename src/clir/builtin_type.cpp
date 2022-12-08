// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builtin_type.hpp"

#include <ostream>

namespace clir {

static const char *builtin_type_strings[] = {
    "bool",   "char",      "uchar",    "short",     "ushort", "int",
    "uint",   "long",      "ulong",    "float",     "double", "half",
    "size_t", "ptrdiff_t", "intptr_t", "uintptr_t", "void",   "cl_mem_fence_flags"};
static const char *cl_mem_fence_flags_strings[] = {"CLK_GLOBAL_MEM_FENCE", "CLK_LOCAL_MEM_FENCE",
                                                   "CLK_IMAGE_MEM_FENCE"};
static const char *address_space_strings[] = {"", "global", "local", "constant", "private"};
static const char *endianess_strings[] = {"device", "host"};

char const *to_string(builtin_type basic_data_type) {
    return builtin_type_strings[static_cast<int>(basic_data_type)];
}
char const *to_string(cl_mem_fence_flags f) {
    return cl_mem_fence_flags_strings[static_cast<int>(f)];
}
char const *to_string(address_space as) { return address_space_strings[static_cast<int>(as)]; }
char const *to_string(endianess e) { return endianess_strings[static_cast<int>(e)]; }
std::ostream &operator<<(std::ostream &os, builtin_type basic_data_type) {
    return os << to_string(basic_data_type);
}
std::ostream &operator<<(std::ostream &os, cl_mem_fence_flags f) { return os << to_string(f); }
std::ostream &operator<<(std::ostream &os, address_space as) { return os << to_string(as); }
std::ostream &operator<<(std::ostream &os, endianess e) { return os << to_string(e); }

} // namespace clir
