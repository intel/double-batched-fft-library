// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builtin_type.hpp"

#include <cstddef>
#include <ostream>
#include <sstream>
#include <type_traits>

namespace clir {

static const char *builtin_type_strings[] = {
    "bool",   "char",      "uchar",    "short",     "ushort", "int",
    "uint",   "long",      "ulong",    "float",     "double", "half",
    "size_t", "ptrdiff_t", "intptr_t", "uintptr_t", "void",   "cl_mem_fence_flags"};
static const char *cl_mem_fence_flags_strings[] = {"CLK_GLOBAL_MEM_FENCE", "CLK_LOCAL_MEM_FENCE",
                                                   "CLK_IMAGE_MEM_FENCE"};
static const char *address_space_strings[] = {"", "global", "local", "constant", "private"};
static const char *endianess_strings[] = {"device", "host"};
static const char *function_qualifier_strings[] = {"extern", "inline", "kernel"};
void to_string_helper(std::ostream &os, function_qualifier q, char sep) {
    using ut = std::underlying_type_t<function_qualifier>;
    constexpr std::size_t num_qualifiers =
        sizeof(function_qualifier_strings) / sizeof(const char *);
    for (std::size_t i = 0; i < num_qualifiers; ++i) {
        ut i2 = 1 << i;
        if (ut(q) & i2) {
            os << function_qualifier_strings[i];
            ut i3 = (~0) << (i + 1);
            if (ut(q) & i3) {
                os << sep;
            }
        }
    }
}

char const *to_string(builtin_type basic_data_type) {
    return builtin_type_strings[static_cast<int>(basic_data_type)];
}
char const *to_string(cl_mem_fence_flags f) {
    return cl_mem_fence_flags_strings[static_cast<int>(f)];
}
char const *to_string(address_space as) { return address_space_strings[static_cast<int>(as)]; }
char const *to_string(endianess e) { return endianess_strings[static_cast<int>(e)]; }
std::string to_string(function_qualifier q, char sep) {
    std::ostringstream oss;
    to_string_helper(oss, q, sep);
    return oss.str();
}
std::ostream &operator<<(std::ostream &os, builtin_type basic_data_type) {
    return os << to_string(basic_data_type);
}
std::ostream &operator<<(std::ostream &os, cl_mem_fence_flags f) { return os << to_string(f); }
std::ostream &operator<<(std::ostream &os, address_space as) { return os << to_string(as); }
std::ostream &operator<<(std::ostream &os, endianess e) { return os << to_string(e); }
std::ostream &operator<<(std::ostream &os, function_qualifier q) {
    to_string_helper(os, q, ' ');
    return os;
}

function_qualifier operator|(function_qualifier x, function_qualifier y) {
    using ut = std::underlying_type_t<function_qualifier>;
    return function_qualifier(ut(x) | ut(y));
}
function_qualifier operator&(function_qualifier x, function_qualifier y) {
    using ut = std::underlying_type_t<function_qualifier>;
    return function_qualifier(ut(x) & ut(y));
}
bool test(function_qualifier x) { return std::underlying_type_t<function_qualifier>(x); }

} // namespace clir
