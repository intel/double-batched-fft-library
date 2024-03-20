// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builtin_type.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <type_traits>

namespace clir {

static const char *builtin_type_strings[] = {"bool",
                                             "char",
                                             "uchar",
                                             "short",
                                             "ushort",
                                             "int",
                                             "uint",
                                             "long",
                                             "ulong",
                                             "float",
                                             "double",
                                             "half",
                                             "size_t",
                                             "ptrdiff_t",
                                             "intptr_t",
                                             "uintptr_t",
                                             "void",
                                             "cl_mem_fence_flags",
                                             "memory_scope",
                                             "memory_order",
                                             "atomic_flag",
                                             "atomic_int",
                                             "atomic_uint",
                                             "atomic_long",
                                             "atomic_ulong",
                                             "atomic_float",
                                             "atomic_double",
                                             "atomic_intptr_t",
                                             "atomic_uintptr_t",
                                             "atomic_size_t",
                                             "atomic_ptrdiff_t",
                                             "atomic_half"};
static const char *cl_mem_fence_flags_strings[] = {"CLK_GLOBAL_MEM_FENCE", "CLK_LOCAL_MEM_FENCE",
                                                   "CLK_IMAGE_MEM_FENCE"};
static const char *memory_scope_strings[] = {
    "memory_scope_work_item", "memory_scope_sub_group",       "memory_scope_work_group",
    "memory_scope_device",    "memory_scope_all_svm_devices", "memory_scope_all_devices"};
static const char *memory_order_strings[] = {"memory_order_relaxed", "memory_order_acquire",
                                             "memory_order_release", "memory_order_acq_rel",
                                             "memory_order_seq_cst"};
static const char *address_space_strings[] = {"", "global", "local", "constant", "private"};
static const char *endianess_strings[] = {"device", "host"};
static auto function_qualifier_strings = std::array<char const *, 3u>{"extern", "inline", "kernel"};
static auto type_qualifier_strings = std::array<char const *, 3u>{"const", "restrict", "volatile"};
template <typename T>
void to_string_helper(std::ostream &os, T q, char sep,
                      std::array<char const *, 3u> const &strings) {
    using ut = std::underlying_type_t<T>;
    const std::size_t num_qualifiers = strings.size();
    for (std::size_t i = 0; i < num_qualifiers; ++i) {
        ut i2 = 1 << i;
        if (ut(q) & i2) {
            os << strings[i];
            ut i3 = (~0u) << (i + 1);
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
char const *to_string(memory_scope f) { return memory_scope_strings[static_cast<int>(f)]; }
char const *to_string(memory_order f) { return memory_order_strings[static_cast<int>(f)]; }
char const *to_string(address_space as) { return address_space_strings[static_cast<int>(as)]; }
char const *to_string(endianess e) { return endianess_strings[static_cast<int>(e)]; }
std::string to_string(function_qualifier q, char sep) {
    std::ostringstream oss;
    to_string_helper(oss, q, sep, function_qualifier_strings);
    return oss.str();
}
std::string to_string(type_qualifier q, char sep) {
    std::ostringstream oss;
    to_string_helper(oss, q, sep, type_qualifier_strings);
    return oss.str();
}
std::ostream &operator<<(std::ostream &os, builtin_type basic_data_type) {
    return os << to_string(basic_data_type);
}
std::ostream &operator<<(std::ostream &os, cl_mem_fence_flags f) { return os << to_string(f); }
std::ostream &operator<<(std::ostream &os, memory_scope f) { return os << to_string(f); }
std::ostream &operator<<(std::ostream &os, memory_order f) { return os << to_string(f); }
std::ostream &operator<<(std::ostream &os, address_space as) { return os << to_string(as); }
std::ostream &operator<<(std::ostream &os, endianess e) { return os << to_string(e); }
std::ostream &operator<<(std::ostream &os, function_qualifier q) {
    to_string_helper(os, q, ' ', function_qualifier_strings);
    return os;
}
std::ostream &operator<<(std::ostream &os, type_qualifier q) {
    to_string_helper(os, q, ' ', type_qualifier_strings);
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

type_qualifier operator|(type_qualifier x, type_qualifier y) {
    using ut = std::underlying_type_t<type_qualifier>;
    return type_qualifier(ut(x) | ut(y));
}
type_qualifier operator&(type_qualifier x, type_qualifier y) {
    using ut = std::underlying_type_t<type_qualifier>;
    return type_qualifier(ut(x) & ut(y));
}
bool test(type_qualifier x) { return std::underlying_type_t<type_qualifier>(x); }

} // namespace clir
