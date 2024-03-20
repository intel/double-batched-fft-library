// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builtin_function.hpp"
#include "clir/expr.hpp"

#include <ostream>
#include <utility>

#define DEFINE_BUILTIN_FUNCTION_0_0(NAME)                                                          \
    expr NAME() { return call_builtin(builtin_function::NAME, {}); }

#define DEFINE_BUILTIN_FUNCTION_1_1(NAME)                                                          \
    expr NAME(expr e1) { return call_builtin(builtin_function::NAME, {std::move(e1)}); }

#define DEFINE_BUILTIN_FUNCTION_2_2(NAME)                                                          \
    expr NAME(expr e1, expr e2) {                                                                  \
        return call_builtin(builtin_function::NAME, {std::move(e1), std::move(e2)});               \
    }

#define DEFINE_BUILTIN_FUNCTION_3_3(NAME)                                                          \
    expr NAME(expr e1, expr e2, expr e3) {                                                         \
        return call_builtin(builtin_function::NAME,                                                \
                            {std::move(e1), std::move(e2), std::move(e3)});                        \
    }

#define DEFINE_BUILTIN_FUNCTION_4_4(NAME)                                                          \
    expr NAME(expr e1, expr e2, expr e3, expr e4) {                                                \
        return call_builtin(builtin_function::NAME,                                                \
                            {std::move(e1), std::move(e2), std::move(e3), std::move(e4)});         \
    }

#define DEFINE_BUILTIN_FUNCTION_5_5(NAME)                                                          \
    expr NAME(expr e1, expr e2, expr e3, expr e4, expr e5) {                                       \
        return call_builtin(builtin_function::NAME, {std::move(e1), std::move(e2), std::move(e3),  \
                                                     std::move(e4), std::move(e5)});               \
    }

#define DEFINE_BUILTIN_FUNCTION_6_6(NAME)                                                          \
    expr NAME(expr e1, expr e2, expr e3, expr e4, expr e5, expr e6) {                              \
        return call_builtin(builtin_function::NAME,                                                \
                            {std::move(e1), std::move(e2), std::move(e3), std::move(e4),           \
                             std::move(e5), std::move(e6)});                                       \
    }

#define DEFINE_BUILTIN_FUNCTION_0_inf(NAME)                                                        \
    expr NAME(std::vector<expr> args) {                                                            \
        return call_builtin(builtin_function::NAME, std::move(args));                              \
    }
#define DEFINE_BUILTIN_FUNCTION_1_2(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_1_1(NAME) DEFINE_BUILTIN_FUNCTION_2_2(NAME)
#define DEFINE_BUILTIN_FUNCTION_1_3(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_1_1(NAME)                                                              \
    DEFINE_BUILTIN_FUNCTION_2_2(NAME) DEFINE_BUILTIN_FUNCTION_3_3(NAME)
#define DEFINE_BUILTIN_FUNCTION_2_3(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_2_2(NAME) DEFINE_BUILTIN_FUNCTION_3_3(NAME)
#define DEFINE_BUILTIN_FUNCTION_3_4(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_3_3(NAME) DEFINE_BUILTIN_FUNCTION_4_4(NAME)
#define DEFINE_BUILTIN_FUNCTION_5_6(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_5_5(NAME) DEFINE_BUILTIN_FUNCTION_6_6(NAME)
#define DEFINE_BUILTIN_FUNCTION(NAME, A, B) DEFINE_BUILTIN_FUNCTION_##A##_##B(NAME)

#define BUILTIN_FN_CASE_3(x, y, z) case builtin_function::x:

namespace clir {

extension get_extension(builtin_function fn) {
    switch (fn) {
        CLIR_STANDARD_BUILTIN_FUNCTION(BUILTIN_FN_CASE_3)
        return extension::builtin;
        CLIR_EXTENSION_INTEL_SUBGROUPS(BUILTIN_FN_CASE_3)
        return extension::cl_intel_subgroups;
        CLIR_EXTENSION_INTEL_SUBGROUPS_LONG(BUILTIN_FN_CASE_3)
        return extension::cl_intel_subgroups_long;
        CLIR_EXTENSION_INTEL_SUBGROUPS_SHORT(BUILTIN_FN_CASE_3)
        return extension::cl_intel_subgroups_short;
    default:
        break;
    }
    return extension::unknown;
}

static const char *builtin_function_strings[] = {CLIR_BUILTIN_FUNCTION(CLIR_STRING_LIST_3)};

char const *to_string(builtin_function fn) {
    return builtin_function_strings[static_cast<int>(fn)];
}
char const *to_string(extension ext) {
    switch (ext) {
    case extension::cl_intel_subgroups:
        return "cl_intel_subgroups";
    case extension::cl_intel_subgroups_long:
        return "cl_intel_subgroups_long";
    case extension::cl_intel_subgroups_short:
        return "cl_intel_subgroups_short";
    case extension::cl_ext_float_atomics:
        return "cl_ext_float_atomics";
    case extension::builtin:
        return "builtin";
    case extension::unknown:
        break;
    };
    return "unknown";
}

CLIR_BUILTIN_FUNCTION(DEFINE_BUILTIN_FUNCTION)

} // namespace clir

namespace std {
std::ostream &operator<<(std::ostream &os, clir::builtin_function fn) {
    return os << clir::to_string(fn);
}
std::ostream &operator<<(std::ostream &os, clir::extension ext) {
    return os << clir::to_string(ext);
}
} // namespace std
