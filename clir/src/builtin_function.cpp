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

#define DEFINE_BUILTIN_FUNCTION_0_inf(NAME)                                                        \
    expr NAME(std::vector<expr> args) {                                                            \
        return call_builtin(builtin_function::NAME, std::move(args));                              \
    }
#define DEFINE_BUILTIN_FUNCTION_1_2(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_1_1(NAME) DEFINE_BUILTIN_FUNCTION_2_2(NAME)
#define DEFINE_BUILTIN_FUNCTION_2_3(NAME)                                                          \
    DEFINE_BUILTIN_FUNCTION_2_2(NAME) DEFINE_BUILTIN_FUNCTION_3_3(NAME)
#define DEFINE_BUILTIN_FUNCTION(NAME, A, B) DEFINE_BUILTIN_FUNCTION_##A##_##B(NAME)

namespace clir {

static const char *builtin_function_strings[] = {CLIR_BUILTIN_FUNCTION(CLIR_STRING_LIST_3)};

char const *to_string(builtin_function fn) {
    return builtin_function_strings[static_cast<int>(fn)];
}
std::ostream &operator<<(std::ostream &os, builtin_function fn) { return os << to_string(fn); }

CLIR_BUILTIN_FUNCTION(DEFINE_BUILTIN_FUNCTION)

} // namespace clir
