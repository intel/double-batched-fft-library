// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef OP_20220405_HPP
#define OP_20220405_HPP

#include "clir/export.hpp"

#include <array>
#include <iosfwd>

namespace clir {

enum class ternary_operation { conditional };

enum class binary_operation {
    add,
    subtract,
    multiply,
    divide,
    modulo,
    greater_than,
    less_than,
    greater_than_or_equal,
    less_than_or_equal,
    equal,
    not_equal,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    logical_and,
    logical_or,
    left_shift,
    right_shift,
    assignment,
    add_into,
    subtract_from,
    multiply_into,
    divide_into,
    modulus_into,
    left_shift_by,
    right_shift_by,
    and_into,
    or_into,
    xor_into,
    comma
};

enum class unary_operation {
    minus,
    bitwise_not,
    logical_not,
    indirection,
    address,
    pre_increment,
    pre_decrement,
    post_increment,
    post_decrement
};

enum class associativity { left_to_right, right_to_left, none };

CLIR_EXPORT char const *to_string(ternary_operation op, short component);
CLIR_EXPORT char const *to_string(binary_operation op);
CLIR_EXPORT char const *to_string(unary_operation op);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, binary_operation op);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, unary_operation op);

// https://en.cppreference.com/w/c/language/operator_precedence
CLIR_EXPORT unsigned operation_precedence(ternary_operation op);
CLIR_EXPORT unsigned operation_precedence(binary_operation op);
CLIR_EXPORT unsigned operation_precedence(unary_operation op);
CLIR_EXPORT associativity operation_associativity(ternary_operation op);
CLIR_EXPORT associativity operation_associativity(binary_operation op);
CLIR_EXPORT associativity operation_associativity(unary_operation op);

} // namespace clir

#endif // OP_20220405_HPP
