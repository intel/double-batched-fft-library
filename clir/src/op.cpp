// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/op.hpp"

#include <ostream>

namespace clir {

static const std::array<const char *, 2> ternary_operator_strings[] = {{"?", ":"}};

static const char *binary_operator_strings[] = {
    "+",  "-",  "*",  "/", "%",  ">",  "<",  ">=", "<=", "==",  "!=",  "&",  "|",  "^",  "&&",
    "||", "<<", ">>", "=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=", ","};

static const char *unary_operator_strings[] = {"-", "~", "!", "*", "&", "++", "--", "++", "--"};

char const *to_string(ternary_operation op, short component) {
    return ternary_operator_strings[static_cast<int>(op)][component];
}
char const *to_string(binary_operation op) { return binary_operator_strings[static_cast<int>(op)]; }
char const *to_string(unary_operation op) { return unary_operator_strings[static_cast<int>(op)]; }
std::ostream &operator<<(std::ostream &os, binary_operation op) { return os << to_string(op); }
std::ostream &operator<<(std::ostream &os, unary_operation op) { return os << to_string(op); }

unsigned operation_precedence(ternary_operation op) {
    switch (op) {
    case ternary_operation::conditional:
        return 13;
    }
    return 16;
}
unsigned operation_precedence(binary_operation op) {
    switch (op) {
    case binary_operation::multiply:
    case binary_operation::divide:
    case binary_operation::modulo:
        return 3;
    case binary_operation::add:
    case binary_operation::subtract:
        return 4;
    case binary_operation::left_shift:
    case binary_operation::right_shift:
        return 5;
    case binary_operation::greater_than:
    case binary_operation::less_than:
    case binary_operation::greater_than_or_equal:
    case binary_operation::less_than_or_equal:
        return 6;
    case binary_operation::equal:
    case binary_operation::not_equal:
        return 7;
    case binary_operation::bitwise_and:
        return 8;
    case binary_operation::bitwise_xor:
        return 9;
    case binary_operation::bitwise_or:
        return 10;
    case binary_operation::logical_and:
        return 11;
    case binary_operation::logical_or:
        return 12;
    case binary_operation::assignment:
    case binary_operation::add_into:
    case binary_operation::subtract_from:
    case binary_operation::multiply_into:
    case binary_operation::divide_into:
    case binary_operation::modulus_into:
    case binary_operation::left_shift_by:
    case binary_operation::right_shift_by:
    case binary_operation::and_into:
    case binary_operation::or_into:
    case binary_operation::xor_into:
        return 14;
    case binary_operation::comma:
        return 15;
    }
    return 16;
}
unsigned operation_precedence(unary_operation op) {
    switch (op) {
    case unary_operation::post_increment:
    case unary_operation::post_decrement:
        return 1;
    case unary_operation::pre_increment:
    case unary_operation::pre_decrement:
    case unary_operation::minus:
    case unary_operation::bitwise_not:
    case unary_operation::logical_not:
    case unary_operation::indirection:
    case unary_operation::address:
        return 2;
    }
    return 3;
}
associativity operation_associativity(ternary_operation op) {
    switch (op) {
    case ternary_operation::conditional:
        return associativity::right_to_left;
    }
    return associativity::right_to_left;
}
associativity operation_associativity(binary_operation op) {
    switch (op) {
    case binary_operation::multiply:
    case binary_operation::divide:
    case binary_operation::modulo:
    case binary_operation::add:
    case binary_operation::subtract:
    case binary_operation::left_shift:
    case binary_operation::right_shift:
    case binary_operation::greater_than:
    case binary_operation::less_than:
    case binary_operation::greater_than_or_equal:
    case binary_operation::less_than_or_equal:
    case binary_operation::equal:
    case binary_operation::not_equal:
    case binary_operation::bitwise_and:
    case binary_operation::bitwise_xor:
    case binary_operation::bitwise_or:
    case binary_operation::logical_and:
    case binary_operation::logical_or:
    case binary_operation::comma:
        return associativity::left_to_right;
    case binary_operation::assignment:
    case binary_operation::add_into:
    case binary_operation::subtract_from:
    case binary_operation::multiply_into:
    case binary_operation::divide_into:
    case binary_operation::modulus_into:
    case binary_operation::left_shift_by:
    case binary_operation::right_shift_by:
    case binary_operation::and_into:
    case binary_operation::or_into:
    case binary_operation::xor_into:
        return associativity::right_to_left;
    }
    return associativity::right_to_left;
}
associativity operation_associativity(unary_operation op) {
    switch (op) {
    case unary_operation::post_increment:
    case unary_operation::post_decrement:
        return associativity::left_to_right;
    case unary_operation::pre_increment:
    case unary_operation::pre_decrement:
    case unary_operation::minus:
    case unary_operation::bitwise_not:
    case unary_operation::logical_not:
    case unary_operation::indirection:
    case unary_operation::address:
        return associativity::right_to_left;
    }
    return associativity::right_to_left;
}

} // namespace clir
