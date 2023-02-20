// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EXPR_20220405_HPP
#define EXPR_20220405_HPP

#include "clir/attr.hpp"
#include "clir/builtin_function.hpp"
#include "clir/export.hpp"
#include "clir/handle.hpp"
#include "clir/op.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define CLIR_MAKE_UNARY_OPERATOR(SYM, UN_OP)                                                       \
    inline expr operator SYM(expr a) {                                                             \
        return make_unary_operation(unary_operation::UN_OP, std::move(a));                         \
    }

#define CLIR_MAKE_BINARY_OPERATOR(SYM, BIN_OP)                                                     \
    inline expr operator SYM(expr a, expr b) {                                                     \
        return make_binary_operation(binary_operation::BIN_OP, std::move(a), std::move(b));        \
    }

#define CLIR_MAKE_NAMED_BINARY_OPERATOR(BIN_OP)                                                    \
    inline expr BIN_OP(expr a, expr b) {                                                           \
        return make_binary_operation(binary_operation::BIN_OP, std::move(a), std::move(b));        \
    }

namespace clir {

namespace internal {
class expr_node;
}

enum class cl_mem_fence_flags;
class data_type;
class expr;
class stmt;
class var;

CLIR_EXPORT expr call_builtin(builtin_function fn, std::vector<expr> args);
CLIR_EXPORT expr call_external(std::string name, std::vector<expr> args);
CLIR_EXPORT expr cast(data_type to, expr term);

CLIR_EXPORT stmt declaration(data_type ty, var v, std::vector<attr> attributes = {});
CLIR_EXPORT stmt declaration_assignment(data_type ty, var v, expr b,
                                        std::vector<attr> attributes = {});
CLIR_EXPORT expr dereference(expr a);
CLIR_EXPORT expr address_of(expr a);
CLIR_EXPORT expr ternary_conditional(expr condition, expr then, expr otherwise);
CLIR_EXPORT expr make_unary_operation(unary_operation op, expr a);
CLIR_EXPORT expr make_binary_operation(binary_operation op, expr a, expr b);

CLIR_EXPORT short choose_bits(int64_t value);
CLIR_EXPORT short choose_bits(uint64_t value);
CLIR_EXPORT expr constant(int64_t value, short bits);
CLIR_EXPORT expr constant(uint64_t value, short bits);

CLIR_EXPORT expr init_vector(data_type ty, std::vector<expr> args);

class CLIR_EXPORT expr : public handle<internal::expr_node> {
  public:
    using handle<internal::expr_node>::handle;
    expr(int8_t value);
    expr(int16_t value);
    expr(int32_t value);
    expr(int64_t value);
    expr(int64_t value, short bits);
    expr(uint8_t value);
    expr(uint16_t value);
    expr(uint32_t value);
    expr(uint64_t value);
    expr(uint64_t value, short bits);
    expr(float value);
    expr(double value);
    expr(double value, short bits);
    expr(cl_mem_fence_flags value);
    expr(char const *value);
    expr(std::string value);

    expr operator[](expr a) const;
    expr operator[](int8_t a) const;
    expr operator[](int16_t a) const;
    expr operator[](int32_t a) const;
    expr operator[](int64_t a) const;
    expr operator[](uint8_t a) const;
    expr operator[](uint16_t a) const;
    expr operator[](uint32_t a) const;
    expr operator[](uint64_t a) const;

    expr s(short i0) const;
    expr s(short i0, short i1) const;
    expr s(short i0, short i1, short i2) const;
    expr s(short i0, short i1, short i2, short i3) const;
    expr s(short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7) const;
    expr s(short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7, short i8,
           short i9, short i10, short i11, short i12, short i13, short i14, short i15) const;
    expr lo() const;
    expr hi() const;
    expr even() const;
    expr odd() const;
};

CLIR_MAKE_UNARY_OPERATOR(-, minus)
CLIR_MAKE_UNARY_OPERATOR(~, bitwise_not)
CLIR_MAKE_UNARY_OPERATOR(!, logical_not)
CLIR_MAKE_UNARY_OPERATOR(++, pre_increment)
CLIR_MAKE_UNARY_OPERATOR(--, pre_decrement)
inline expr operator++(expr a, int) {
    return make_unary_operation(unary_operation::post_increment, std::move(a));
}
inline expr operator--(expr a, int) {
    return make_unary_operation(unary_operation::post_decrement, std::move(a));
}

CLIR_MAKE_BINARY_OPERATOR(+, add)
CLIR_MAKE_BINARY_OPERATOR(-, subtract)
CLIR_MAKE_BINARY_OPERATOR(*, multiply)
CLIR_MAKE_BINARY_OPERATOR(/, divide)
CLIR_MAKE_BINARY_OPERATOR(%, modulo)
CLIR_MAKE_BINARY_OPERATOR(>, greater_than)
CLIR_MAKE_BINARY_OPERATOR(<, less_than)
CLIR_MAKE_BINARY_OPERATOR(>=, greater_than_or_equal)
CLIR_MAKE_BINARY_OPERATOR(<=, less_than_or_equal)
CLIR_MAKE_BINARY_OPERATOR(==, equal)
CLIR_MAKE_BINARY_OPERATOR(!=, not_equal)
CLIR_MAKE_BINARY_OPERATOR(&, bitwise_and)
CLIR_MAKE_BINARY_OPERATOR(|, bitwise_or)
CLIR_MAKE_BINARY_OPERATOR(^, bitwise_xor)
CLIR_MAKE_BINARY_OPERATOR(&&, logical_and)
CLIR_MAKE_BINARY_OPERATOR(||, logical_or)
CLIR_MAKE_BINARY_OPERATOR(<<, left_shift)
CLIR_MAKE_BINARY_OPERATOR(>>, right_shift)

CLIR_MAKE_NAMED_BINARY_OPERATOR(assignment)
CLIR_MAKE_NAMED_BINARY_OPERATOR(add_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(subtract_from)
CLIR_MAKE_NAMED_BINARY_OPERATOR(multiply_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(divide_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(modulus_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(left_shift_by)
CLIR_MAKE_NAMED_BINARY_OPERATOR(right_shift_by)
CLIR_MAKE_NAMED_BINARY_OPERATOR(and_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(or_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(xor_into)
CLIR_MAKE_NAMED_BINARY_OPERATOR(comma)

} // namespace clir

#endif // EXPR_20220405_HPP
