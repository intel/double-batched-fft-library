// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/expr.hpp"
#include "clir/data_type.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/internal/stmt_node.hpp"
#include "clir/op.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"

#include <limits>
#include <numeric>

namespace clir {

expr::expr(int8_t value) : expr(std::make_shared<internal::int_imm>(value)) {}
expr::expr(int16_t value) : expr(std::make_shared<internal::int_imm>(value)) {}
expr::expr(int32_t value) : expr(std::make_shared<internal::int_imm>(value)) {}
expr::expr(int64_t value) : expr(std::make_shared<internal::int_imm>(value, choose_bits(value))) {}
expr::expr(int64_t value, short bits) : expr(std::make_shared<internal::int_imm>(value, bits)) {}
expr::expr(uint8_t value) : expr(std::make_shared<internal::uint_imm>(value)) {}
expr::expr(uint16_t value) : expr(std::make_shared<internal::uint_imm>(value)) {}
expr::expr(uint32_t value) : expr(std::make_shared<internal::uint_imm>(value)) {}
expr::expr(uint64_t value)
    : expr(std::make_shared<internal::uint_imm>(value, choose_bits(value))) {}
expr::expr(uint64_t value, short bits) : expr(std::make_shared<internal::uint_imm>(value, bits)) {}
expr::expr(float value) : expr(std::make_shared<internal::float_imm>(value)) {}
expr::expr(double value) : expr(std::make_shared<internal::float_imm>(value)) {}
expr::expr(double value, short bits) : expr(std::make_shared<internal::float_imm>(value, bits)) {}
expr::expr(cl_mem_fence_flags value)
    : expr(std::make_shared<internal::cl_mem_fence_flags_imm>(value)) {}
expr::expr(char const *value) : expr(std::make_shared<internal::string_imm>(std::move(value))) {}
expr::expr(std::string value) : expr(std::make_shared<internal::string_imm>(std::move(value))) {}

expr call_builtin(builtin_function fn, std::vector<expr> args) {
    return expr(std::make_shared<internal::call_builtin>(fn, std::move(args)));
}
expr call_external(std::string name, std::vector<expr> args) {
    return expr(std::make_shared<internal::call_external>(std::move(name), std::move(args)));
}
expr cast(data_type to, expr term) {
    return expr(std::make_shared<internal::cast>(std::move(to), std::move(term)));
}

expr expr::operator[](expr a) const {
    return expr(std::make_shared<internal::access>(*this, std::move(a)));
}
expr expr::operator[](int8_t a) const { return expr(std::make_shared<internal::access>(*this, a)); }
expr expr::operator[](int16_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](int32_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](int64_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](uint8_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](uint16_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](uint32_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}
expr expr::operator[](uint64_t a) const {
    return expr(std::make_shared<internal::access>(*this, a));
}

expr expr::s(short i0) const {
    return expr(std::make_shared<internal::swizzle>(*this, std::vector<short>{i0}));
}
expr expr::s(short i0, short i1) const {
    return expr(std::make_shared<internal::swizzle>(*this, std::vector<short>{i0, i1}));
}
expr expr::s(short i0, short i1, short i2) const {
    return expr(std::make_shared<internal::swizzle>(*this, std::vector<short>{i0, i1, i2}));
}
expr expr::s(short i0, short i1, short i2, short i3) const {
    return expr(std::make_shared<internal::swizzle>(*this, std::vector<short>{i0, i1, i2, i3}));
}
expr expr::s(short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7) const {
    return expr(std::make_shared<internal::swizzle>(
        *this, std::vector<short>{i0, i1, i2, i3, i4, i5, i6, i7}));
}
expr expr::s(short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
             short i8, short i9, short i10, short i11, short i12, short i13, short i14,
             short i15) const {
    return expr(std::make_shared<internal::swizzle>(
        *this,
        std::vector<short>{i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15}));
}
expr expr::lo() const {
    return expr(std::make_shared<internal::swizzle>(*this, internal::swizzle_selector::lo));
}
expr expr::hi() const {
    return expr(std::make_shared<internal::swizzle>(*this, internal::swizzle_selector::hi));
}
expr expr::even() const {
    return expr(std::make_shared<internal::swizzle>(*this, internal::swizzle_selector::even));
}
expr expr::odd() const {
    return expr(std::make_shared<internal::swizzle>(*this, internal::swizzle_selector::odd));
}

stmt declaration(data_type ty, var v, std::vector<attr> attributes) {
    return stmt(std::make_shared<internal::declaration>(std::move(ty), std::move(v),
                                                        std::move(attributes)));
}

stmt declaration_assignment(data_type ty, var v, expr b, std::vector<attr> attributes) {
    auto decl =
        std::make_shared<internal::declaration>(std::move(ty), std::move(v), std::move(attributes));
    return stmt(std::make_shared<internal::declaration_assignment>(std::move(decl), std::move(b)));
}

expr dereference(expr a) {
    return make_unary_operation(unary_operation::indirection, std::move(a));
}

expr address_of(expr a) { return make_unary_operation(unary_operation::address, std::move(a)); }

expr make_unary_operation(unary_operation op, expr a) {
    return expr(std::make_shared<internal::unary_op>(op, std::move(a)));
}

expr make_binary_operation(binary_operation op, expr a, expr b) {
    return expr(std::make_shared<internal::binary_op>(op, std::move(a), std::move(b)));
}

short choose_bits(int64_t value) {
    if (std::numeric_limits<int32_t>::min() <= value &&
        value <= std::numeric_limits<int32_t>::max()) {
        return 32;
    }
    return 64;
}

short choose_bits(uint64_t value) {
    if (value <= std::numeric_limits<uint32_t>::max()) {
        return 32;
    }
    return 64;
}

expr constant(int64_t value, short bits) {
    return expr(std::make_shared<internal::int_imm>(value, bits));
}

expr constant(uint64_t value, short bits) {
    return expr(std::make_shared<internal::uint_imm>(value, bits));
}

expr init_vector(data_type ty, std::vector<expr> args) {
    if (args.size() == 0) {
        return nullptr;
    }
    expr list = args[0];
    for (std::size_t i = 1; i < args.size(); ++i) {
        list = comma(list, args[i]);
    }
    return cast(ty, list);
}

} // namespace clir
