// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EXPR_NODE_20220405_HPP
#define EXPR_NODE_20220405_HPP

#include "clir/builtin_function.hpp"
#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/op.hpp"
#include "clir/virtual_type_list.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace clir::internal {

class CLIR_EXPORT expr_node
    : public virtual_type_list<class variable, class int_imm, class uint_imm, class float_imm,
                               class cl_mem_fence_flags_imm, class string_imm, class unary_op,
                               class binary_op, class ternary_op, class access, class call_builtin,
                               class call, class cast, class swizzle> {
  public:
    virtual unsigned precedence() const = 0;
    virtual associativity assoc() const = 0;
};

class CLIR_EXPORT variable : public visitable<variable, expr_node> {
  public:
    variable(std::string prefix = "") : name_(std::move(prefix)) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    std::string_view name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

  private:
    std::string name_;
};

class CLIR_EXPORT int_imm : public visitable<int_imm, expr_node> {
  public:
    int_imm(int8_t value) : value_(static_cast<int64_t>(value)), bits_(8) {}
    int_imm(int16_t value) : value_(static_cast<int64_t>(value)), bits_(16) {}
    int_imm(int32_t value) : value_(static_cast<int64_t>(value)), bits_(32) {}
    int_imm(int64_t value) : value_(value), bits_(64) {}
    int_imm(int64_t value, short bits) : value_(value), bits_(bits) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    int64_t value() const { return value_; }
    short bits() const { return bits_; }

  private:
    int64_t value_;
    short bits_;
};

class CLIR_EXPORT uint_imm : public visitable<uint_imm, expr_node> {
  public:
    uint_imm(uint8_t value) : value_(static_cast<uint64_t>(value)), bits_(8) {}
    uint_imm(uint16_t value) : value_(static_cast<uint64_t>(value)), bits_(16) {}
    uint_imm(uint32_t value) : value_(static_cast<uint64_t>(value)), bits_(32) {}
    uint_imm(uint64_t value) : value_(value), bits_(64) {}
    uint_imm(uint64_t value, short bits) : value_(value), bits_(bits) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    uint64_t value() const { return value_; }
    short bits() const { return bits_; }

  private:
    uint64_t value_;
    short bits_;
};

class CLIR_EXPORT float_imm : public visitable<float_imm, expr_node> {
  public:
    float_imm(float value) : value_(value), bits_(32) {}
    float_imm(double value) : value_(value), bits_(64) {}
    float_imm(double value, short bits) : value_(value), bits_(bits) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    double value() const { return value_; }
    short bits() const { return bits_; }

  private:
    double value_;
    short bits_;
};

class CLIR_EXPORT cl_mem_fence_flags_imm : public visitable<cl_mem_fence_flags_imm, expr_node> {
  public:
    cl_mem_fence_flags_imm(cl_mem_fence_flags value) : value_(value) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    auto value() const { return value_; }

  private:
    cl_mem_fence_flags value_;
};

class CLIR_EXPORT string_imm : public visitable<string_imm, expr_node> {
  public:
    string_imm(char const *value) : value_(value) {}
    string_imm(std::string value) : value_(std::move(value)) {}
    unsigned precedence() const override { return 0; }
    associativity assoc() const override { return associativity::none; }

    std::string_view value() const { return value_; }

  private:
    std::string value_;
};

class CLIR_EXPORT unary_op : public visitable<unary_op, expr_node> {
  public:
    unary_op(unary_operation op, expr term) : op_(op), term_(std::move(term)) {}
    unsigned precedence() const override { return operation_precedence(op_); }
    associativity assoc() const override { return operation_associativity(op_); }

    unary_operation op() { return op_; }
    expr &term() { return term_; }
    void term(expr e) { term_ = std::move(e); }

  private:
    unary_operation op_;
    expr term_;
};

class CLIR_EXPORT binary_op : public visitable<binary_op, expr_node> {
  public:
    binary_op(binary_operation op, expr lhs, expr rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
    unsigned precedence() const override { return operation_precedence(op_); }
    associativity assoc() const override { return operation_associativity(op_); }

    binary_operation op() { return op_; }
    expr &lhs() { return lhs_; }
    expr &rhs() { return rhs_; }
    void lhs(expr e) { lhs_ = std::move(e); }
    void rhs(expr e) { rhs_ = std::move(e); }

  private:
    binary_operation op_;
    expr lhs_, rhs_;
};

class CLIR_EXPORT ternary_op : public visitable<ternary_op, expr_node> {
  public:
    ternary_op(ternary_operation op, expr term0, expr term1, expr term2)
        : op_(op), term0_(std::move(term0)), term1_(std::move(term1)), term2_(std::move(term2)) {}
    unsigned precedence() const override { return operation_precedence(op_); }
    associativity assoc() const override { return operation_associativity(op_); }

    ternary_operation op() { return op_; }
    expr &term0() { return term0_; }
    expr &term1() { return term1_; }
    expr &term2() { return term2_; }
    void term0(expr e) { term0_ = std::move(e); }
    void term1(expr e) { term1_ = std::move(e); }
    void term2(expr e) { term2_ = std::move(e); }

  private:
    ternary_operation op_;
    expr term0_, term1_, term2_;
};

class CLIR_EXPORT access : public visitable<access, expr_node> {
  public:
    access(expr field, expr address) : field_(std::move(field)), address_(std::move(address)) {}
    unsigned precedence() const override { return 1; }
    associativity assoc() const override { return associativity::left_to_right; }

    expr &field() { return field_; }
    expr &address() { return address_; }
    void field(expr e) { field_ = std::move(e); }
    void address(expr e) { address_ = std::move(e); }

  private:
    expr field_, address_;
};

class CLIR_EXPORT call_builtin : public visitable<call_builtin, expr_node> {
  public:
    call_builtin(builtin_function fn, std::vector<expr> args)
        : fn_(std::move(fn)), args_(std::move(args)) {}
    unsigned precedence() const override { return 1; }
    associativity assoc() const override { return associativity::left_to_right; }

    builtin_function fn() const { return fn_; }
    std::vector<expr> &args() { return args_; }

  private:
    builtin_function fn_;
    std::vector<expr> args_;
};

class CLIR_EXPORT call : public visitable<call, expr_node> {
  public:
    call(std::string name, std::vector<expr> args)
        : name_(std::move(name)), args_(std::move(args)) {}
    unsigned precedence() const override { return 1; }
    associativity assoc() const override { return associativity::left_to_right; }

    std::string_view name() const { return name_; }
    std::vector<expr> &args() { return args_; }

  private:
    std::string name_;
    std::vector<expr> args_;
};

class CLIR_EXPORT cast : public visitable<cast, expr_node> {
  public:
    cast(data_type to, expr term) : to_(std::move(to)), term_(std::move(term)) {}
    unsigned precedence() const override { return 2; }
    associativity assoc() const override { return associativity::right_to_left; }

    data_type &target_ty() { return to_; }
    expr &term() { return term_; }
    void term(expr e) { term_ = std::move(e); }

  private:
    data_type to_;
    expr term_;
};

enum class swizzle_selector { index, lo, hi, even, odd };
class CLIR_EXPORT swizzle : public visitable<swizzle, expr_node> {
  public:
    swizzle(expr term, swizzle_selector sel) : term_(std::move(term)), selector_(sel), indices_{} {}
    swizzle(expr term, std::vector<short> indices)
        : term_(std::move(term)), selector_(swizzle_selector::index), indices_(std::move(indices)) {
    }
    unsigned precedence() const override { return 1; }
    associativity assoc() const override { return associativity::left_to_right; }

    expr &term() { return term_; }
    void term(expr e) { term_ = std::move(e); }
    swizzle_selector selector() const { return selector_; }
    std::vector<short> &indices() { return indices_; }

  private:
    expr term_;
    swizzle_selector selector_;
    std::vector<short> indices_;
};

} // namespace clir::internal

#endif // EXPR_NODE_20220405_HPP
