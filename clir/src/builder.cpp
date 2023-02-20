// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builder.hpp"
#include "clir/data_type.hpp"
#include "clir/kernel.hpp"
#include "clir/var.hpp"

#include <optional>
#include <utility>

namespace clir {

/* block builder */
block_builder::block_builder() : block_(std::make_shared<internal::block>()) {}
block_builder::block_builder(std::shared_ptr<internal::block> block) : block_(std::move(block)) {}

stmt block_builder::get_product() { return stmt(block_); }

void block_builder::add(expr e) {
    auto node = std::make_shared<internal::expression_statement>(std::move(e));
    add(stmt(std::move(node)));
}

void block_builder::add(stmt s) { block_->stmts().emplace_back(std::move(s)); }

void block_builder::assign(expr a, expr b) { add(assignment(std::move(a), std::move(b))); }

void block_builder::declare(data_type ty, var v, std::vector<attr> attributes) {
    auto node =
        std::make_shared<internal::declaration>(std::move(ty), std::move(v), std::move(attributes));
    add(stmt(std::move(node)));
}

var block_builder::declare(data_type ty, std::string prefix, std::vector<attr> attributes) {
    auto v = var(std::move(prefix));
    declare(std::move(ty), v, std::move(attributes));
    return v;
}

void block_builder::declare_assign(data_type ty, var v, expr b, std::vector<attr> attributes) {
    auto decl =
        std::make_shared<internal::declaration>(std::move(ty), std::move(v), std::move(attributes));
    auto node = std::make_shared<internal::declaration_assignment>(std::move(decl), std::move(b));
    add(stmt(std::move(node)));
}

var block_builder::declare_assign(data_type ty, std::string prefix, expr b,
                                  std::vector<attr> attributes) {
    auto v = var(std::move(prefix));
    declare_assign(std::move(ty), v, std::move(b), std::move(attributes));
    return v;
}

/* for loop builder */
for_loop_builder::for_loop_builder(expr start, expr condition, expr step)
    : start_(expression_statement(std::move(start))), condition_(std::move(condition)),
      step_(std::move(step)), body_(nullptr) {}
for_loop_builder::for_loop_builder(stmt start, expr condition, expr step)
    : start_(std::move(start)), condition_(std::move(condition)), step_(std::move(step)),
      body_(nullptr) {}

stmt for_loop_builder::get_product() {
    if (!body_.get()) {
        body_ = stmt(std::make_shared<internal::block>());
    }
    return stmt(
        std::make_shared<internal::for_loop>(start_, condition_, step_, body_, attributes_));
}

for_loop_builder &for_loop_builder::attribute(attr a) {
    attributes_.emplace_back(std::move(a));
    return *this;
}

/* if selection builder */
if_selection_builder::if_selection_builder(expr condition)
    : condition_(std::move(condition)), then_(nullptr), otherwise_(nullptr) {}

stmt if_selection_builder::get_product() {
    if (!then_.get()) {
        then_ = stmt(std::make_shared<internal::block>());
    }
    auto other = std::optional<stmt>{std::nullopt};
    if (otherwise_.get()) {
        other = std::make_optional(otherwise_);
    }
    return stmt(
        std::make_shared<internal::if_selection>(condition_, stmt(then_), std::move(other)));
}

if_selection_builder &if_selection_builder::otherwise(stmt other) {
    otherwise_ = std::move(other);
    return *this;
}

/* function builder */
function_builder::function_builder(std::string name)
    : proto_(std::make_shared<internal::prototype>(std::move(name))), body_(nullptr) {}

kernel function_builder::get_product() {
    if (!body_.get()) {
        body_ = stmt(std::make_shared<internal::block>());
    }
    return kernel(std::make_shared<internal::function>(kernel(proto_), body_));
}

void function_builder::argument(data_type ty, var v) {
    proto_->args().emplace_back(std::move(ty), std::move(v));
}

void function_builder::attribute(attr a) { proto_->attributes().emplace_back(std::move(a)); }

} // namespace clir
