// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef STMT_NODE_20220405_HPP
#define STMT_NODE_20220405_HPP

#include "clir/attr.hpp"
#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/op.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/virtual_type_list.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace clir::internal {

class CLIR_EXPORT stmt_node
    : public virtual_type_list<class declaration, class declaration_assignment,
                               class expression_statement, class block, class for_loop,
                               class if_selection> {};

class CLIR_EXPORT declaration : public visitable<declaration, stmt_node> {
  public:
    declaration(data_type ty, var v, std::vector<attr> attributes = {})
        : ty_(std::move(ty)), v_(std::move(v)), attributes_(std::move(attributes)) {}

    data_type &ty() { return ty_; }
    var &variable() { return v_; }
    std::vector<attr> &attributes() { return attributes_; }

  private:
    data_type ty_;
    var v_;
    std::vector<attr> attributes_;
};

class CLIR_EXPORT declaration_assignment : public visitable<declaration_assignment, stmt_node> {
  public:
    declaration_assignment(std::shared_ptr<declaration> decl, expr rhs)
        : decl_(std::move(decl)), rhs_(std::move(rhs)) {}

    declaration &decl() { return *decl_; }
    expr &rhs() { return rhs_; }
    void rhs(expr e) { rhs_ = std::move(e); }

  private:
    std::shared_ptr<declaration> decl_;
    expr rhs_;
};

class CLIR_EXPORT expression_statement : public visitable<expression_statement, stmt_node> {
  public:
    expression_statement(expr term) : term_(std::move(term)) {}
    expr &term() { return term_; }
    void term(expr e) { term_ = std::move(e); }

  private:
    expr term_;
};

class CLIR_EXPORT block : public visitable<block, stmt_node> {
  public:
    block() {}
    block(std::vector<stmt> stmts) : stmts_(std::move(stmts)) {}
    std::vector<stmt> &stmts() { return stmts_; }

  private:
    std::vector<stmt> stmts_;
};

class CLIR_EXPORT for_loop : public visitable<for_loop, stmt_node> {
  public:
    for_loop(stmt start, expr condition, expr step, stmt body, std::vector<attr> attributes = {})
        : start_(std::move(start)), condition_(std::move(condition)), step_(std::move(step)),
          body_(std::move(body)), attributes_(std::move(attributes)) {}

    stmt &start() { return start_; }
    expr &condition() { return condition_; }
    void condition(expr e) { condition_ = std::move(e); }
    expr &step() { return step_; }
    void step(expr e) { step_ = std::move(e); }
    stmt &body() { return body_; }
    std::vector<attr> &attributes() { return attributes_; }

  private:
    stmt start_;
    expr condition_;
    expr step_;
    stmt body_;
    std::vector<attr> attributes_;
};

class CLIR_EXPORT if_selection : public visitable<if_selection, stmt_node> {
  public:
    if_selection(expr condition, stmt then, std::optional<stmt> otherwise = std::nullopt)
        : condition_(std::move(condition)), then_(std::move(then)),
          otherwise_(std::move(otherwise)) {}

    expr &condition() { return condition_; }
    void condition(expr e) { condition_ = std::move(e); }
    stmt &then() { return then_; }
    std::optional<stmt> &otherwise() { return otherwise_; }

  private:
    expr condition_;
    stmt then_;
    std::optional<stmt> otherwise_;
};

} // namespace clir::internal

#endif // STMT_NODE_20220405_HPP
