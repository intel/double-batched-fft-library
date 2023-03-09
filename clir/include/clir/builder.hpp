// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BUILDER_20220405_HPP
#define BUILDER_20220405_HPP

#include "clir/attr.hpp"
#include "clir/builtin_type.hpp"
#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/internal/program_node.hpp"
#include "clir/internal/stmt_node.hpp"
#include "clir/prog.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"

#include <memory>
#include <string>
#include <vector>

namespace clir {

class data_type;
class func;
class var;

namespace internal {
class CLIR_EXPORT declaration_builder {
  public:
    void declare(data_type ty, var v, std::vector<attr> attributes = {});
    var declare(data_type ty, std::string prefix, std::vector<attr> attributes = {});
    void declare_assign(data_type ty, var v, expr b, std::vector<attr> attributes = {});
    var declare_assign(data_type ty, std::string prefix, expr b, std::vector<attr> attributes = {});

  protected:
    virtual void add(std::shared_ptr<internal::declaration> d) = 0;
    virtual void add(std::shared_ptr<internal::declaration_assignment> da) = 0;
};
} // namespace internal

class CLIR_EXPORT block_builder : public internal::declaration_builder {
  public:
    block_builder();
    block_builder(std::shared_ptr<internal::block> block);

    stmt get_product();

    void add(expr e);
    void add(stmt s);
    void assign(expr a, expr b);

    template <typename F> block_builder &body(F &&f) {
        f(*this);
        return *this;
    }

  protected:
    void add(std::shared_ptr<internal::declaration> d) override;
    void add(std::shared_ptr<internal::declaration_assignment> da) override;

  private:
    std::shared_ptr<internal::block> block_;
};

class CLIR_EXPORT for_loop_builder {
  public:
    for_loop_builder(expr start, expr condition, expr step);
    for_loop_builder(stmt start, expr condition, expr step);

    stmt get_product();

    for_loop_builder &attribute(attr a);
    template <typename F> for_loop_builder &body(F &&f) {
        auto bb = block_builder{};
        f(bb);
        body_ = bb.get_product();
        return *this;
    }

  private:
    stmt start_;
    expr condition_;
    expr step_;
    stmt body_;
    std::vector<attr> attributes_;
};

class CLIR_EXPORT if_selection_builder {
  public:
    if_selection_builder(expr condition);

    stmt get_product();

    template <typename F> if_selection_builder &then(F &&f) {
        auto bb = block_builder{};
        f(bb);
        then_ = bb.get_product();
        return *this;
    }

    template <typename F> if_selection_builder &otherwise(F &&f) {
        auto bb = block_builder{};
        f(bb);
        otherwise_ = bb.get_product();
        return *this;
    }

    if_selection_builder &otherwise(stmt other);

  private:
    expr condition_;
    stmt then_;
    stmt otherwise_;
};

class CLIR_EXPORT function_builder {
  public:
    function_builder(std::string name);

    func get_product();

    void argument(data_type ty, var v);
    void qualifier(function_qualifier q);
    void attribute(attr a);
    template <typename F> void body(F &&f) {
        auto bb = block_builder{};
        f(bb);
        body_ = bb.get_product();
    }

  private:
    std::shared_ptr<internal::prototype> proto_;
    stmt body_;
};

class CLIR_EXPORT kernel_builder : public function_builder {
  public:
    kernel_builder(std::string name);
};

class CLIR_EXPORT program_builder : public internal::declaration_builder {
  public:
    program_builder();

    prog get_product();
    void add(func f);

  protected:
    void add(std::shared_ptr<internal::declaration> d) override;
    void add(std::shared_ptr<internal::declaration_assignment> da) override;

  private:
    std::shared_ptr<internal::program> program_;
};

} // namespace clir

#endif // BUILDER_20220405_HPP
