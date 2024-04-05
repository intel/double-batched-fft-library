// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/unique_names.hpp"
#include "clir/expr.hpp"
#include "clir/func.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/prog.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/visit.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace clir {

void unique_names::make_unique_name(var e) {
    internal::variable *v = dynamic_cast<internal::variable *>(e.get());
    if (v == nullptr) {
        throw std::runtime_error("unique_names.declaration: Expected variable");
    }

    if (declared_vars_.empty()) {
        throw std::runtime_error("unique_names.declaration: Missing scope");
    }

    uintptr_t u = reinterpret_cast<uintptr_t>(v);
    for (auto const &vars : declared_vars_) {
        if (vars.find(u) != vars.end()) {
            throw std::runtime_error("unique_names.declaration: Variable already declared");
        }
    }
    declared_vars_.back().emplace(u);

    if (v->name().empty()) {
        v->set_name("x");
    }

    auto name = std::string(v->name());

    for (auto &name_counter : name_counters_) {
        auto nc = name_counter.find(name);
        if (nc != name_counter.end()) {
            std::string new_name;
            do {
                ++nc->second;
                new_name = name + std::to_string(nc->second);
            } while (name_counter.find(new_name) != name_counter.end());
            v->set_name(new_name);
            name = new_name;
            break;
        }
    }
    name_counters_.back()[name] = 0;
}

/* Expr nodes */
void unique_names::operator()(internal::expr_node &) {}

/* Stmt nodes */
void unique_names::operator()(internal::declaration &d) { make_unique_name(d.variable()); }

void unique_names::operator()(internal::declaration_assignment &d) { operator()(d.decl()); }

void unique_names::operator()(internal::expression_statement &e) { visit(*this, *e.term()); }

void unique_names::operator()(internal::block &b) {
    push_scope();
    for (auto &s : b.stmts()) {
        visit(*this, *s);
    }
    pop_scope();
}

void unique_names::operator()(internal::for_loop &loop) {
    push_scope();
    visit(*this, *loop.start());
    visit(*this, *loop.body());
    pop_scope();
}

void unique_names::operator()(internal::if_selection &is) {
    push_scope();
    visit(*this, *is.then());
    if (is.otherwise()) {
        visit(*this, **is.otherwise());
    }
    pop_scope();
}

void unique_names::operator()(internal::while_loop &loop) {
    push_scope();
    visit(*this, *loop.body());
    pop_scope();
}

/* Kernel nodes */
void unique_names::operator()(internal::prototype &p) {
    for (auto const &[t, v] : p.args()) {
        make_unique_name(v);
    }
}
void unique_names::operator()(internal::function &fn) {
    push_scope();
    visit(*this, *fn.prototype());
    visit(*this, *fn.body());
    pop_scope();
}
void unique_names::operator()(internal::global_declaration &d) { visit(*this, *d.term()); }

/* Program nodes */
void unique_names::operator()(internal::program &prg) {
    push_scope();
    for (auto &d : prg.declarations()) {
        visit(*this, *d);
    }
    pop_scope();
}

/* Helper */
void unique_names::push_scope() {
    declared_vars_.push_back(std::unordered_set<uintptr_t>{});
    name_counters_.push_back(std::unordered_map<std::string, unsigned long>{});
}

void unique_names::pop_scope() {
    declared_vars_.pop_back();
    name_counters_.pop_back();
}

void make_names_unique(prog p) { visit(unique_names{}, *p); }
void make_names_unique(func k) { visit(unique_names{}, *k); }
void make_names_unique(stmt s) { visit(unique_names{}, *s); }

} // namespace clir
