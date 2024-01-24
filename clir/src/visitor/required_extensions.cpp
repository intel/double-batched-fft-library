// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/required_extensions.hpp"
#include "clir/func.hpp"
#include "clir/prog.hpp"
#include "clir/visit.hpp"

#include <stdexcept>

namespace clir {

/* Expr nodes */
void required_extensions::operator()(internal::expr_node &) {}
void required_extensions::operator()(internal::call_builtin &fn) {
    auto ext = get_extension(fn.fn());
    if (ext == extension::unknown) {
        throw std::logic_error("Unknown extension");
    }
    needs_ext_[static_cast<int>(ext)] = true;
}

/* Stmt nodes */
void required_extensions::operator()(internal::stmt_node &) {}
void required_extensions::operator()(internal::declaration_assignment &d) {
    visit(*this, *d.rhs());
}
void required_extensions::operator()(internal::expression_statement &e) { visit(*this, *e.term()); }
void required_extensions::operator()(internal::block &b) {
    for (auto &s : b.stmts()) {
        visit(*this, *s);
    }
}
void required_extensions::operator()(internal::for_loop &loop) {
    visit(*this, *loop.start());
    visit(*this, *loop.condition());
    visit(*this, *loop.step());
    visit(*this, *loop.body());
}

void required_extensions::operator()(internal::if_selection &is) {
    visit(*this, *is.condition());
    if (is.otherwise()) {
        visit(*this, *is.then());
        visit(*this, **is.otherwise());
    } else {
        visit(*this, *is.then());
    }
}

/* Kernel nodes */
void required_extensions::operator()(internal::prototype &) {}
void required_extensions::operator()(internal::function &fn) {
    visit(*this, *fn.prototype());
    visit(*this, *fn.body());
}
void required_extensions::operator()(internal::global_declaration &d) { visit(*this, *d.term()); }

/* Program nodes */
void required_extensions::operator()(internal::program &prg) {
    for (auto &d : prg.declarations()) {
        visit(*this, *d);
    }
}

auto required_extensions::extensions() const -> std::vector<extension> {
    auto result = std::vector<extension>{};
    for (int i = 0; i < static_cast<int>(needs_ext_.size()); ++i) {
        if (needs_ext_[i] && i != static_cast<int>(extension::builtin)) {
            result.emplace_back(static_cast<extension>(i));
        }
    }
    return result;
}

auto get_required_extensions(prog p) -> std::vector<extension> {
    auto r = required_extensions{};
    visit(r, *p);
    return r.extensions();
}
auto get_required_extensions(func k) -> std::vector<extension> {
    auto r = required_extensions{};
    visit(r, *k);
    return r.extensions();
}

} // namespace clir
