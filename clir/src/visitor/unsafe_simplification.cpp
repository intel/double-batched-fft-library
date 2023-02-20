// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/unsafe_simplification.hpp"
#include "clir/builtin_function.hpp"
#include "clir/kernel.hpp"
#include "clir/op.hpp"
#include "clir/stmt.hpp"
#include "clir/visit.hpp"

namespace clir {

/* Expr nodes */
expr unsafe_simplification::operator()(internal::variable &) { return nullptr; }
expr unsafe_simplification::operator()(internal::int_imm &) { return nullptr; }
expr unsafe_simplification::operator()(internal::uint_imm &) { return nullptr; }
expr unsafe_simplification::operator()(internal::float_imm &) { return nullptr; }
expr unsafe_simplification::operator()(internal::cl_mem_fence_flags_imm &) { return nullptr; }
expr unsafe_simplification::operator()(internal::string_imm &) { return nullptr; }

expr unsafe_simplification::operator()(internal::unary_op &e) {
    if (auto t = visit(*this, *e.term()); t) {
        e.term(std::move(t));
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::binary_op &e) {
    if (auto lhs = visit(*this, *e.lhs()); lhs) {
        e.lhs(std::move(lhs));
    }
    if (auto rhs = visit(*this, *e.rhs()); rhs) {
        e.rhs(std::move(rhs));
    }

    auto lhs_type = visit(internal::determine_number_type{}, *e.lhs());
    auto rhs_type = visit(internal::determine_number_type{}, *e.rhs());
    if (lhs_type == internal::number_type::zero) {
        switch (e.op()) {
        case binary_operation::add:
        case binary_operation::bitwise_or:
        case binary_operation::bitwise_xor:
            return e.rhs();
        case binary_operation::subtract:
            return -e.rhs();
        case binary_operation::multiply:
        case binary_operation::divide:
        case binary_operation::modulo:
        case binary_operation::bitwise_and:
        case binary_operation::left_shift:
        case binary_operation::right_shift:
            return e.lhs();
        default:
            break;
        }
    }
    if (rhs_type == internal::number_type::zero) {
        switch (e.op()) {
        case binary_operation::add:
        case binary_operation::bitwise_or:
        case binary_operation::bitwise_xor:
        case binary_operation::subtract:
        case binary_operation::left_shift:
        case binary_operation::right_shift:
            return e.lhs();
        case binary_operation::multiply:
        case binary_operation::bitwise_and:
            return e.rhs();
        default:
            break;
        }
    }
    if (lhs_type == internal::number_type::one) {
        switch (e.op()) {
        case binary_operation::multiply:
            return e.rhs();
        default:
            break;
        }
    }
    if (rhs_type == internal::number_type::one) {
        switch (e.op()) {
        case binary_operation::multiply:
        case binary_operation::divide:
            return e.lhs();
        case binary_operation::modulo:
            return 0;
        default:
            break;
        }
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::ternary_op &e) {
    if (auto t = visit(*this, *e.term0()); t) {
        e.term0(std::move(t));
    }
    if (auto t = visit(*this, *e.term1()); t) {
        e.term1(std::move(t));
    }
    if (auto t = visit(*this, *e.term2()); t) {
        e.term2(std::move(t));
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::access &e) {
    if (auto f = visit(*this, *e.field()); f) {
        e.field(std::move(f));
    }
    if (auto a = visit(*this, *e.address()); a) {
        e.address(std::move(a));
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::call_builtin &fn) {
    for (auto &arg : fn.args()) {
        if (auto a = visit(*this, *arg); a) {
            arg = std::move(a);
        }
    }
    if (fn.fn() == builtin_function::intel_sub_group_shuffle_xor && fn.args().size() > 0) {
        auto arg0_type = visit(internal::determine_number_type{}, *fn.args().front());
        if (arg0_type == internal::number_type::zero) {
            return fn.args().front();
        }
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::call_external &fn) {
    for (auto &arg : fn.args()) {
        if (auto a = visit(*this, *arg); a) {
            arg = std::move(a);
        }
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::cast &c) {
    if (auto t = visit(*this, *c.term()); t) {
        c.term(std::move(t));
    }
    return nullptr;
}

expr unsafe_simplification::operator()(internal::swizzle &s) {
    if (auto t = visit(*this, *s.term()); t) {
        s.term(std::move(t));
    }
    return nullptr;
}

/* Stmt nodes */
void unsafe_simplification::operator()(internal::declaration &) {}
void unsafe_simplification::operator()(internal::declaration_assignment &d) {
    if (auto r = visit(*this, *d.rhs()); r) {
        d.rhs(std::move(r));
    }
}
void unsafe_simplification::operator()(internal::expression_statement &e) {
    if (auto r = visit(*this, *e.term()); r) {
        e.term(std::move(r));
    }
}

void unsafe_simplification::operator()(internal::block &b) {
    for (auto &s : b.stmts()) {
        visit(*this, *s);
    }
}

void unsafe_simplification::operator()(internal::for_loop &loop) {
    visit(*this, *loop.start());
    if (auto r = visit(*this, *loop.condition()); r) {
        loop.condition(std::move(r));
    }
    if (auto r = visit(*this, *loop.step()); r) {
        loop.step(std::move(r));
    }
    visit(*this, *loop.body());
}

void unsafe_simplification::operator()(internal::if_selection &is) {
    if (auto r = visit(*this, *is.condition()); r) {
        is.condition(std::move(r));
    }
    visit(*this, *is.then());
    if (is.otherwise()) {
        visit(*this, **is.otherwise());
    }
}

/* Kernel nodes */
void unsafe_simplification::operator()(internal::prototype &) {}
void unsafe_simplification::operator()(internal::function &fn) { visit(*this, *fn.body()); }

expr unsafe_simplify(expr e) {
    auto f = visit(unsafe_simplification{}, *e);
    return f ? f : e;
}
void unsafe_simplify(stmt s) { visit(unsafe_simplification{}, *s); }
void unsafe_simplify(kernel k) { visit(unsafe_simplification{}, *k); }

} // namespace clir
