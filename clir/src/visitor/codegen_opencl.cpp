// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/codegen_opencl.hpp"
#include "clir/attr.hpp"
#include "clir/builtin_function.hpp"
#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/func.hpp"
#include "clir/handle.hpp"
#include "clir/op.hpp"
#include "clir/stmt.hpp"
#include "clir/string_util.hpp"
#include "clir/var.hpp"
#include "clir/visit.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace clir {

codegen_opencl::codegen_opencl(std::ostream &os) : os_(os), stream_fmt_(os_.flags()) {
    os_ << std::hexfloat;
}
codegen_opencl::~codegen_opencl() { os_.setf(stream_fmt_); }

/* Attributes */
void codegen_opencl::operator()(internal::attr_node &attr) {
    os_ << "__attribute__((";
    attr.print(os_);
    os_ << "))";
}

/* Data type nodes */
void codegen_opencl::operator()(internal::scalar_data_type &v) {
    if (v.space() != address_space::generic_t) {
        os_ << v.space() << " ";
    }
    os_ << v.type();
}
void codegen_opencl::operator()(internal::vector_data_type &v) {
    if (v.space() != address_space::generic_t) {
        os_ << v.space() << " ";
    }

    os_ << v.type() << v.size();
}
void codegen_opencl::operator()(internal::pointer &v) {
    visit(*this, *v.ty());
    if (v.space() != address_space::generic_t) {
        os_ << " " << v.space();
    }
    os_ << "*";
}
void codegen_opencl::operator()(internal::array &a) {
    visit(*this, *a.ty());
    auto tmp = post_.str();
    post_.str(std::string());
    post_ << "[" << a.size() << "]" << tmp;
}

/* Expr nodes */
void codegen_opencl::operator()(internal::variable &v) { os_ << v.name(); }
void codegen_opencl::operator()(internal::int_imm &v) {
    os_ << v.value();
    int_suffix(v.bits());
}
void codegen_opencl::operator()(internal::uint_imm &v) {
    os_ << v.value();
    int_suffix(v.bits());
    os_ << 'u';
}
void codegen_opencl::operator()(internal::float_imm &v) {
    os_ << v.value();
    if (v.bits() == 32) {
        os_ << 'f';
    }
}
void codegen_opencl::operator()(internal::cl_mem_fence_flags_imm &v) { os_ << v.value(); }
void codegen_opencl::operator()(internal::string_imm &v) {
    os_ << "\"" << escaped_string(v.value()) << "\"";
}

void codegen_opencl::operator()(internal::unary_op &e) {
    auto a = e.assoc();
    if (a == associativity::left_to_right) {
        visit_check_parentheses(e, *e.term(), false);
    }
    os_ << e.op();
    if (a == associativity::right_to_left) {
        visit_check_parentheses(e, *e.term(), true);
    }
}

void codegen_opencl::operator()(internal::binary_op &e) {
    visit_check_parentheses(e, *e.lhs(), false);
    if (e.op() != binary_operation::comma) {
        os_ << " ";
    }
    os_ << e.op() << " ";
    visit_check_parentheses(e, *e.rhs(), true);
}

void codegen_opencl::operator()(internal::ternary_op &e) {
    visit_check_parentheses(e, *e.term0(), false);
    os_ << " " << to_string(e.op(), 0) << " ";
    visit_check_parentheses(e, *e.term1(), true);
    os_ << " " << to_string(e.op(), 1) << " ";
    visit_check_parentheses(e, *e.term2(), true);
}

void codegen_opencl::operator()(internal::access &e) {
    visit_check_parentheses(e, *e.field(), false);
    os_ << "[";
    visit(*this, *e.address());
    os_ << "]";
}

void codegen_opencl::operator()(internal::call_builtin &fn) {
    os_ << fn.fn() << "(";
    do_with_infix(fn.args().begin(), fn.args().end(), [this](auto x) { visit(*this, *x); });
    os_ << ")";
}

void codegen_opencl::operator()(internal::call_external &fn) {
    os_ << fn.name() << "(";
    do_with_infix(fn.args().begin(), fn.args().end(), [this](auto x) { visit(*this, *x); });
    os_ << ")";
}

void codegen_opencl::operator()(internal::cast &c) {
    os_ << "(";
    visit(*this, *c.target_ty());
    os_ << ") ";
    visit_check_parentheses(c, *c.term(), true);
}

void codegen_opencl::operator()(internal::swizzle &s) {
    constexpr std::array<char, 4> names = {'x', 'y', 'z', 'w'};
    visit_check_parentheses(s, *s.term(), false);
    os_ << ".";
    switch (s.selector()) {
    case internal::swizzle_selector::index:
        if (s.indices().size() <= 4) {
            for (auto index : s.indices()) {
                os_ << names.at(index);
            }
        } else {
            os_ << 's' << std::hex;
            for (auto index : s.indices()) {
                os_ << index;
            }
            os_ << std::dec;
        }
        break;
    case internal::swizzle_selector::lo:
        os_ << "lo";
        break;
    case internal::swizzle_selector::hi:
        os_ << "hi";
        break;
    case internal::swizzle_selector::even:
        os_ << "even";
        break;
    case internal::swizzle_selector::odd:
        os_ << "odd";
        break;
    };
}

/* Stmt nodes */
void codegen_opencl::operator()(internal::declaration &d) {
    print_declaration(d);
    end_statement();
}

void codegen_opencl::operator()(internal::declaration_assignment &d) {
    print_declaration(d.decl());
    os_ << " = ";
    visit(*this, *d.rhs());
    end_statement();
}

void codegen_opencl::operator()(internal::expression_statement &e) {
    visit(*this, *e.term());
    end_statement();
}

void codegen_opencl::operator()(internal::block &b) {
    os_ << "{" << std::endl;
    ++lvl_;
    auto block_endl_bak = block_endl_;
    block_endl_ = true;
    for (auto &s : b.stmts()) {
        os_ << indent();
        visit(*this, *s);
    }
    block_endl_ = block_endl_bak;
    --lvl_;
    os_ << indent() << "}";
    if (block_endl_) {
        os_ << std::endl;
    }
}

void codegen_opencl::operator()(internal::for_loop &loop) {
    for (auto &a : loop.attributes()) {
        visit(*this, *a);
        os_ << std::endl << indent();
    }
    inline_ = true;
    os_ << "for (";
    visit(*this, *loop.start());
    visit(*this, *loop.condition());
    os_ << "; ";
    visit(*this, *loop.step());
    os_ << ") ";
    inline_ = false;
    visit(*this, *loop.body());
}

void codegen_opencl::operator()(internal::if_selection &is) {
    os_ << "if (";
    visit(*this, *is.condition());
    os_ << ") ";
    if (is.otherwise()) {
        block_endl_ = false;
        visit(*this, *is.then());
        block_endl_ = true;
        os_ << " else ";
        visit(*this, **is.otherwise());
    } else {
        visit(*this, *is.then());
    }
}

/* Kernel nodes */
void codegen_opencl::operator()(internal::prototype &proto) {
    if (test(proto.qualifiers())) {
        os_ << proto.qualifiers() << std::endl;
    }
    for (auto &a : proto.attributes()) {
        visit(*this, *a);
        os_ << std::endl << indent();
    }
    os_ << "void " << proto.name() << "(";
    do_with_infix(proto.args().begin(), proto.args().end(), [this](auto x) {
        visit(*this, *x.first);
        os_ << " ";
        visit(*this, *x.second);
        os_ << post_.str();
        post_.str(std::string());
    });
    os_ << ") ";
}

void codegen_opencl::operator()(internal::function &fn) {
    visit(*this, *fn.prototype());
    visit(*this, *fn.body());
}

/* Helper functions */
void codegen_opencl::visit_check_parentheses(internal::expr_node &op, internal::expr_node &term,
                                             bool is_term_right) {
    auto p_op = op.precedence();
    auto a_op = op.assoc();
    auto p_term = term.precedence();
    if (p_term > p_op ||
        (p_term == p_op && ((a_op == associativity::left_to_right && is_term_right) ||
                            (a_op == associativity::right_to_left && !is_term_right)))) {
        os_ << "(";
        visit(*this, term);
        os_ << ")";
    } else {
        visit(*this, term);
    }
}

std::string codegen_opencl::indent() const { return std::string(4 * lvl_, ' '); }
void codegen_opencl::int_suffix(short bits) {
    if (bits > 32) {
        os_ << "ll";
    }
}

void codegen_opencl::end_statement() {
    os_ << ";";
    if (inline_) {
        os_ << " ";
    } else {
        os_ << std::endl;
    }
}

void codegen_opencl::print_declaration(internal::declaration &d) {
    visit(*this, *d.ty());
    os_ << " ";
    visit(*this, *d.variable());
    os_ << post_.str();
    post_.str(std::string());
    for (auto &a : d.attributes()) {
        os_ << " ";
        visit(*this, *a);
    }
}

void generate_opencl(std::ostream &os, func k) { visit(codegen_opencl(os), *k); }
void generate_opencl(std::ostream &os, stmt s) { visit(codegen_opencl(os), *s); }
void generate_opencl(std::ostream &os, expr e) { visit(codegen_opencl(os), *e); }
void generate_opencl(std::ostream &os, data_type d) { visit(codegen_opencl(os), *d); }

} // namespace clir
