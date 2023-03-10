// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CODEGEN_OPENCL_20220405_HPP
#define CODEGEN_OPENCL_20220405_HPP

#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/internal/attr_node.hpp"
#include "clir/internal/data_type_node.hpp"
#include "clir/internal/expr_node.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/internal/program_node.hpp"
#include "clir/internal/stmt_node.hpp"

#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

namespace clir {

class CLIR_EXPORT data_type;
class CLIR_EXPORT func;
class CLIR_EXPORT stmt;
class CLIR_EXPORT prog;

class CLIR_EXPORT codegen_opencl {
  public:
    codegen_opencl(std::ostream &os);
    ~codegen_opencl();

    /* Attributes */
    void operator()(internal::attr_node &attr);

    /* Data type nodes */
    void operator()(internal::scalar_data_type &v);
    void operator()(internal::vector_data_type &v);
    void operator()(internal::pointer &v);
    void operator()(internal::array &a);

    /* Expr nodes */
    void operator()(internal::variable &v);
    void operator()(internal::int_imm &v);
    void operator()(internal::uint_imm &v);
    void operator()(internal::float_imm &v);
    void operator()(internal::cl_mem_fence_flags_imm &v);
    void operator()(internal::string_imm &v);
    void operator()(internal::unary_op &e);
    void operator()(internal::binary_op &e);
    void operator()(internal::ternary_op &e);
    void operator()(internal::access &e);
    void operator()(internal::call_builtin &fn);
    void operator()(internal::call &fn);
    void operator()(internal::cast &c);
    void operator()(internal::swizzle &s);

    /* Stmt nodes */
    void operator()(internal::declaration &d);
    void operator()(internal::declaration_assignment &d);
    void operator()(internal::expression_statement &e);
    void operator()(internal::block &b);
    void operator()(internal::for_loop &loop);
    void operator()(internal::if_selection &is);

    /* Kernel nodes */
    void operator()(internal::prototype &proto);
    void operator()(internal::function &fn);
    void operator()(internal::global_declaration &d);

    /* Program nodes */
    void operator()(internal::program &prg);

  private:
    template <typename Iterator, typename Action>
    CLIR_NO_EXPORT void do_with_infix(Iterator begin, Iterator end, Action a) {
        for (auto it = begin; it != end; ++it) {
            if (it != begin) {
                os_ << ", ";
            }
            a(*it);
        }
    }
    CLIR_NO_EXPORT void visit_check_parentheses(internal::expr_node &op, internal::expr_node &term,
                                                bool is_term_right);
    CLIR_NO_EXPORT std::string indent() const;
    CLIR_NO_EXPORT void int_suffix(short bits);
    CLIR_NO_EXPORT void end_statement();
    CLIR_NO_EXPORT void print_declaration(internal::declaration &d);

    int lvl_ = 0;
    std::ostream &os_;
    std::ios_base::fmtflags stream_fmt_;
    std::stringstream post_;
    bool inline_ = false;
    bool block_endl_ = true;
    bool definition_ = false;
};

CLIR_EXPORT void generate_opencl(std::ostream &os, prog p);
CLIR_EXPORT void generate_opencl(std::ostream &os, func k);
CLIR_EXPORT void generate_opencl(std::ostream &os, stmt s);
CLIR_EXPORT void generate_opencl(std::ostream &os, expr e);
CLIR_EXPORT void generate_opencl(std::ostream &os, data_type d);

} // namespace clir

#endif // CODEGEN_OPENCL_20220405_HPP
