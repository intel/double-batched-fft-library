// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/visitor/to_imm.hpp"
#include "clir/visit.hpp"

namespace clir {

/* Expr nodes */
auto to_imm::operator()(internal::expr_node &) -> return_t { return {}; }
auto to_imm::operator()(internal::int_imm &i) -> return_t { return i.value(); }
auto to_imm::operator()(internal::uint_imm &i) -> return_t { return i.value(); }
auto to_imm::operator()(internal::float_imm &i) -> return_t { return i.value(); }

auto get_imm(expr e) -> to_imm::return_t { return visit(to_imm{}, *e); }

} // namespace clir
