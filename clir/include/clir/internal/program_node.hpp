// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PROGRAM_NODE_20230309_HPP
#define PROGRAM_NODE_20230309_HPP

#include "clir/export.hpp"
#include "clir/func.hpp"
#include "clir/virtual_type_list.hpp"

#include <vector>

namespace clir::internal {

class CLIR_EXPORT program_node : public virtual_type_list<class program> {};

class CLIR_EXPORT program : public visitable<program, program_node> {
  public:
    program(std::vector<func> decls) : decls_(std::move(decls)) {}
    std::vector<func> &declarations() { return decls_; }

  private:
    std::vector<func> decls_;
};

} // namespace clir::internal

#endif // PROGRAM_NODE_20230309_HPP
