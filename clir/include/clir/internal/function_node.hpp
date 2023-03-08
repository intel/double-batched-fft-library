// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FUNCTION_NODE_20220405_HPP
#define FUNCTION_NODE_20220405_HPP

#include "clir/attr.hpp"
#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/func.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/virtual_type_list.hpp"

#include <string>
#include <utility>
#include <vector>

namespace clir::internal {

class CLIR_EXPORT function_node : public virtual_type_list<class prototype, class function> {};

class CLIR_EXPORT prototype : public visitable<prototype, function_node> {
  public:
    prototype(std::string name, std::vector<std::pair<data_type, var>> args = {},
              function_qualifier qualifiers = function_qualifier::none,
              std::vector<attr> attributes = {})
        : name_(std::move(name)), args_(std::move(args)), qualifiers_(qualifiers),
          attributes_(std::move(attributes)) {}

    std::string_view name() const { return name_; }
    std::vector<std::pair<data_type, var>> &args() { return args_; }
    function_qualifier qualifiers() { return qualifiers_; }
    void qualifiers(function_qualifier q) { qualifiers_ = q; }
    std::vector<attr> &attributes() { return attributes_; }

  private:
    std::string name_;
    std::vector<std::pair<data_type, var>> args_;
    function_qualifier qualifiers_;
    std::vector<attr> attributes_;
};

class CLIR_EXPORT function : public visitable<function, function_node> {
  public:
    function(func prototype, stmt body)
        : prototype_(std::move(prototype)), body_(std::move(body)) {}

    func &prototype() { return prototype_; }
    stmt &body() { return body_; }

  private:
    func prototype_;
    stmt body_;
};

} // namespace clir::internal

#endif // FUNCTION_NODE_20220405_HPP
