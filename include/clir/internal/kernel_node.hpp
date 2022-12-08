// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef KERNEL_NODE_20220405_HPP
#define KERNEL_NODE_20220405_HPP

#include "clir/attr.hpp"
#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/kernel.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"
#include "clir/virtual_type_list.hpp"

#include <string>
#include <utility>
#include <vector>

namespace clir::internal {

class CLIR_EXPORT kernel_node : public virtual_type_list<class prototype, class function> {};

class CLIR_EXPORT prototype : public visitable<prototype, kernel_node> {
  public:
    prototype(std::string name, std::vector<std::pair<data_type, var>> args = {},
              std::vector<attr> attributes = {})
        : name_(std::move(name)), args_(std::move(args)), attributes_(std::move(attributes)) {}

    std::string_view name() const { return name_; }
    std::vector<std::pair<data_type, var>> &args() { return args_; }
    std::vector<attr> &attributes() { return attributes_; }

  private:
    std::string name_;
    std::vector<std::pair<data_type, var>> args_;
    std::vector<attr> attributes_;
};

class CLIR_EXPORT function : public visitable<function, kernel_node> {
  public:
    function(kernel prototype, stmt body)
        : prototype_(std::move(prototype)), body_(std::move(body)) {}

    kernel &prototype() { return prototype_; }
    stmt &body() { return body_; }

  private:
    kernel prototype_;
    stmt body_;
};

} // namespace clir::internal

#endif // KERNEL_NODE_20220405_HPP
