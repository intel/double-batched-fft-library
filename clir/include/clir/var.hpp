// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VAR_20220405_HPP
#define VAR_20220405_HPP

#include "clir/export.hpp"
#include "clir/expr.hpp"
#include "clir/handle.hpp"

#include <string>
#include <utility>

namespace clir {

class CLIR_EXPORT var {
  public:
    var(std::string prefix = "");

    operator expr &() { return e_; }

    template <typename T> expr operator[](T a) const { return e_[std::move(a)]; }
    template <typename... Is> expr s(Is... is) const { return e_.s(std::move(is)...); }
    inline expr lo() const { return e_.lo(); }
    inline expr hi() const { return e_.hi(); }
    inline expr even() const { return e_.even(); }
    inline expr odd() const { return e_.odd(); }

    inline auto &operator*() { return *e_; }
    inline auto *operator->() { return e_.get(); }
    inline auto *get() { return e_.get(); }

  private:
    expr e_;
};

} // namespace clir

#endif // VAR_20220405_HPP
