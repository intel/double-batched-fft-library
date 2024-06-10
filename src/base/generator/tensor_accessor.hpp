// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TENSOR_ACCESSOR_20230718_HPP
#define TENSOR_ACCESSOR_20230718_HPP

#include "generator/utility.hpp"

#include "clir/builder.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"

#include <utility>

namespace bbfft {

class tensor_accessor {
  public:
    virtual ~tensor_accessor() {}
    virtual clir::expr operator()(clir::expr const &offset) const = 0;
    virtual clir::expr store(clir::expr value, clir::expr const &offset) const = 0;
    virtual auto subview(clir::block_builder &bb, clir::expr const &offset) const
        -> std::shared_ptr<tensor_accessor> = 0;
};

class zero_accessor : public tensor_accessor {
  public:
    zero_accessor(precision fp);
    clir::expr operator()(clir::expr const &) const override;
    clir::expr store(clir::expr, clir::expr const &) const override;
    auto subview(clir::block_builder &, clir::expr const &) const
        -> std::shared_ptr<tensor_accessor> override;

  private:
    precision_helper fph_;
};

class array_accessor : public tensor_accessor {
  public:
    array_accessor(clir::expr x, clir::data_type type, int component = -1);

    clir::expr operator()(clir::expr const &offset) const override;
    clir::expr store(clir::expr value, clir::expr const &offset) const override;
    auto subview(clir::block_builder &bb, clir::expr const &offset) const
        -> std::shared_ptr<tensor_accessor> override;

    inline int component() const { return component_; }
    inline void component(int c) { component_ = c; }

  private:
    clir::expr x_;
    clir::data_type type_;
    int component_;
};

class callback_accessor : public tensor_accessor {
  public:
    callback_accessor(clir::expr x, clir::data_type type, char const *load, char const *store,
                      clir::expr user_data, clir::expr offset = 0);

    clir::expr operator()(clir::expr const &offset) const override;
    clir::expr store(clir::expr value, clir::expr const &offset) const override;
    auto subview(clir::block_builder &bb, clir::expr const &offset) const
        -> std::shared_ptr<tensor_accessor> override;

  private:
    clir::expr x_;
    clir::data_type type_;
    char const *load_;
    char const *store_;
    clir::expr user_data_;
    clir::expr offset_;
};

} // namespace bbfft

#endif // TENSOR_ACCESSOR_20230718_HPP
