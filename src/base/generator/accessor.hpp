// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACCESSOR_20220520_HPP
#define ACCESSOR_20220520_HPP

#include "bbfft/configuration.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "clir/expr.hpp"
#include "generator/utility.hpp"

#include <array>
#include <functional>
#include <utility>

using clir::call;
using clir::expr;

namespace bbfft {

template <unsigned int D> class accessor {
  public:
    virtual ~accessor() {}
    virtual expr operator()(std::array<expr, D> const &idx) const = 0;
    virtual expr store(std::array<expr, D> const &idx, expr value) const = 0;

    template <typename... Is> expr operator()(Is... is) const {
        return operator()({std::move(is)...});
    }
};

template <unsigned int D> class zero_accessor : public accessor<D> {
  public:
    zero_accessor(precision fp) : fph_(fp) {}
    expr operator()(std::array<expr, D> const &) const override { return fph_.zero(); }
    expr store(std::array<expr, D> const &, expr) const override { return nullptr; }

  private:
    precision_helper fph_;
};

template <unsigned int D> class array_accessor : public accessor<D> {
  public:
    array_accessor(expr x, std::array<expr, D> shape)
        : x_(std::move(x)), indexer_(std::move(shape)) {}
    array_accessor(expr x, std::array<expr, D> shape, std::array<expr, D> stride)
        : x_(std::move(x)), indexer_(std::move(shape), std::move(stride)) {}

    expr operator()(std::array<expr, D> const &idx) const override { return x_[indexer_(idx)]; }
    expr store(std::array<expr, D> const &idx, expr value) const override {
        return assignment(x_[indexer_(idx)], std::move(value));
    }

  protected:
    auto &indexer() const { return indexer_; }
    auto &x() const { return x_; }

  private:
    expr x_;
    tensor_indexer<expr, D, layout::col_major> indexer_;
};

template <unsigned int D> class callback_accessor : public accessor<D> {
  public:
    using mnk_function = std::function<expr(std::array<expr, D> const &idx)>;

    callback_accessor(expr x, std::array<expr, D> shape, std::array<expr, D> stride, expr offset,
                      mnk_function mnk, char const *load = nullptr, char const *store = nullptr)
        : x_(std::move(x)), indexer_(std::move(shape), std::move(stride)),
          offset_(std::move(offset)), mnk_(std::move(mnk)), load_(load), store_(store) {}

    expr operator()(std::array<expr, D> const &idx) const override {
        if (load_) {
            return call(load_, {x_, offset_ + indexer_(idx), mnk_(idx)});
        }
        return x_[offset_ + indexer_(idx)];
    }
    expr store(std::array<expr, D> const &idx, expr value) const override {
        if (store_) {
            return call(store_, {x_, offset_ + indexer_(idx), std::move(value), mnk_(idx)});
        }
        return assignment(x_[indexer_(idx)], std::move(value));
    }

  private:
    expr x_;
    tensor_indexer<expr, D, layout::col_major> indexer_;
    expr offset_;
    mnk_function mnk_;
    char const *load_;
    char const *store_;
};

} // namespace bbfft

#endif // ACCESSOR_20220520_HPP
