// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TENSOR_VIEW_20230718_HPP
#define TENSOR_VIEW_20230718_HPP

#include "bbfft/tensor_indexer.hpp"
#include "template_magic.hpp"
#include "tensor_accessor.hpp"

#include "clir/builder.hpp"
#include "clir/expr.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

using clir::expr;

namespace bbfft {

struct slice {
    expr begin = nullptr;
    expr size = nullptr;
};

template <std::size_t D> class tensor_view {
  public:
    tensor_view(std::shared_ptr<tensor_accessor> accessor, std::array<expr, D> shape)
        : accessor_(std::move(accessor)), indexer_(std::move(shape)) {}
    tensor_view(std::shared_ptr<tensor_accessor> accessor, std::array<expr, D> shape,
                std::array<expr, D> stride)
        : accessor_(std::move(accessor)), indexer_(std::move(shape), std::move(stride)) {}
    tensor_view(std::shared_ptr<tensor_accessor> accessor,
                tensor_indexer<expr, D, layout::col_major> indexer)
        : accessor_(std::move(accessor)), indexer_(std::move(indexer)) {}
    virtual ~tensor_view() {}

    auto shape(unsigned int d) const { return indexer_.shape(d); }

    expr operator()(std::array<expr, D> const &idx) const { return (*accessor_)(indexer_(idx)); }
    template <typename... Entry, typename = std::enable_if_t<sizeof...(Entry) == D, int>>
    auto operator()(Entry &&...idx) const {
        return (*accessor_)(indexer_(std::forward<Entry>(idx)...));
    }

    expr store(expr value, std::array<expr, D> const &idx) const {
        return accessor_->store(std::move(value), indexer_(idx));
    }
    template <typename... Entry, typename = std::enable_if_t<sizeof...(Entry) == D, int>>
    auto store(expr value, Entry &&...idx) const {
        return store(std::move(value), {std::forward<Entry>(idx)...});
    }

    template <typename... Entry, typename = std::enable_if_t<sizeof...(Entry) == D, int>>
    auto subview(clir::block_builder &bb, Entry &&...entry) const {
        constexpr auto slice_positions = enumerate_true(std::is_same<slice, Entry>()...);
        auto const to_slice = [](auto &&entry) {
            if constexpr (std::is_same_v<slice, std::decay_t<decltype(entry)>>) {
                return entry;
            } else {
                return slice{entry, 1};
            }
        };
        auto region = std::array<slice, sizeof...(Entry)>{to_slice(entry)...};
        auto new_shape = std::array<expr, sizeof...(Entry)>{};
        expr offset = 0u;
        for (std::size_t i = 0; i < D; ++i) {
            if (bool(region[i].begin)) {
                offset = offset + region[i].begin * indexer_.stride(i);
            }

            if (!bool(region[i].size)) {
                region[i].size =
                    bool(region[i].begin) ? indexer_.shape(i) - region[i].begin : indexer_.shape(i);
            }
            new_shape[i] = region[i].size;
        }

        auto sub_shape = select(new_shape, slice_positions);
        auto sub_stride = select(indexer_.stride(), slice_positions);
        auto sub_accessor = accessor_->subview(bb, std::move(offset));
        return tensor_view<slice_positions.size()>(std::move(sub_accessor), std::move(sub_shape),
                                                   std::move(sub_stride));
    }

    template <std::size_t E>
    auto reshaped_mode(int mode, std::array<expr, E> const &mode_shape) const {
        return tensor_view<D + E - 1u>(accessor_, indexer_.reshaped_mode(mode, mode_shape));
    }

  private:
    std::shared_ptr<tensor_accessor> accessor_;
    tensor_indexer<expr, D, layout::col_major> indexer_;
};

} // namespace bbfft

#endif // TENSOR_VIEW_20230718_HPP
