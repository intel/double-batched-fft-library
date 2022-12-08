// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TENSOR20220210_H
#define TENSOR20220210_H

#include "allocator.hpp"
#include "bbfft/tensor_indexer.hpp"

#include <array>
#include <memory>
#include <type_traits>
#include <utility>

template <typename T, unsigned int D, bbfft::layout L = bbfft::layout::row_major> class tensor {
  public:
    using value_t = T;
    using idx_t = unsigned int;
    using multi_idx_t = typename bbfft::tensor_indexer<idx_t, D, L>::multi_idx_t;
    tensor(T *data, multi_idx_t shape) : data_(data), indexer_(std::move(shape)) {}
    tensor(T *data, multi_idx_t shape, multi_idx_t stride)
        : data_(data), indexer_(std::move(shape), std::move(stride)) {}

    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == D, int>>
    T &operator()(Indices &&...is) const {
        return data_[indexer_(std::forward<Indices>(is)...)];
    }

    auto shape() const { return indexer_.shape(); }
    auto shape(unsigned int d) const { return indexer_.shape(d); }
    auto stride() const { return indexer_.stride(); }
    auto stride(unsigned int d) const { return indexer_.stride(d); }
    idx_t size() const { return indexer_.size(); }

    auto data() const { return data_; }

    tensor<T, D> make_ref() { return tensor<T, D>(data_, shape(), stride()); }

  protected:
    T *data_;

  private:
    bbfft::tensor_indexer<idx_t, D, L> indexer_;
    multi_idx_t shape_, stride_;
};

template <typename T, unsigned int D, bbfft::layout L = bbfft::layout::row_major>
class managed_tensor : public tensor<T, D, L> {
  public:
    using multi_idx_t = typename tensor<T, D, L>::multi_idx_t;
    managed_tensor(std::shared_ptr<allocator> alloc, multi_idx_t shape)
        : tensor<T, D, L>(nullptr, shape),
          managed_data_(make_storage(std::move(alloc), this->size())) {
        this->data_ = managed_data_.get();
    }
    managed_tensor(std::shared_ptr<allocator> alloc, multi_idx_t shape, multi_idx_t stride)
        : tensor<T, D, L>(nullptr, shape, stride),
          managed_data_(make_storage(std::move(alloc), this->size())) {
        this->data_ = managed_data_.get();
    }

  private:
    struct Deleter {
        std::shared_ptr<allocator> alloc;
        Deleter(std::shared_ptr<allocator> &&alloc) : alloc(std::move(alloc)) {}
        void operator()(T *ptr) { alloc->free(ptr); }
    };
    std::unique_ptr<T[], Deleter> make_storage(std::shared_ptr<allocator> &&alloc,
                                               std::size_t num_Ts) {
        T *ptr = malloc_shared<T>(num_Ts, *alloc);
        return std::unique_ptr<T[], Deleter>(ptr, Deleter(std::move(alloc)));
    }

    std::unique_ptr<T[], Deleter> managed_data_;
};

template <typename T, bbfft::layout L = bbfft::layout::row_major> using vector = tensor<T, 1u, L>;
template <typename T, bbfft::layout L = bbfft::layout::row_major>
using managed_vector = managed_tensor<T, 1u, L>;

template <typename T, bbfft::layout L = bbfft::layout::row_major> using matrix = tensor<T, 2u, L>;
template <typename T, bbfft::layout L = bbfft::layout::row_major>
using managed_matrix = managed_tensor<T, 2u, L>;

#endif // TENSOR20220210_H
