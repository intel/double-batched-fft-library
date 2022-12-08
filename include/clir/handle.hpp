// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HANDLE_20220405_HPP
#define HANDLE_20220405_HPP

#include <cstddef>
#include <memory>
#include <utility>

namespace clir {

template <typename T> class handle {
  public:
    handle() : node_(nullptr) {}
    handle(std::nullptr_t) : node_(nullptr) {}
    handle(std::shared_ptr<T> node) : node_(std::move(node)) {}

    T &operator*() { return *node_; }
    T *operator->() { return node_.get(); }
    T *get() { return node_.get(); }
    explicit operator bool() const noexcept { return bool(node_); }

  private:
    std::shared_ptr<T> node_;
};

} // namespace clir

#endif // HANDLE_20220405_HPP
