// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ARGS_IMPL_20220405_HPP
#define ARGS_IMPL_20220405_HPP

#include <array>
#include <optional>
#include <ostream>
#include <utility>

namespace clir::internal {

template <typename T> class args_impl {
  public:
    args_impl(T arg) : arg_(std::move(arg)) {}
    auto const &arg() const { return arg_; }

  private:
    T arg_;
};
template <typename T> std::ostream &operator<<(std::ostream &os, args_impl<T> const &args) {
    return os << "(" << args.arg() << ")";
}

template <> class args_impl<void> {};
template <typename T> std::ostream &operator<<(std::ostream &os, args_impl<void> const &) {
    return os;
}

template <typename T> class args_impl<std::optional<T>> {
  public:
    args_impl() : arg_(std::nullopt) {}
    args_impl(T arg) : arg_(std::move(arg)) {}
    auto const &arg() const { return arg_; }

  private:
    std::optional<T> arg_;
};
template <typename T>
std::ostream &operator<<(std::ostream &os, args_impl<std::optional<T>> const &args) {
    if (args.arg()) {
        os << "(" << *args.arg() << ")";
    }
    return os;
}

template <typename T, std::size_t N> class args_impl<std::array<T, N>> {
  public:
    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == N, int>>
    args_impl(Args... arg) : args_{std::move(arg)...} {}
    auto cbegin() const { return args_.cbegin(); }
    auto cend() const { return args_.cend(); }

  private:
    std::array<T, N> args_;
};
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, args_impl<std::array<T, N>> const &args) {
    os << "(";
    auto begin = args.cbegin();
    for (auto it = begin; it != args.cend(); ++it) {
        if (it != begin) {
            os << ",";
        }
        os << *it;
    }
    os << ")";
    return os;
}

} // namespace clir::internal

#endif // ARGS_IMPL_20220405_HPP
