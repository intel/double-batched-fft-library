// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEMPLATE_MAGIC_20230718_HPP
#define TEMPLATE_MAGIC_20230718_HPP

#include <utility>

namespace bbfft {

template <class T, T... As, T... Bs>
constexpr std::integer_sequence<T, As..., Bs...> operator+(std::integer_sequence<T, As...>,
                                                           std::integer_sequence<T, Bs...>) {
    return {};
}

template <std::size_t Index> constexpr auto enumerate_true() { return std::index_sequence<>{}; }

template <std::size_t Index, typename Head, typename... Tail>
constexpr auto enumerate_true(Head head, Tail... tail) {
    if constexpr (head) {
        return std::index_sequence<Index>{} + enumerate_true<Index + 1u, Tail...>(tail...);
    } else {
        return enumerate_true<Index + 1u, Tail...>(tail...);
    }
}

template <typename... Entry> constexpr auto enumerate_true(Entry... entry) {
    return enumerate_true<0u, Entry...>(entry...);
}

template <typename Container, std::size_t... Is>
auto select(Container const &a, std::index_sequence<Is...>)
    -> std::array<typename Container::value_type, sizeof...(Is)> {
    return {a[Is]...};
}

} // namespace bbfft

#endif // TEMPLATE_MAGIC_20230718_HPP
