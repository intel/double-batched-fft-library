// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VIRTUAL_TYPE_LIST_20220405_HPP
#define VIRTUAL_TYPE_LIST_20220405_HPP

#include <cstddef>
#include <type_traits>

namespace clir::internal {

template <typename Type, typename Head, typename... Tail>
constexpr std::size_t type_index(std::size_t index = 0) {
    if constexpr (std::is_same_v<Type, Head>) {
        return index;
    } else {
        return type_index<Type, Tail...>(index + 1);
    }
}

template <std::size_t I, typename... T> struct type_at;

template <std::size_t I, typename Head, typename... Tail> struct type_at<I, Head, Tail...> {
    using type = typename type_at<I - 1, Tail...>::type;
};

template <typename Head, typename... Tail> struct type_at<0, Head, Tail...> { using type = Head; };

template <std::size_t I, typename... T> using type_at_t = typename type_at<I, T...>::type;

template <std::size_t I, typename... VTLs> struct stride_at;

template <std::size_t I, typename Head, typename... Tail> struct stride_at<I, Head, Tail...> {
    static constexpr std::size_t value = Head::number_of_types() * stride_at<I - 1, Tail...>::value;
};

template <typename Head, typename... Tail> struct stride_at<0, Head, Tail...> {
    static constexpr std::size_t value = 1;
};

template <std::size_t I, typename... T>
inline constexpr std::size_t stride_at_v = stride_at<I, T...>::value;

constexpr std::size_t flatten() { return 0; }

template <typename Head, typename... Tail>
constexpr std::size_t flatten(Head &&head, Tail &&...tail) {
    return head.type_index() + flatten(tail...) * std::decay_t<Head>::number_of_types();
}

template <std::size_t Index, std::size_t Mode, typename... VTLs> constexpr auto unflatten() {
    return (Index / stride_at_v<Mode, VTLs...>) %
           std::decay_t<type_at_t<Mode, VTLs...>>::number_of_types();
}

} // namespace clir::internal

namespace clir {

template <typename... Ts> class virtual_type_list {
  public:
    template <std::size_t I, std::enable_if_t<I<sizeof...(Ts), int> = 0> using type_at =
                                 internal::type_at_t<I, Ts...>;

    virtual ~virtual_type_list() {}

    virtual std::size_t type_index() const noexcept = 0;
    static constexpr auto number_of_types() { return sizeof...(Ts); }

  protected:
    template <typename T,
              std::enable_if_t<internal::type_index<T, Ts...>() < sizeof...(Ts), int> = 0>
    static constexpr std::size_t compute_type_index(T const *) {
        return internal::type_index<T, Ts...>();
    }
};

template <typename Derived, typename VirtualTypeList> class visitable : public VirtualTypeList {
  public:
    using VirtualTypeList::VirtualTypeList;

    std::size_t type_index() const noexcept override {
        return VirtualTypeList::compute_type_index(static_cast<Derived const *>(this));
    }
};

} // namespace clir

#endif // VIRTUAL_TYPE_LIST_20220405_HPP
