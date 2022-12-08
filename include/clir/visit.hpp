// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VISIT_20220405_HPP
#define VISIT_20220405_HPP

#include "clir/virtual_type_list.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace clir::internal {

template <typename U, typename V> struct copy_cv_qualifiers {
  private:
    using N = std::remove_reference_t<U>;
    using V1 = std::conditional_t<std::is_const<N>::value, std::add_const_t<V>, V>;
    using V2 = std::conditional_t<std::is_volatile<N>::value, std::add_volatile_t<V1>, V1>;
    using V3 =
        std::conditional_t<std::is_lvalue_reference<U>::value, std::add_lvalue_reference_t<V2>, V2>;
    using V4 =
        std::conditional_t<std::is_rvalue_reference<U>::value, std::add_rvalue_reference_t<V3>, V3>;

  public:
    using type = V4;
};
template <typename Source, typename Target>
using copy_cv_qualifiers_t = typename copy_cv_qualifiers<Source, Target>::type;

template <std::size_t Index, typename VTL, typename... VTLs>
using dispatch_type_at = copy_cv_qualifiers_t<
    VTL, typename VTL::template type_at<unflatten<Index, type_index<VTL, VTLs...>(), VTLs...>()>>;

template <typename Functional, std::size_t... Is>
auto compile_time_switch(Functional f, std::size_t i, std::index_sequence<Is...>) {
    using return_type =
        std::common_type_t<decltype(f(std::integral_constant<std::size_t, Is>{}))...>;
    if constexpr (std::is_same_v<return_type, void>) {
        [[maybe_unused]] bool discard =
            ((i == Is ? f(std::integral_constant<std::size_t, Is>{}), false : false) || ...);
        return;
    } else {
        return_type ret = {};
        [[maybe_unused]] bool discard =
            ((i == Is ? (ret = f(std::integral_constant<std::size_t, Is>{})), false : false) ||
             ...);
        return ret;
    }
}

} // namespace clir::internal

namespace clir {

template <typename Visitor, typename... VTLs> auto visit(Visitor &&visitor, VTLs &...ts) {
    constexpr std::size_t table_size = (std::decay_t<VTLs>::number_of_types() * ...);
    return internal::compile_time_switch(
        [&](auto index) -> auto {
            return visitor(static_cast<internal::copy_cv_qualifiers_t<
                               VTLs, internal::dispatch_type_at<index, VTLs, VTLs...>> &>(ts)...);
        },
        internal::flatten(ts...), std::make_index_sequence<table_size>{});
}

} // namespace clir

#endif // VISIT_20220405_HPP
