// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "build_wrapper.hpp"

#include "bbfft/sycl/online_compiler.hpp"

#include <CL/sycl.hpp>
#include <cstdio>
#include <stdexcept>
#include <utility>

using ::sycl::backend;
using ::sycl::bundle_state;
using ::sycl::context;
using ::sycl::device;
using ::sycl::kernel;
using ::sycl::kernel_bundle;

namespace bbfft::sycl {

template <backend B> struct as_type {
    constexpr static backend value = B;
};
template <typename... Ts> struct type_list {};
template <backend... Bs> struct make_type_list {
    using type = type_list<as_type<Bs>...>;
};

using supported_backends = make_type_list<backend::ext_oneapi_level_zero, backend::opencl>::type;

template <typename Fun, typename Head> auto dispatch_impl(backend, Fun &&f, Head) {
    return f(Head{});
}
template <typename Fun, typename Head, typename... Tail>
auto dispatch_impl(backend b, Fun &&f, Head, Tail...) {
    if (b == Head::value) {
        return f(Head{});
    }
    return dispatch_impl(b, std::forward<Fun>(f), Tail{}...);
}

template <typename Fun, template <typename...> class TL, typename... Bs>
auto dispatch(backend b, Fun &&f, TL<Bs...>) {
    return dispatch_impl(b, std::forward<Fun>(f), Bs{}...);
}

auto build_native_module(std::string const &source, context c, device d,
                         std::vector<std::string> const &options,
                         std::vector<std::string> const &extensions) -> module_handle_t {
    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>(c, d).build_module(source, options, extensions);
    };
    return dispatch(c.get_backend(), f, supported_backends{});
}

auto build_native_module(uint8_t const *binary, std::size_t binary_size, module_format format,
                         context c, device d) -> module_handle_t {
    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>(c, d).build_module(binary, binary_size, format);
    };
    return dispatch(c.get_backend(), f, supported_backends{});
}

auto make_shared_handle(module_handle_t mod, ::sycl::backend be) -> shared_handle<module_handle_t> {
    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>::make_shared_handle(mod);
    };
    return dispatch(be, f, supported_backends{});
}

auto make_kernel_bundle(module_handle_t mod, bool keep_ownership, context c)
    -> kernel_bundle<bundle_state::executable> {
    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>::make_kernel_bundle(mod, keep_ownership,
                                                                     std::move(c));
    };
    return dispatch(c.get_backend(), f, supported_backends{});
}

auto create_kernel(kernel_bundle<bundle_state::executable> bundle, std::string const &name)
    -> kernel {
    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>::create_kernel(std::move(bundle), name);
    };
    return dispatch(bundle.get_backend(), f, supported_backends{});
}

aot_module create_aot_module(uint8_t const *binary, std::size_t binary_size, module_format format,
                             context c, device d) {

    auto const f = [&](auto b) {
        return build_wrapper<decltype(b)::value>(c, d).create_aot_module(binary, binary_size,
                                                                         format);
    };
    return dispatch(c.get_backend(), f, supported_backends{});
}

} // namespace bbfft::sycl
