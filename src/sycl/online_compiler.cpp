// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/sycl/online_compiler.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/cl/online_compiler.hpp"
#include "bbfft/ze/online_compiler.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

using ::sycl::backend;
using ::sycl::backend_input_t;
using ::sycl::bundle_state;
using ::sycl::context;
using ::sycl::device;
using ::sycl::get_native;
using ::sycl::kernel;
using ::sycl::kernel_bundle;
using ::sycl::make_kernel;
using ::sycl::make_kernel_bundle;

namespace bbfft::sycl {

template <backend B> struct build_wrapper;
template <> struct build_wrapper<backend::ext_oneapi_level_zero> {
    template <typename... Args>
    static auto build_kernel_bundle(context c, device d, Args &&...args) {
        auto native_context = get_native<backend::ext_oneapi_level_zero, context>(c);
        auto native_device = get_native<backend::ext_oneapi_level_zero, device>(d);
        auto native_module =
            ze::build_kernel_bundle(std::forward<Args>(args)..., native_context, native_device);
        return make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>(
            {native_module, ::sycl::ext::oneapi::level_zero::ownership::transfer}, c);
    }
    static auto create_kernel(kernel_bundle<bundle_state::executable> bundle,
                              std::string const &name) {
        auto native_bundle =
            get_native<backend::ext_oneapi_level_zero, bundle_state::executable>(bundle);
        auto native_kernel = ze::create_kernel(native_bundle.front(), name);
        return make_kernel<backend::ext_oneapi_level_zero>(
            {bundle, native_kernel, ::sycl::ext::oneapi::level_zero::ownership::transfer},
            bundle.get_context());
    }
    static auto get_native_binary(kernel_bundle<bundle_state::executable> bundle) {
        auto native_bundle =
            get_native<backend::ext_oneapi_level_zero, bundle_state::executable>(bundle);
        return ze::get_native_binary(native_bundle.front());
    }
};
template <> struct build_wrapper<backend::opencl> {
    template <typename... Args>
    static auto build_kernel_bundle(context c, device d, Args &&...args) {
        auto native_context = get_native<backend::opencl, context>(c);
        auto native_device = get_native<backend::opencl, device>(d);
        auto native_module =
            cl::build_kernel_bundle(std::forward<Args>(args)..., native_context, native_device);
        cl::get_native_binary(native_module, native_device);
        CL_CHECK(clReleaseContext(native_context));
        CL_CHECK(clReleaseDevice(native_device));
        auto bundle =
            make_kernel_bundle<backend::opencl, bundle_state::executable>(native_module, c);
        // TODO: Should be necessary according to spec but runtime does not call retain
        // CL_CHECK(clReleaseProgram(native_module));
        return bundle;
    }
    static auto create_kernel(kernel_bundle<bundle_state::executable> bundle,
                              std::string const &name) {
        auto native_bundle = get_native<backend::opencl, bundle_state::executable>(bundle);
        auto native_kernel = cl::create_kernel(native_bundle.front(), name);
        // TODO: Should be necessary according to spec but runtime does not call retain
        // for (auto& nm : native_bundle) {
        // CL_CHECK(clReleaseProgram(nm));
        // }
        auto k = make_kernel<backend::opencl>(native_kernel, bundle.get_context());
        CL_CHECK(clReleaseKernel(native_kernel));
        return k;
    }
    static auto get_native_binary(kernel_bundle<bundle_state::executable> bundle, device d) {
        auto native_device = get_native<backend::opencl, device>(d);
        auto native_bundle = get_native<backend::opencl, bundle_state::executable>(bundle);
        auto bin = cl::get_native_binary(native_bundle.front(), native_device);
        CL_CHECK(clReleaseDevice(native_device));
        // TODO: Should be necessary according to spec but runtime does not call retain
        // for (auto& nm : native_bundle) {
        // CL_CHECK(clReleaseProgram(nm));
        // }
        return bin;
    }
};

auto build_kernel_bundle(std::string const &source, context c, device d)
    -> kernel_bundle<bundle_state::executable> {
    if (c.get_backend() == backend::ext_oneapi_level_zero) {
        return build_wrapper<backend::ext_oneapi_level_zero>::build_kernel_bundle(
            std::move(c), std::move(d), source);
    }
    return build_wrapper<backend::opencl>::build_kernel_bundle(std::move(c), std::move(d), source);
}

auto build_kernel_bundle(uint8_t const *binary, std::size_t binary_size, context c, device d)
    -> kernel_bundle<bundle_state::executable> {
    if (c.get_backend() == backend::ext_oneapi_level_zero) {
        return build_wrapper<backend::ext_oneapi_level_zero>::build_kernel_bundle(
            std::move(c), std::move(d), binary, binary_size);
    }
    return build_wrapper<backend::opencl>::build_kernel_bundle(std::move(c), std::move(d), binary,
                                                               binary_size);
}

auto create_kernel(kernel_bundle<bundle_state::executable> bundle, std::string const &name)
    -> kernel {
    if (bundle.get_backend() == backend::ext_oneapi_level_zero) {
        return build_wrapper<backend::ext_oneapi_level_zero>::create_kernel(std::move(bundle),
                                                                            name);
    }
    return build_wrapper<backend::opencl>::create_kernel(std::move(bundle), name);
}

std::vector<uint8_t> get_native_binary(kernel_bundle<bundle_state::executable> bundle,
                                       device device) {
    if (bundle.get_backend() == backend::ext_oneapi_level_zero) {
        return build_wrapper<backend::ext_oneapi_level_zero>::get_native_binary(std::move(bundle));
    }
    return build_wrapper<backend::opencl>::get_native_binary(std::move(bundle), std::move(device));
}

} // namespace bbfft::sycl
