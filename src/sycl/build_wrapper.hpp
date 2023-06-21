#ifndef BUILD_WRAPPER_20230206_HPP
#define BUILD_WRAPPER_20230206_HPP

#include "bbfft/aot_cache.hpp"
#include "bbfft/cl/error.hpp"
#include "bbfft/cl/online_compiler.hpp"
#include "bbfft/detail/cast.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"
#include "bbfft/ze/online_compiler.hpp"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

#include <cstdint>
#include <utility>

namespace bbfft::sycl {

template <::sycl::backend B> struct build_wrapper;
template <> struct build_wrapper<::sycl::backend::ext_oneapi_level_zero> {
    ze_context_handle_t native_context;
    ze_device_handle_t native_device;

    constexpr static auto be_t = ::sycl::backend::ext_oneapi_level_zero;
    constexpr static auto bstate_t = ::sycl::bundle_state::executable;
    using bundle_t = ::sycl::kernel_bundle<bstate_t>;

    build_wrapper(::sycl::context c, ::sycl::device d)
        : native_context(::sycl::get_native<be_t, ::sycl::context>(c)),
          native_device(::sycl::get_native<be_t, ::sycl::device>(d)) {}

    auto build_module(std::string const &source, std::vector<std::string> const &options)
        -> module_handle_t {
        return detail::cast<module_handle_t>(
            ze::build_kernel_bundle(source, native_context, native_device, options));
    }
    auto build_module(uint8_t const *binary, std::size_t binary_size, module_format format)
        -> module_handle_t {
        return detail::cast<module_handle_t>(
            ze::build_kernel_bundle(binary, binary_size, format, native_context, native_device));
    }
    template <typename... Args> auto create_aot_module(Args &&...args) -> aot_module {
        return ze::create_aot_module(std::forward<Args>(args)..., native_context, native_device);
    }
    static auto make_shared_handle(module_handle_t mod) -> shared_handle<module_handle_t> {
        return shared_handle<module_handle_t>(mod, [](module_handle_t mod) {
            zeModuleDestroy(detail::cast<ze_module_handle_t>(mod));
        });
    }
    static auto make_kernel_bundle(module_handle_t mod, bool keep_ownership, ::sycl::context c)
        -> bundle_t {
        auto own = keep_ownership ? ::sycl::ext::oneapi::level_zero::ownership::keep
                                  : ::sycl::ext::oneapi::level_zero::ownership::transfer;
        return ::sycl::make_kernel_bundle<be_t, bstate_t>(
            {detail::cast<ze_module_handle_t>(mod), own}, c);
    }
    static auto create_kernel(bundle_t bundle, std::string const &name) -> ::sycl::kernel {
        auto native_bundle = ::sycl::get_native<be_t, bstate_t>(bundle);
        auto native_kernel = ze::create_kernel(native_bundle.front(), name);
        return ::sycl::make_kernel<be_t>(
            {bundle, native_kernel, ::sycl::ext::oneapi::level_zero::ownership::transfer},
            bundle.get_context());
    }
};
template <> struct build_wrapper<::sycl::backend::opencl> {
    cl_context native_context;
    cl_device_id native_device;

    constexpr static auto be_t = ::sycl::backend::opencl;
    constexpr static auto bstate_t = ::sycl::bundle_state::executable;
    using bundle_t = ::sycl::kernel_bundle<bstate_t>;

    build_wrapper(::sycl::context c, ::sycl::device d)
        : native_context(::sycl::get_native<be_t, ::sycl::context>(c)),
          native_device(::sycl::get_native<be_t, ::sycl::device>(d)) {}
    ~build_wrapper() {
        CL_CHECK(clReleaseContext(native_context));
        CL_CHECK(clReleaseDevice(native_device));
    }

    auto build_module(std::string const &source, std::vector<std::string> const &options)
        -> module_handle_t {
        return detail::cast<module_handle_t>(
            cl::build_kernel_bundle(source, native_context, native_device, options));
    }
    auto build_module(uint8_t const *binary, std::size_t binary_size, module_format format)
        -> module_handle_t {
        return detail::cast<module_handle_t>(
            cl::build_kernel_bundle(binary, binary_size, format, native_context, native_device));
    }
    template <typename... Args> auto create_aot_module(Args &&...args) -> aot_module {
        return cl::create_aot_module(std::forward<Args>(args)..., native_context, native_device);
    }
    static auto make_shared_handle(module_handle_t mod) -> shared_handle<module_handle_t> {
        return shared_handle<module_handle_t>(
            mod, [](module_handle_t mod) { clReleaseProgram(detail::cast<cl_program>(mod)); });
    }
    static auto make_kernel_bundle(module_handle_t mod, bool keep_ownership, ::sycl::context c)
        -> bundle_t {
        auto native_module = detail::cast<cl_program>(mod);
        auto bundle = ::sycl::make_kernel_bundle<be_t, bstate_t>(native_module, c);
        // TODO: The runtime should actually call retain but it currently does not
        if (keep_ownership) {
            CL_CHECK(clRetainProgram(native_module));
        }
        // The following code should be used if the run-time calls retain
        // if (!keep_ownership) {
        // CL_CHECK(clReleaseProgram(native_module));
        // }
        return bundle;
    }
    static auto create_kernel(bundle_t bundle, std::string const &name) -> ::sycl::kernel {
        auto native_bundle = ::sycl::get_native<be_t, bstate_t>(bundle);
        auto native_kernel = cl::create_kernel(native_bundle.front(), name);
        // TODO: Should be necessary according to spec but runtime does not call retain
        // for (auto& nm : native_bundle) {
        // CL_CHECK(clReleaseProgram(nm));
        // }
        auto k = ::sycl::make_kernel<be_t>(native_kernel, bundle.get_context());
        CL_CHECK(clReleaseKernel(native_kernel));
        return k;
    }
};

} // namespace bbfft::sycl

#endif // BUILD_WRAPPER_20230206_HPP
