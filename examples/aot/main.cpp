// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/aot_cache.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"
#include "bbfft/sycl/online_compiler.hpp"

#include <CL/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

using namespace std::chrono;
using namespace bbfft;

template <typename F> auto bench(F &&f) {
    std::array<double, 5> result;
    for (int i = 0; i < result.size(); ++i) {
        auto start = high_resolution_clock::now();
        f();
        auto end = high_resolution_clock::now();
        duration<double> time = end - start;
        result[i] = time.count();
    }
    return result;
}

int main(int argc, char **argv) {
    auto q = ::sycl::queue{};

    std::cout
        << "This example measures plan creation time with and without ahead-of-time compilation."
        << std::endl;

    std::size_t N = 32;
    if (argc >= 2) {
        N = static_cast<std::size_t>(std::atol(argv[1]));
    }

    auto cfg =
        configuration{1, {1, N, 12893}, precision::f32, direction::forward, transform_type::r2c};

    auto const print_result = [](char const *description, auto result) {
        std::cout << description << ":" << std::endl;
        int i = 0;
        for (auto r : result) {
            std::cout << "  " << i++ << ": " << r << std::endl;
        }
    };

    auto t1 = bench([&]() { make_plan(cfg, q); });
    print_result("no aot kernels", t1);

    auto start = high_resolution_clock::now();
    auto cache = aot_cache{};
    try {
        extern std::uint8_t _binary_kernels_bin_start, _binary_kernels_bin_end;
        cache.register_module(bbfft::sycl::create_aot_module(
            &_binary_kernels_bin_start, &_binary_kernels_bin_end - &_binary_kernels_bin_start,
            module_format::native, q.get_context(), q.get_device()));
    } catch (std::exception const &e) {
        std::cerr << "Could not load ahead-of-time compiled FFT kernels:" << std::endl
                  << e.what() << std::endl;
    }
    auto end = high_resolution_clock::now();
    duration<double> time = end - start;
    std::cout << "aot_cache initialization time: " << time.count() << std::endl;

    auto t2 = bench([&]() { make_plan(cfg, q, &cache); });
    print_result("aot kernels", t2);

    return 0;
}
