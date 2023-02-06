// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "aot_cache.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"

#include <CL/sycl.hpp>

#include <chrono>
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
    auto q = sycl::queue{};
    std::cout
        << "This example measures plan creation time with and without ahead-of-time compilation."
        << std::endl;

    auto cfg =
        configuration{1, {1, 128, 12893}, precision::f32, direction::forward, transform_type::r2c};

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
    auto cache = aot_cache(q);
    auto end = high_resolution_clock::now();
    duration<double> time = end - start;
    std::cout << "aot_cache initialization time: " << time.count() << std::endl;
    auto t2 = bench([&]() { make_plan(cfg, q, &cache); });
    print_result("aot kernels", t2);

    return 0;
}
