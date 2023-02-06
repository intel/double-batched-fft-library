// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/jit_cache_all.hpp"
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
    std::cout << "This example measures plan creation time with and without cache." << std::endl;

    auto cfg = configuration{1, {1, 32, 2048}, precision::f32};

    auto const print_result = [](char const *description, auto result) {
        std::cout << description << ":" << std::endl;
        int i = 0;
        for (auto r : result) {
            std::cout << "  " << i++ << ": " << r << std::endl;
        }
    };

    auto t1 = bench([&]() { make_plan(cfg, q); });
    print_result("no cache", t1);

    jit_cache_all cache;
    auto t2 = bench([&]() { make_plan(cfg, q, &cache); });
    print_result("cache all", t2);

    cfg.shape[2] = 4096;
    auto t3 = bench([&]() { make_plan(cfg, q, &cache); });
    print_result("cache all different batch size", t3);

    return 0;
}
