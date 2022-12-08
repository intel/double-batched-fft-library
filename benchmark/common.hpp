// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMMON_20220613_H
#define COMMON_20220613_H

#include <chrono>
#include <limits>

template <typename F> double bench(F f, int nrepeat = 10) {
    double min_exec_time_ns = std::numeric_limits<double>::max();
    for (int i = 0; i < nrepeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        double exec_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        min_exec_time_ns = std::min(min_exec_time_ns, exec_time_ns);
    }
    return min_exec_time_ns;
}

#endif // COMMON_20220613_H
