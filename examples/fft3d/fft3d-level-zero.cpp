// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "common.hpp"

#include "test_bench_level_zero.hpp"

#include <utility>

template <typename T> void test(args a) {
    if (a.reuse) {
        test_runtime<test_bench_level_zero_regular, real_type_t<T>>(std::move(a));
    } else {
        test_runtime<test_bench_level_zero_immediate, real_type_t<T>>(std::move(a));
    }
}

template void test<float>(args);
template void test<double>(args);
template void test<std::complex<float>>(args);
template void test<std::complex<double>>(args);
