// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "adapter.hpp"
#include "args.hpp"
#include "check.hpp"
#include "common.hpp"
#include "csv_printer.hpp"
#include "cuda_allocator.hpp"
#include "cufft_descriptor.hpp"
#include "initialize.hpp"
#include "result.hpp"
#include "signal.hpp"
#include "tensor.hpp"

#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#include <complex>
#include <memory>
#include <vector>

template <typename T, bool Inplace>
result test_body(unsigned int M, unsigned int N, unsigned int K, bool inverse) {
    auto apt = adapter<T, Inplace>(M, N, K);

    auto [x, X] = apt.make_inout(std::make_shared<cuda_allocator>());
    auto x_ref = x.make_ref();
    auto X_ref = X.make_ref();

    initialize(x_ref, X_ref, inverse);

    // Initialize FFT descriptor
    auto plan = cufft_descriptor<T>::make(M, N, K, Inplace, inverse);

    auto const compute_fwd = [&]() {
        for (unsigned int i = 0; i < M; ++i) {
            cufftXtExec(plan, x_ref.data() + i, X_ref.data() + i, CUFFT_FORWARD);
        }
    };
    auto const compute_bwd = [&]() {
        for (unsigned int i = 0; i < M; ++i) {
            cufftXtExec(plan, X_ref.data() + i, x_ref.data() + i, CUFFT_INVERSE);
        }
    };
    auto const compute = [&]() {
        if (inverse) {
            compute_bwd();
        } else {
            compute_fwd();
        }
        cudaDeviceSynchronize();
    };

    compute();
    check(x_ref, X_ref, inverse);
    // Perform forward transforms on real arrays

    double min_exec_time_ns = bench(compute);
    cufftDestroy(plan);

    return result{apt.precision(),
                  N,
                  Inplace,
                  M,
                  K,
                  apt.domain(),
                  inverse,
                  min_exec_time_ns * 1e-9,
                  apt.bytes() / min_exec_time_ns,
                  apt.flops() / min_exec_time_ns};
}

void test(args const &a) {
    auto printer = csv_printer(&std::cout, column_names());

    using test_fun_t = result (*)(unsigned int, unsigned int, unsigned int, bool);
    test_fun_t test_fun_i = nullptr, test_fun_o = nullptr;
    if (a.p == 's') {
        if (a.d == 'r') {
            test_fun_i = &test_body<float, true>;
            test_fun_o = &test_body<float, false>;
        } else {
            test_fun_i = &test_body<std::complex<float>, true>;
            test_fun_o = &test_body<std::complex<float>, false>;
        }
    } else {
        if (a.d == 'r') {
            test_fun_i = &test_body<double, true>;
            test_fun_o = &test_body<double, false>;
        } else {
            test_fun_i = &test_body<std::complex<double>, true>;
            test_fun_o = &test_body<std::complex<double>, false>;
        }
    }

    for (auto M : a.MM) {
        for (auto N : a.NN) {
            if (a.inplace && test_fun_i) {
                auto r = (*test_fun_i)(M, N, a.KK(M, N), a.inverse);
                print(r, printer);
            } else if (test_fun_o) {
                auto r = (*test_fun_o)(M, N, a.KK(M, N), a.inverse);
                print(r, printer);
            }
        }
    }
}
