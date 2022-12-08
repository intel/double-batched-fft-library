// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "adapter.hpp"
#include "args.hpp"
#include "check.hpp"
#include "common.hpp"
#include "common_sycl.hpp"
#include "csv_printer.hpp"
#include "mkl_descriptor.hpp"
#include "result.hpp"
#include "sycl_allocator.hpp"
#include "tensor.hpp"

#include <CL/sycl.hpp>
#include <mkl.h>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>

#include <complex>
#include <memory>

template <typename T, bool Inplace>
result test_body(sycl::queue Q, unsigned int M, unsigned int N, unsigned int K = 1,
                 bool inverse = false) {
    auto apt = adapter<T, Inplace>(M, N, K);

    auto [x, X] = apt.make_inout(std::make_shared<sycl_allocator>(Q));
    auto x_ref = x.make_ref();
    auto X_ref = X.make_ref();

    initialize_input_tensors(Q, x_ref, X_ref, inverse);

    // Initialize FFT descriptor
    auto plan = mkl_descriptor<T>::make(M, N, K);
    if constexpr (Inplace) {
        plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
    } else {
        plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    }
    plan->commit(Q);

    auto const compute_fwd = [&]() {
        if constexpr (Inplace) {
            for (unsigned int i = 0; i < M; ++i) {
                oneapi::mkl::dft::compute_forward(*plan, x_ref.data() + i);
            }
        } else {
            for (unsigned int i = 0; i < M; ++i) {
                oneapi::mkl::dft::compute_forward(*plan, x_ref.data() + i, X_ref.data() + i);
            }
        }
    };
    auto const compute_bwd = [&]() {
        if constexpr (Inplace) {
            for (unsigned int i = 0; i < M; ++i) {
                oneapi::mkl::dft::compute_backward(*plan, X_ref.data() + i);
            }
        } else {
            for (unsigned int i = 0; i < M; ++i) {
                oneapi::mkl::dft::compute_forward(*plan, X_ref.data() + i, x_ref.data() + i);
            }
        }
    };
    auto compute = [&]() {
        if (inverse) {
            compute_bwd();
        } else {
            compute_fwd();
        }
        Q.wait_and_throw();
    };

    compute();
    check(x_ref, X_ref, inverse);
    // Perform forward transforms on real arrays
    double min_exec_time_ns = bench(compute);
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

void test(sycl::queue Q, args const &a) {
    auto printer = csv_printer(&std::cout, column_names());

    using test_fun_t = result (*)(sycl::queue, unsigned int, unsigned int, unsigned int, bool);
    test_fun_t test_fun_i = nullptr, test_fun_o = nullptr;
    if (a.p == 's') {
        if (a.d == 'r') {
            test_fun_i = &test_body<float, true>;
            test_fun_o = nullptr;
        } else {
            test_fun_i = &test_body<std::complex<float>, true>;
            test_fun_o = &test_body<std::complex<float>, false>;
        }
    }
#ifndef NO_DOUBLE_PRECISION
    else {
        if (a.d == 'r') {
            test_fun_i = &test_body<double, true>;
            test_fun_o = nullptr;
        } else {
            test_fun_i = &test_body<std::complex<double>, true>;
            test_fun_o = &test_body<std::complex<double>, false>;
        }
    }
#endif

    for (auto M : a.MM) {
        for (auto N : a.NN) {
            if (a.inplace && test_fun_i) {
                auto r = (*test_fun_i)(Q, M, N, a.KK(M, N), a.inverse);
                print(r, printer);
            } else if (test_fun_o) {
                auto r = (*test_fun_o)(Q, M, N, a.KK(M, N), a.inverse);
                print(r, printer);
            }
        }
    }
}
