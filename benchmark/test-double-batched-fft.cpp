// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "adapter.hpp"
#include "args.hpp"
#include "check.hpp"
#include "common.hpp"
#include "common_sycl.hpp"
#include "csv_printer.hpp"
#include "real_type.hpp"
#include "result.hpp"
#include "sycl_allocator.hpp"
#include "tensor.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <memory>
#include <tuple>

using namespace bbfft;

template <typename T, bool Inplace>
result test_body(sycl::queue Q, unsigned int M, unsigned int N, unsigned int K, bool inverse) {
    auto apt = adapter<T, Inplace>(M, N, K);

    auto [x, X] = apt.make_inout(std::make_shared<sycl_allocator>(Q));
    auto x_ref = x.make_ref();
    auto X_ref = X.make_ref();

    initialize_input_tensors(Q, x_ref, X_ref, inverse);

    direction dir = inverse ? direction::backward : direction::forward;
    transform_type type = transform_type::c2c;
    if (apt.is_r2c()) {
        type = inverse ? transform_type::c2r : transform_type::r2c;
    }

    auto istride = x.stride();
    auto ostride = X.stride();
    if (inverse) {
        std::swap(istride, ostride);
    }
    configuration cfg = {
        1,                                           // dim
        {M, N, K},                                   // shape
        to_precision_v<typename real_type<T>::type>, // precision
        dir,                                         // direction
        type,                                        // transform type
        {istride[2], istride[1], istride[0]},        // input stride
        {ostride[2], ostride[1], ostride[0]}         // output stride
    };

    auto p = make_plan(cfg, Q);
    void *in = x.data();
    void *out = X.data();
    if (inverse) {
        std::swap(in, out);
    }
    p.execute(in, out).wait_and_throw();
    check(x_ref, X_ref, inverse);

    auto min_exec_time_ns = bench([&]() { p.execute(in, out).wait_and_throw(); });
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
    test_fun_t test_fun_i = nullptr;
    test_fun_t test_fun_o = nullptr;
    if (a.p == 's') {
        if (a.d == 'r') {
            test_fun_i = &test_body<float, true>;
            test_fun_o = &test_body<float, false>;
        } else {
            test_fun_i = &test_body<std::complex<float>, true>;
            test_fun_o = &test_body<std::complex<float>, false>;
        }
    }
#ifndef NO_DOUBLE_PRECISION
    else {
        if (a.d == 'r') {
            test_fun_i = &test_body<double, true>;
            test_fun_o = &test_body<double, false>;
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
