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

#include "bbfft/cl/error.hpp"
#include "vkFFT.h"
#include "vkfft_error.hpp"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <algorithm>
#include <memory>
#include <tuple>
#include <type_traits>

using namespace bbfft;

template <typename T, bool Inplace>
result test_body(sycl::queue Q, unsigned int M, unsigned int N, unsigned int K, bool inverse) {
    auto apt = adapter<T, Inplace>(M, N, K);

    auto [x, X] = apt.make_inout(std::make_shared<sycl_allocator>(Q));
    auto x_ref = x.make_ref();
    auto X_ref = X.make_ref();

    initialize_input_tensors(Q, x_ref, X_ref, inverse);

    auto istride = x.stride();
    auto ostride = X.stride();
    if (inverse) {
        std::swap(istride, ostride);
    }

    auto native_queue = sycl::get_native<sycl::backend::opencl, sycl::queue>(Q);
    auto native_device = sycl::get_native<sycl::backend::opencl, sycl::device>(Q.get_device());
    auto native_context = sycl::get_native<sycl::backend::opencl, sycl::context>(Q.get_context());

    using Tx = typename decltype(x_ref)::value_t;
    using TX = typename decltype(X_ref)::value_t;
    std::size_t buffer_size = inverse ? sizeof(TX) * X_ref.size() : sizeof(Tx) * x_ref.size();
    cl_mem buffer;
    if (inverse) {
        TX *X_host = new TX[X_ref.size()];
        Q.memcpy(X_host, X_ref.data(), buffer_size).wait();
        cl_int err;
        buffer = clCreateBuffer(native_context, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(native_queue, buffer, CL_TRUE, 0, buffer_size, X_host, 0,
                                      nullptr, nullptr));
        delete[] X_host;
    } else {
        Tx *x_host = new Tx[x_ref.size()];
        Q.memcpy(x_host, x_ref.data(), buffer_size).wait();
        cl_int err;
        buffer = clCreateBuffer(native_context, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(native_queue, buffer, CL_TRUE, 0, buffer_size, x_host, 0,
                                      nullptr, nullptr));
        delete[] x_host;
    }

    bool is_dp = std::is_same_v<double, typename real_type<T>::type>;
    int direction = inverse ? 1 : -1;
    VkFFTApplication app = {};
    VkFFTConfiguration configuration = {};
    if (M == 1) {
        configuration.FFTdim = 1;
        configuration.size[0] = N;
    } else {
        configuration.FFTdim = 2;
        configuration.size[0] = M;
        configuration.size[1] = N;
        configuration.omitDimension[0] = 1;
    }
    configuration.numberBatches = K;
    configuration.device = &native_device;
    configuration.context = &native_context;
    configuration.performR2C = apt.is_r2c();
    configuration.doublePrecision = is_dp;
    VKFFT_CHECK(initializeVkFFT(&app, configuration));

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = &buffer;
    launchParams.commandQueue = &native_queue;
    VKFFT_CHECK(VkFFTAppend(&app, direction, &launchParams));
    CL_CHECK(clFinish(native_queue));

    if (inverse) {
        Tx *x_host = new Tx[x_ref.size()];
        CL_CHECK(clEnqueueReadBuffer(native_queue, buffer, CL_TRUE, 0, buffer_size, x_host, 0,
                                     nullptr, nullptr));
        check(tensor<Tx, 3u>(x_host, x_ref.shape()), X_ref, inverse);
        delete[] x_host;
    } else {
        TX *X_host = new TX[X_ref.size()];
        CL_CHECK(clEnqueueReadBuffer(native_queue, buffer, CL_TRUE, 0, buffer_size, X_host, 0,
                                     nullptr, nullptr));
        check(x_ref, tensor<TX, 3u>(X_host, X_ref.shape()), inverse);
        delete[] X_host;
    }

    double min_exec_time_ns = bench([&]() {
        VKFFT_CHECK(VkFFTAppend(&app, direction, &launchParams));
        CL_CHECK(clFinish(native_queue));
    });

    deleteVkFFT(&app);
    CL_CHECK(clReleaseMemObject(buffer));
    CL_CHECK(clReleaseCommandQueue(native_queue));
    CL_CHECK(clReleaseContext(native_context));

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
        } else {
            test_fun_i = &test_body<std::complex<float>, true>;
        }
    }
#ifndef NO_DOUBLE_PRECISION
    else {
        if (a.d == 'r') {
            test_fun_i = &test_body<double, true>;
        } else {
            test_fun_i = &test_body<std::complex<double>, true>;
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
