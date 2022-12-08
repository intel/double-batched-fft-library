// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "fft.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"
#include "bbfft/tensor_indexer.hpp"

#include <complex>
#include <random>
#include <vector>

using namespace bbfft;
using namespace sycl;

template <typename T> void test_c2c_forward(configuration cfg, queue Q) {
    std::size_t M = cfg.shape[0], N = cfg.shape[1], K = cfg.shape[2];
    auto xi = tensor_indexer<std::size_t, 3u, layout::col_major>(fit_array<3u>(cfg.shape),
                                                                 fit_array<3u>(cfg.istride));
    auto Xi = tensor_indexer<std::size_t, 3u, layout::col_major>(fit_array<3u>(cfg.shape),
                                                                 fit_array<3u>(cfg.ostride));

    auto x = malloc_device<std::complex<T>>(xi.size(), Q);
    auto X = new std::complex<T>[Xi.size()];
    auto plan = make_plan(cfg, Q);

    auto scale = [K](std::size_t k) { return T(1.0) + k / T(K); };
    auto basis_no = [N](std::size_t m, std::size_t k) -> long { return (m + k) % N; };
    Q.parallel_for(range{K, N, M}, [=](id<3> idx) {
         auto k = idx[0], n = idx[1], m = idx[2];
         T arg = (T(tau) / N) * basis_no(m, k) * n;
         x[xi(m, n, k)] = scale(k) * std::complex{std::cos(arg), std::sin(arg)} / T(N);
     }).wait();
    plan.execute(x).wait();
    Q.copy(x, X, xi.size()).wait();

    double eps = tol<T>(N);
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            for (std::size_t m = 0; m < M; ++m) {
                T ref = scale(k) * periodic_delta<T>(static_cast<long>(n) - basis_no(m, k), N);
                REQUIRE(X[Xi(m, n, k)].real() == doctest::Approx(ref).epsilon(eps));
                REQUIRE(X[Xi(m, n, k)].imag() == doctest::Approx(T(0.0)).epsilon(eps));
            }
        }
    }

    free(x, Q);
    delete[] X;
}

TEST_CASE_TEMPLATE("c2c forward", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{1, 32};
    auto MM = std::vector<std::size_t>{1, 2, 3, 16, 17, 64, 256, 1024};
    auto NN =
        std::vector<std::size_t>{2, 3, 5, 7, 11, 13, 4, 8, 16, 32, 128, 256, 512, 27, 63, 105, 363};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    configuration cfg = {1, {M, N, K}, to_precision_v<T>, direction::forward};
    test_c2c_forward<T>(cfg, Q);
}

TEST_CASE_TEMPLATE("c2c non-packed forward", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{33};
    auto MM = std::vector<std::size_t>{1, 32};
    auto NN = std::vector<std::size_t>{6, 17, 102};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    std::array<std::size_t, bbfft::max_tensor_dim> stride = {1, (M + 1), (M + 1) * (N + 1)};
    configuration cfg = {
        1, {M, N, K}, to_precision_v<T>, direction::forward, transform_type::c2c, stride, stride};
    test_c2c_forward<T>(cfg, Q);
}

TEST_CASE_TEMPLATE("c2c identity", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{16};
    auto MM = std::vector<std::size_t>{1, 32};
    auto NN = std::vector<std::size_t>{5, 63, 212};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    auto rd = std::random_device{};
    auto gen = std::mt19937(rd());
    auto Y = std::uniform_real_distribution<T>(0.0, 1.0);

    auto xi = tensor_indexer<std::size_t, 3u>({K, N, M});
    std::size_t size = xi.size();
    auto x_ref = new std::complex<T>[size];
    auto x_device = malloc_device<std::complex<T>>(size, Q);
    auto x_host = new std::complex<T>[size];

    configuration cfg = {1, {M, N, K}, to_precision_v<T>, direction::forward};
    auto plan = make_plan(cfg, Q);
    cfg.dir = direction::backward;
    auto iplan = make_plan(cfg, Q);

    for (std::size_t j = 0; j < size; ++j) {
        x_ref[j] = std::complex{Y(gen), Y(gen)};
    }

    Q.copy(x_ref, x_device, size).wait();
    plan.execute(x_device).wait();
    iplan.execute(x_device).wait();
    Q.copy(x_device, x_host, M * N * K).wait();

    double eps = tol<T>(N);
    for (std::size_t j = 0; j < size; ++j) {
        REQUIRE(x_ref[j].real() == doctest::Approx(x_host[j].real() / N).epsilon(eps));
        REQUIRE(x_ref[j].imag() == doctest::Approx(x_host[j].imag() / N).epsilon(eps));
    }

    free(x_device, Q);
    delete[] x_host;
    delete[] x_ref;
}
