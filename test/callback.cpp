// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "fft.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"
#include "bbfft/tensor_indexer.hpp"

#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace bbfft;
using namespace sycl;

TEST_CASE_TEMPLATE("load callback", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{16};
    auto MM = std::vector<std::size_t>{1, 32};
    auto NN = std::vector<std::size_t>{8, 64, 212};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    auto rd = std::random_device{};
    auto gen = std::mt19937(rd());
    auto Y = std::uniform_real_distribution<T>(0.0, 1.0);

    std::size_t N_ext = 2 * N;

    auto Xi_ref = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N_ext / 2 + 1, K});
    auto Xi = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N / 2 + 1, K});
    auto xi = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N_ext, K});
    auto X_ref = new std::complex<T>[Xi_ref.size()];
    auto X_ref_d = malloc_device<std::complex<T>>(Xi_ref.size(), Q);
    auto x_ref_d = malloc_device<T>(xi.size(), Q);
    auto x_ref = new T[xi.size()];
    auto X = new std::complex<T>[Xi.size()];
    auto X_d = malloc_device<std::complex<T>>(Xi.size(), Q);
    auto x_d = malloc_device<T>(xi.size(), Q);
    auto x = new T[xi.size()];

    configuration cfg_ref = {1,
                             {M, N_ext, K},
                             to_precision_v<T>,
                             direction::backward,
                             transform_type::c2r,
                             fit_array<max_tensor_dim>(Xi_ref.stride()),
                             fit_array<max_tensor_dim>(xi.stride())};
    auto plan_ref = make_plan(cfg_ref, Q);

    char const load_template[] = R"OpenCL(
%s2 load(global %s2* in, size_t offset) {
    uint n = offset / %lu %% %lu;
    if (n < %lu) {
        uint k = offset / %lu;
        return in[offset - k * %lu];
    }
    return 0;
})OpenCL";
    char load[1024];
    char const *real_type = std::is_same_v<double, T> ? "double" : "float";
    std::size_t length =
        snprintf(load, sizeof(load), load_template, real_type, real_type, Xi_ref.shape(0),
                 Xi_ref.shape(1), Xi.shape(1), Xi_ref.shape(0) * Xi_ref.shape(1),
                 Xi_ref.shape(0) * (Xi_ref.shape(1) - Xi.shape(1)));

    configuration cfg = {1,
                         {M, N_ext, K},
                         to_precision_v<T>,
                         direction::backward,
                         transform_type::c2r,
                         fit_array<max_tensor_dim>(Xi_ref.stride()),
                         fit_array<max_tensor_dim>(xi.stride()),
                         {load, length, "load"}};
    auto plan = make_plan(cfg, Q);

    for (std::size_t k = 0; k < Xi_ref.shape(2); ++k) {
        for (std::size_t m = 0; m < Xi_ref.shape(0); ++m) {
            for (std::size_t n = 0; n < Xi.shape(1); ++n) {
                auto v = std::complex<T>{Y(gen), Y(gen)};
                X_ref[Xi_ref(m, n, k)] = v;
                X[Xi(m, n, k)] = v;
            }
            for (std::size_t n = Xi.shape(1); n < Xi_ref.shape(1); ++n) {
                X_ref[Xi_ref(m, n, k)] = T(0.0);
            }
        }
    }

    Q.copy(X_ref, X_ref_d, Xi_ref.size()).wait();
    plan_ref.execute(X_ref_d, x_ref_d).wait();
    Q.copy(x_ref_d, x_ref, xi.size()).wait();

    Q.copy(X, X_d, Xi.size()).wait();
    plan.execute(X_d, x_d).wait();
    Q.copy(x_d, x, xi.size()).wait();

    for (std::size_t j = 0; j < xi.size(); ++j) {
        REQUIRE(x_ref[j] == x[j]);
    }

    delete[] X_ref;
    free(X_ref_d, Q);
    free(x_ref_d, Q);
    delete[] x_ref;
    delete[] X;
    free(X_d, Q);
    free(x_d, Q);
    delete[] x;
}

TEST_CASE_TEMPLATE("store callback", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{16};
    auto MM = std::vector<std::size_t>{1, 32};
    auto NN = std::vector<std::size_t>{8, 64, 212};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    auto rd = std::random_device{};
    auto gen = std::mt19937(rd());
    auto Y = std::uniform_real_distribution<T>(0.0, 1.0);

    std::size_t N_cut = N / 4;

    auto xi = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N, K});
    auto Xi_ref = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N / 2 + 1, K});
    auto Xi = tensor_indexer<std::size_t, 3u, layout::col_major>({M, N_cut, K});
    auto x = new T[xi.size()];
    auto x_d = malloc_device<T>(xi.size(), Q);
    auto X_ref = new std::complex<T>[Xi_ref.size()];
    auto X_ref_d = malloc_device<std::complex<T>>(Xi_ref.size(), Q);
    auto X = new std::complex<T>[Xi.size()];
    auto X_d = malloc_device<std::complex<T>>(Xi.size(), Q);

    configuration cfg_ref = {1,
                             {M, N, K},
                             to_precision_v<T>,
                             direction::forward,
                             transform_type::r2c,
                             fit_array<bbfft::max_tensor_dim>(xi.stride()),
                             fit_array<bbfft::max_tensor_dim>(Xi_ref.stride())};
    auto plan_ref = make_plan(cfg_ref, Q);

    char const store_template[] = R"OpenCL(
void store(global %s2* out, size_t offset, %s2 value) {
    uint n = offset / %lu %% %lu;
    if (n < %lu) {
        uint k = offset / %lu;
        out[offset - k * %lu] = value * ((%s) %a);
    }
})OpenCL";
    char store[1024];
    char const *real_type = std::is_same_v<double, T> ? "double" : "float";
    std::size_t length =
        snprintf(store, sizeof(store), store_template, real_type, real_type, Xi_ref.shape(0),
                 Xi_ref.shape(1), Xi.shape(1), Xi_ref.shape(0) * Xi_ref.shape(1),
                 Xi_ref.shape(0) * (Xi_ref.shape(1) - Xi.shape(1)), real_type, 1.0 / N);

    configuration cfg = {1,
                         {M, N, K},
                         to_precision_v<T>,
                         direction::forward,
                         transform_type::r2c,
                         fit_array<bbfft::max_tensor_dim>(xi.stride()),
                         fit_array<bbfft::max_tensor_dim>(Xi_ref.stride()),
                         {store, length, nullptr, "store"}};
    auto plan = make_plan(cfg, Q);

    for (std::size_t j = 0; j < xi.size(); ++j) {
        x[j] = Y(gen);
    }

    Q.copy(x, x_d, xi.size()).wait();
    plan_ref.execute(x_d, X_ref_d).wait();
    plan.execute(x_d, X_d).wait();
    Q.copy(X_ref_d, X_ref, Xi_ref.size()).wait();
    Q.copy(X_d, X, Xi.size()).wait();

    for (std::size_t k = 0; k < Xi.shape(2); ++k) {
        for (std::size_t n = 0; n < Xi.shape(1); ++n) {
            for (std::size_t m = 0; m < Xi.shape(0); ++m) {
                auto ref = X_ref[Xi_ref(m, n, k)] * T(1.0 / N);
                REQUIRE(ref.real() == X[Xi(m, n, k)].real());
                REQUIRE(ref.imag() == X[Xi(m, n, k)].imag());
            }
        }
    }

    delete[] x;
    free(x_d, Q);
    delete[] X_ref;
    free(X_ref_d, Q);
    delete[] X;
    free(X_d, Q);
}
