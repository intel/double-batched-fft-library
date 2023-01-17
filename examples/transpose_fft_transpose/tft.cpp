// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/sycl/make_plan.hpp>

#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace bbfft;
using namespace sycl;
using namespace std::chrono;

static constexpr std::size_t default_tensor_size = 512 * 1000 * 1000;

template <typename T> double bench(int Ntimes, T fun) {
    double min_time = std::numeric_limits<double>::max();
    for (int i = 0; i < Ntimes; ++i) {
        auto start = high_resolution_clock::now();
        fun();
        auto end = high_resolution_clock::now();
        duration<double> time = end - start;
        min_time = std::min(min_time, time.count());
    }
    return min_time;
}

template <typename T, std::size_t M, std::size_t N>
auto transpose_simple(queue q, T const *x, T *xt, std::size_t K) {
    return q.submit([&](handler &h) {
        h.parallel_for(range{K, N, M}, [=](id<3> idx) {
            auto k = idx[0];
            auto n = idx[1];
            auto m = idx[2];
            xt[n + m * N + k * M * N] = x[m + n * M + k * M * N];
        });
    });
}

template <typename T, std::size_t M, std::size_t N>
auto transpose_optimized(queue q, T const *x, T *xt, std::size_t K) {
    constexpr std::size_t tile_size = 16;
    constexpr std::size_t num_slices = 8;
    std::size_t Mb = (M - 1) / tile_size + 1;
    std::size_t Nb = (N - 1) / tile_size + 1;
    return q.submit([&](handler &h) {
#if __INTEL_CLANG_COMPILER < 20230000
        using local_accessor_t = accessor<T, 2, access::mode::read_write, access::target::local>;
#else
        using local_accessor_t = local_accessor<T, 2>;
#endif
        auto mat = local_accessor_t({tile_size, tile_size + 1}, h);
        h.parallel_for(
            nd_range<3>{{K, Nb * num_slices, Mb * tile_size}, {1, num_slices, tile_size}},
            [=](nd_item<3> idx) {
                auto mg = idx.get_group(2);
                auto ng = idx.get_group(1);
                auto k = idx.get_global_id(0);
                auto x_sub = x + mg * tile_size + ng * tile_size * M + k * M * N;
                auto xt_sub = xt + ng * tile_size + mg * tile_size * N + k * M * N;
                auto i = idx.get_local_id(2);
                auto j = idx.get_local_id(1);
                auto tm = min(tile_size, M - mg * tile_size);
                auto tn = min(tile_size, N - ng * tile_size);
                if (i < tm) {
                    if (N % tile_size == 0 || tn == tile_size) {
#pragma unroll
                        for (int b = 0; b < tile_size; b += num_slices) {
                            mat[b + j][i] = x_sub[i + (b + j) * M];
                        }
                    } else {
                        for (int b = j; b < tn; b += num_slices) {
                            mat[b][i] = x_sub[i + b * M];
                        }
                    }
                }
                group_barrier(idx.get_group());
                if (i < tn) {
                    if (M % tile_size == 0 || tm == tile_size) {
#pragma unroll
                        for (int b = 0; b < tile_size; b += num_slices) {
                            xt_sub[i + (b + j) * N] = mat[i][b + j];
                        }
                    } else {
                        for (int b = j; b < tm; b += num_slices) {
                            xt_sub[i + b * N] = mat[i][b];
                        }
                    }
                }
            });
    });
}

template <typename T, std::size_t M, std::size_t N> int test(queue Q, std::size_t K) {
    if (K == 0) {
        K = default_tensor_size / (sizeof(std::complex<T>) * M * N);
    }
    std::cout << M << " x " << N << " x " << K << std::endl;

    std::size_t size = M * N * K;
    auto x_host = new std::complex<T>[size];
    auto x = malloc_device<std::complex<T>>(size, Q);
    auto xt = malloc_device<std::complex<T>>(size, Q);
    auto X = malloc_device<std::complex<T>>(size, Q);
    auto X_host = new std::complex<T>[size];

    auto const init = [&]() {
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t n = 0; n < N; ++n) {
                for (std::size_t m = 0; m < M; ++m) {
                    x_host[m + n * M + k * M * N] =
                        std::complex<T>{m / T(M) + n / T(N) + k / T(K), T(0.0)};
                }
            }
        }
        Q.copy(x_host, x, size).wait();
    };

    configuration cfg_tft = {
        1, {1, N, M * K}, to_precision_v<T>, direction::forward, transform_type::c2c};
    auto plan_tft = make_plan(cfg_tft, Q);
    constexpr std::size_t tile_size = 16;
    constexpr std::size_t num_slices = 8;
    constexpr std::size_t blocks_per_tile = tile_size / num_slices;
    constexpr std::size_t batch_size = 256 / (num_slices * tile_size);
    auto const execute_tft = [&]() {
        transpose_optimized<std::complex<T>, M, N>(Q, x, xt, K).wait();
        plan_tft.execute(xt).wait();
        transpose_optimized<std::complex<T>, N, M>(Q, xt, X, K).wait();
    };

    configuration cfg_nu = {
        1, {M, N, K}, to_precision_v<T>, direction::forward, transform_type::c2c};
    auto plan_nu = make_plan(cfg_nu, Q);
    auto const execute_nu = [&]() { plan_nu.execute(x).wait(); };

    init();
    execute_tft();
    execute_nu();

    auto const check = [&]() {
        Q.copy(X, X_host, size).wait();
        Q.copy(x, x_host, size).wait();
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t n = 0; n < N; ++n) {
                for (std::size_t m = 0; m < M; ++m) {
                    auto x1 = x_host[m + n * M + k * M * N];
                    auto x2 = X_host[m + n * M + k * M * N];
                    auto e = std::abs(x1 - x2);
                    if (std::abs(x1 - x2) > std::numeric_limits<T>::epsilon()) {
                        std::cout << "Error: " << e << " (" << x1 << "-" << x2 << ") @ (" << m
                                  << "," << n << "," << k << ")" << std::endl;
                        return -1;
                    }
                }
            }
        }
        return 0;
    };
    if (check()) {
        return -1;
    }

    init();
    double time_t1 =
        bench(10, [&]() { transpose_optimized<std::complex<T>, M, N>(Q, x, xt, K).wait(); });
    double time_ufft = bench(10, [&]() { plan_tft.execute(xt).wait(); });
    double time_t2 =
        bench(10, [&]() { transpose_optimized<std::complex<T>, N, M>(Q, xt, X, K).wait(); });
    double time_tft = bench(10, execute_tft);

    init();
    double time_non_unit = bench(10, execute_nu);

    auto bw = [&](double time) { return 2 * sizeof(std::complex<T>) * size / time * 1.0e-9; };
    std::cout << "Transpose 1 " << time_t1 << " s, " << bw(time_t1) << " GB / s " << std::endl;
    std::cout << "Unit-stride FFT " << time_ufft << " s, " << bw(time_ufft) << " GB / s "
              << std::endl;
    std::cout << "Transpose 2 " << time_t2 << " s, " << bw(time_t2) << " GB / s " << std::endl;
    std::cout << "Transpose-FFT-transpose: " << time_tft << " s, " << bw(time_tft) << " GB / s "
              << std::endl;
    std::cout << "Non-unit stride FFT: " << time_non_unit << " s, " << bw(time_non_unit)
              << " GB / s " << std::endl;
    std::cout << "Speed-up: " << time_tft / time_non_unit << "x" << std::endl;

    free(X, Q);
    free(xt, Q);
    free(x, Q);
    delete[] X_host;
    delete[] x_host;

    return 0;
}

int main(int argc, char **argv) {
    using T = double;

    auto Q = queue();

    std::size_t K = argc >= 2 ? atoi(argv[3]) : 0u;

    test<T, 16, 16>(Q, K);
    test<T, 1120, 32>(Q, K);
    test<T, 128, 512>(Q, K);
    test<T, 70, 16>(Q, K);

    return 0;
}
