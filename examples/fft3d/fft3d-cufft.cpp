// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "utility.hpp"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstdlib>
#include <iostream>

template <typename T> void test(args a) {
    constexpr bool r2c = !is_complex_v<T>;

    std::cout << a.N[0] << " x " << a.N[1] << " x " << a.N[2] << std::endl;

    std::size_t N1_out = r2c ? a.N[0] / 2 + 1 : a.N[0];
    std::size_t N1_in = r2c ? 2 * N1_out : N1_out;
    std::size_t size = N1_in * a.N[1] * a.N[2];
    auto x = new T[size];
    void *x_device = nullptr;
    cudaMalloc(&x_device, size * sizeof(T));
    void *X_device = x_device;
    if (!a.inplace) {
        cudaMalloc(&X_device, size * sizeof(T));
    }

    init(x, a.N[0], a.N[1], a.N[2], a.inplace);
    cudaMemcpy(x_device, x, size * sizeof(T), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftType type;
    if constexpr (std::is_same_v<double, real_type_t<T>>) {
        type = r2c ? CUFFT_D2Z : CUFFT_Z2Z;
    } else {
        type = r2c ? CUFFT_R2C : CUFFT_C2C;
    }
    cufftPlan3d(&plan, a.N[2], a.N[1], a.N[0], type);

    auto const execute3d = [&]() {
        cufftXtExec(plan, x_device, X_device, CUFFT_FORWARD);
        cudaDeviceSynchronize();
    };

    if (a.verbose) {
        std::cout << "Check" << std::endl;
    }
    bench(1, execute3d, a.verbose);

    cudaMemcpy(x, X_device, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (check(x, a.N[0], a.N[1], a.N[2])) {
        if (a.verbose) {
            std::cout << "Bench" << std::endl;
        }
        auto time = bench(10, execute3d, a.verbose);
        std::cout << time << " s, " << 2 * sizeof(std::complex<T>) * size / time * 1.0e-9 << " GB/s"
                  << std::endl;
    }

    cufftDestroy(plan);
    cudaFree(x_device);
    if (!a.inplace) {
        cudaFree(X_device);
    }
    delete[] x;
}

template void test<float>(args);
template void test<double>(args);
template void test<std::complex<float>>(args);
template void test<std::complex<double>>(args);
