// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "initialize.hpp"
#define CUDA
#include "signal.hpp"
#undef CUDA
#include <algorithm>
#include <complex>
#include <thrust/complex.h>

template <typename T> struct real_type {
    using type = T;
};
template <typename T> struct real_type<thrust::complex<T>> {
    using type = T;
};

template <typename U, typename V, bool Inverse>
__global__ void init_kernel(V *x, unsigned int M, unsigned int N, unsigned int K, unsigned int s0,
                            unsigned int s1, unsigned int s2) {
    using real_type = typename real_type<U>::type;
    auto k = threadIdx.x + blockIdx.x * blockDim.x;
    auto n = blockIdx.y;
    auto m = blockIdx.z;
    if (k < K) {
        auto s = signal<U>(N, m + k, real_type(1.0) + k / real_type(K));
        if constexpr (Inverse) {
            x[k * s0 + n * s1 + m * s2] = s.X(n);
        } else {
            x[k * s0 + n * s1 + m * s2] = s.x(n) / real_type(N);
        }
    }
}

template <typename T> struct cu_type {
    using type = T;
};
template <typename T> struct cu_type<std::complex<T>> {
    using type = thrust::complex<T>;
};

template <typename U, typename V> void initialize(tensor<U, 3u> x, tensor<V, 3u> X, bool inverse) {
    unsigned int N = x.shape(1);
    auto shape = inverse ? X.shape() : x.shape();
    unsigned int K = shape[0];
    unsigned int NN = shape[1];
    unsigned int M = shape[2];
    unsigned int Kb = std::min(128u, K);
    auto num_threads = dim3(Kb, 1, 1);
    auto num_blocks = dim3((K - 1) / Kb + 1, NN, M);
    if (inverse) {
        init_kernel<typename cu_type<U>::type, typename cu_type<V>::type, true>
            <<<num_blocks, num_threads>>>(reinterpret_cast<typename cu_type<V>::type *>(X.data()),
                                          M, N, K, X.stride(0), X.stride(1), X.stride(2));
    } else {
        init_kernel<typename cu_type<U>::type, typename cu_type<U>::type, false>
            <<<num_blocks, num_threads>>>(reinterpret_cast<typename cu_type<U>::type *>(x.data()),
                                          M, N, K, x.stride(0), x.stride(1), x.stride(2));
    }
    cudaDeviceSynchronize();
}

template void initialize<std::complex<float>, std::complex<float>>(tensor<std::complex<float>, 3u>,
                                                                   tensor<std::complex<float>, 3u>,
                                                                   bool);
template void initialize<float, std::complex<float>>(tensor<float, 3u>,
                                                     tensor<std::complex<float>, 3u>, bool);
template void
initialize<std::complex<double>, std::complex<double>>(tensor<std::complex<double>, 3u>,
                                                       tensor<std::complex<double>, 3u>, bool);
template void initialize<double, std::complex<double>>(tensor<double, 3u>,
                                                       tensor<std::complex<double>, 3u>, bool);

