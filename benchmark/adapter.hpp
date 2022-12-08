// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ADAPTER20220210_H
#define ADAPTER20220210_H

#include "allocator.hpp"
#include "tensor.hpp"

#include <complex>
#include <memory>
#include <utility>

template <typename T, bool Inplace> class adapter {
  private:
    unsigned int M, N, K;

  public:
    adapter(unsigned int M, unsigned int N, unsigned int K) : M(M), N(N), K(K) {}

    std::size_t flops() { return 2.5 * N * std::log2(N) * M * K; }
    std::size_t bytes() {
        auto N_out = N / 2 + 1;
        return (N * sizeof(T) + N_out * sizeof(std::complex<T>)) * M * K;
    }
    auto make_inout(std::shared_ptr<allocator> alloc) {
        auto N_out = N / 2 + 1;
        if constexpr (Inplace) {
            auto in = managed_tensor<T, 3u>(alloc, {K, N, M}, {M * 2 * N_out, M, 1});
            auto out = tensor<std::complex<T>, 3u>(reinterpret_cast<std::complex<T> *>(in.data()),
                                                   {K, N_out, M});
            return std::make_pair(std::move(in), std::move(out));
        } else {
            auto in = managed_tensor<T, 3u>(alloc, {K, N, M});
            auto out = managed_tensor<std::complex<T>, 3u>(std::move(alloc), {K, N_out, M});
            return std::make_pair(std::move(in), std::move(out));
        }
    }
    auto precision() { return typeid(T).name(); }
    auto domain() { return "real"; }
    auto is_r2c() { return true; }
};

template <typename T, bool Inplace> struct adapter<std::complex<T>, Inplace> {
  private:
    unsigned int M, N, K;

  public:
    adapter(unsigned int M, unsigned int N, unsigned int K) : M(M), N(N), K(K) {}

    std::size_t flops() { return 5 * N * std::log2(N) * M * K; }
    std::size_t bytes() { return 2 * N * sizeof(std::complex<T>) * M * K; }
    auto make_inout(std::shared_ptr<allocator> alloc) {
        auto in = managed_tensor<std::complex<T>, 3u>(alloc, {K, N, M});
        if constexpr (Inplace) {
            auto out = tensor<std::complex<T>, 3u>(in.data(), {K, N, M});
            return std::make_pair(std::move(in), std::move(out));
        } else {
            auto out = managed_tensor<std::complex<T>, 3u>(std::move(alloc), {K, N, M});
            return std::make_pair(std::move(in), std::move(out));
        }
    }
    auto precision() { return typeid(T).name(); }
    auto domain() { return "complex"; }
    auto is_r2c() { return false; }
};

#endif // ADAPTER20220210_H
