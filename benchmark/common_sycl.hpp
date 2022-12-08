// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMMON_SYCL_20220513_H
#define COMMON_SYCL_20220513_H

#include "real_type.hpp"
#include "signal.hpp"

#include <CL/sycl.hpp>
#include <complex>

template <typename U, typename V>
void initialize_input_tensors(sycl::queue Q, tensor<U, 3u> &x, tensor<V, 3u> &X, bool inverse) {
    using real_t = typename real_type<U>::type;
    unsigned int N = x.shape(1);
    if (inverse) {
        auto shape = X.shape();
        unsigned int K = shape[0];
        unsigned int NN = shape[1];
        unsigned int M = shape[2];
        Q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::range{K, NN, M}, [=](sycl::id<3> idx) {
                 auto s = signal<U>(N, idx[0] + idx[2], real_t(1.0) + idx[0] / real_t(K));
                 X(idx[0], idx[1], idx[2]) = s.X(idx[1]);
             });
         }).wait_and_throw();
    } else {
        auto shape = x.shape();
        unsigned int K = shape[0];
        unsigned int NN = shape[1];
        unsigned int M = shape[2];
        Q.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::range{K, NN, M}, [=](sycl::id<3> idx) {
                 auto s = signal<U>(N, idx[0] + idx[2], real_t(1.0) + idx[0] / real_t(K));
                 x(idx[0], idx[1], idx[2]) = s.x(idx[1]) / real_t(N);
             });
         }).wait_and_throw();
    }
}

#endif // COMMON_SYCL_20220513_H
