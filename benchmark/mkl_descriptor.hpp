// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MKL_DESCRIPTOR_20220318_H
#define MKL_DESCRIPTOR_20220318_H

#include <oneapi/mkl/dfti.hpp>

template <typename T> struct to_mkl {};
template <> struct to_mkl<float> {
    static constexpr auto value = oneapi::mkl::dft::precision::SINGLE;
};
template <> struct to_mkl<double> {
    static constexpr auto value = oneapi::mkl::dft::precision::DOUBLE;
};
template <typename T> inline constexpr auto to_mkl_v = to_mkl<T>::value;

template <typename T> struct mkl_descriptor {
    static auto make(unsigned int M, unsigned int N, unsigned int K, bool inplace) {
        auto N_out = N / 2 + 1;
        auto N_in = inplace ? 2 * N_out : N;
        auto plan = std::make_unique<
            oneapi::mkl::dft::descriptor<to_mkl_v<T>, oneapi::mkl::dft::domain::REAL>>(N);
        plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, K);
        plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, N_in * M);
        plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, N_out * M);
        std::int64_t strides[2] = {0, M};
        plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE);
        return plan;
    }
};

template <typename T> struct mkl_descriptor<std::complex<T>> {
    static auto make(unsigned int M, unsigned int N, unsigned int K, bool inplace) {
        auto plan = std::make_unique<
            oneapi::mkl::dft::descriptor<to_mkl_v<T>, oneapi::mkl::dft::domain::COMPLEX>>(N);
        plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, K);
        plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, N * M);
        plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, N * M);
        std::int64_t strides[2] = {0, M};
        plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE);
        return plan;
    }
};

#endif // MKL_DESCRIPTOR_20220318_H
