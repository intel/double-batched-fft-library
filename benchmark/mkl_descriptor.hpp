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
    static auto make(unsigned int M, unsigned int N, unsigned int K) {
        auto Nsym = 2 * (N / 2 + 1);
        auto plan = std::make_unique<
            oneapi::mkl::dft::descriptor<to_mkl_v<T>, oneapi::mkl::dft::domain::REAL>>(N);
        plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, K);
        plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, Nsym * M);
        plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, (Nsym / 2) * M);
        std::int64_t strides[2] = {0, M};
        plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides);
        return plan;
    }
};

template <typename T> struct mkl_descriptor<std::complex<T>> {
    static auto make(unsigned int M, unsigned int N, unsigned int K) {
        auto plan = std::make_unique<
            oneapi::mkl::dft::descriptor<to_mkl_v<T>, oneapi::mkl::dft::domain::COMPLEX>>(N);
        plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, K);
        plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, N * M);
        plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, N * M);
        std::int64_t strides[2] = {0, M};
        plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
        plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides);
        return plan;
    }
};

#endif // MKL_DESCRIPTOR_20220318_H
