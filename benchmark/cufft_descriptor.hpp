// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUFFT_DESCRIPTOR_20220318_H
#define CUFFT_DESCRIPTOR_20220318_H

#include <cufft.h>

template <typename T> struct to_cufft {};
template <> struct to_cufft<float> {
    static constexpr auto value = CUFFT_R2C;
};
template <> struct to_cufft<double> {
    static constexpr auto value = CUFFT_D2Z;
};
template <> struct to_cufft<std::complex<float>> {
    static constexpr auto value = CUFFT_C2C;
};
template <> struct to_cufft<std::complex<double>> {
    static constexpr auto value = CUFFT_Z2Z;
};
template <typename T> inline constexpr auto to_cufft_v = to_cufft<T>::value;

template <typename T> struct to_inverse_cufft {};
template <> struct to_inverse_cufft<float> {
    static constexpr auto value = CUFFT_C2R;
};
template <> struct to_inverse_cufft<double> {
    static constexpr auto value = CUFFT_Z2D;
};
template <> struct to_inverse_cufft<std::complex<float>> {
    static constexpr auto value = CUFFT_C2C;
};
template <> struct to_inverse_cufft<std::complex<double>> {
    static constexpr auto value = CUFFT_Z2Z;
};
template <typename T> inline constexpr auto to_inverse_cufft_v = to_inverse_cufft<T>::value;

template <typename T> struct cufft_descriptor {
    static auto make(unsigned int M, unsigned int N, unsigned int K, bool inplace, bool inverse) {
        int n = N;
        int Nout = N / 2 + 1;
        int Nsym = inplace ? 2 * Nout : N;
        int inembed = Nsym;
        int idist = inembed * M;
        int onembed = Nout;
        int odist = onembed * M;
        if (inverse) {
            std::swap(inembed, onembed);
            std::swap(idist, odist);
        }
        cufftHandle plan;
        cufftPlanMany(&plan, 1, &n, &inembed, M, idist, &onembed, M, odist,
                      inverse ? to_inverse_cufft_v<T> : to_cufft_v<T>, K);
        return plan;
    }
};

template <typename T> struct cufft_descriptor<std::complex<T>> {
    static auto make(unsigned int M, unsigned int N, unsigned int K, bool, bool) {
        int n = N;
        cufftHandle plan;
        cufftPlanMany(&plan, 1, &n, &n, M, N * M, &n, M, N * M, to_cufft_v<std::complex<T>>, K);
        return plan;
    }
};

#endif // CUFFT_DESCRIPTOR_20220318_H
