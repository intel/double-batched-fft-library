// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "root_of_unity.hpp"
#include "clir/builtin_function.hpp"

#include <numeric>

namespace bbfft {

std::complex<double> const *get_w_lut(int N) {
    switch (N) {
    case 2:
        return w_lut_2.data();
    case 3:
        return w_lut_3.data();
    case 4:
        return w_lut_4.data();
    case 5:
        return w_lut_5.data();
    case 6:
        return w_lut_6.data();
    case 7:
        return w_lut_7.data();
    case 8:
        return w_lut_8.data();
    case 9:
        return w_lut_9.data();
    case 10:
        return w_lut_10.data();
    case 11:
        return w_lut_11.data();
    case 12:
        return w_lut_12.data();
    case 13:
        return w_lut_13.data();
    case 14:
        return w_lut_14.data();
    case 15:
        return w_lut_15.data();
    case 16:
        return w_lut_16.data();
    case 17:
        return w_lut_17.data();
    case 18:
        return w_lut_18.data();
    case 19:
        return w_lut_19.data();
    case 20:
        return w_lut_20.data();
    case 21:
        return w_lut_21.data();
    case 22:
        return w_lut_22.data();
    case 23:
        return w_lut_23.data();
    case 24:
        return w_lut_24.data();
    case 25:
        return w_lut_25.data();
    case 26:
        return w_lut_26.data();
    case 27:
        return w_lut_27.data();
    case 28:
        return w_lut_28.data();
    case 29:
        return w_lut_29.data();
    case 30:
        return w_lut_30.data();
    case 31:
        return w_lut_31.data();
    case 32:
        return w_lut_32.data();
    case 33:
        return w_lut_33.data();
    case 34:
        return w_lut_34.data();
    case 35:
        return w_lut_35.data();
    case 36:
        return w_lut_36.data();
    case 37:
        return w_lut_37.data();
    case 38:
        return w_lut_38.data();
    case 39:
        return w_lut_39.data();
    case 40:
        return w_lut_40.data();
    case 41:
        return w_lut_41.data();
    case 42:
        return w_lut_42.data();
    case 43:
        return w_lut_43.data();
    case 44:
        return w_lut_44.data();
    case 45:
        return w_lut_45.data();
    case 46:
        return w_lut_46.data();
    case 47:
        return w_lut_47.data();
    case 48:
        return w_lut_48.data();
    case 49:
        return w_lut_49.data();
    case 50:
        return w_lut_50.data();
    case 51:
        return w_lut_51.data();
    case 52:
        return w_lut_52.data();
    case 53:
        return w_lut_53.data();
    case 54:
        return w_lut_54.data();
    case 55:
        return w_lut_55.data();
    case 56:
        return w_lut_56.data();
    case 57:
        return w_lut_57.data();
    case 58:
        return w_lut_58.data();
    case 59:
        return w_lut_59.data();
    case 60:
        return w_lut_60.data();
    case 61:
        return w_lut_61.data();
    case 62:
        return w_lut_62.data();
    case 63:
        return w_lut_63.data();
    case 64:
        return w_lut_64.data();
    default:
        return nullptr;
    };
}

std::pair<int, int> simplify_power_of_w(int k, int N) {
    // Remove 2 pi periods
    k = k % N;

    // Reduce fraction
    auto g = std::gcd(k, N);
    // g == 0 only if k and N are both zero
    if (g == 0) {
        return {0, 0};
    }
    k /= g;
    N /= g;

    // We have
    // cos(x - 2pi) = cos(x)
    // sin(x - 2pi) = sin(x)
    // therefore
    // cos(2pi k / N) = cos(2pi k / N - 2pi) = cos(2pi (k - N) / N)
    if (2 * k > N) {
        k -= N;
    } else if (2 * k < -N) {
        k += N;
    }

    return {k, N};
}

std::complex<double> power_of_w(int k, int N) {
    auto [kk, NN] = simplify_power_of_w(k, N);
    k = kk;
    N = NN;
    auto w_lut = get_w_lut(N);
    if (w_lut) {
        int k_pos = k < 0 ? -k : k;
        auto w = w_lut[k_pos];
        double re = w.real();
        double im = w.imag();
        if (k < 0) {
            im *= -1.0;
        }
        return {re, im};
    }
    constexpr double tau = 6.28318530717958647693;
    auto arg = (tau / N) * k;
    return {std::cos(arg), std::sin(arg)};
}

} // namespace bbfft
