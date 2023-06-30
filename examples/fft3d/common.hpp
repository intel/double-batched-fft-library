// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
#include "args.hpp"
#include "utility.hpp"

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>

#include <iostream>
#include <utility>

template <typename Harness, typename T> void test_runtime(args a) {
    constexpr bool r2c = !is_complex_v<T>;

    auto h = Harness{};

    std::cout << a.N[0] << " x " << a.N[1] << " x " << a.N[2] << std::endl;

    std::size_t N1_out = r2c ? a.N[0] / 2 + 1 : a.N[0];
    std::size_t N1_in = r2c ? 2 * N1_out : N1_out;
    std::size_t size = N1_in * a.N[1] * a.N[2];
    auto x = new T[size];
    auto x_device = h.template malloc_device<T>(size);
    auto X_device = a.inplace ? x_device : h.template malloc_device<T>(size);

    init(x, a.N[0], a.N[1], a.N[2], a.inplace);
    h.copy(x, x_device, size);

    bbfft::configuration cfg = {3,
                                {1, a.N[0], a.N[1], a.N[2], 1},
                                bbfft::to_precision_v<T>,
                                bbfft::direction::forward,
                                r2c ? bbfft::transform_type::r2c : bbfft::transform_type::c2c};
    cfg.set_strides_default(a.inplace);
    h.setup_plan(cfg);
    std::uint32_t ntimes = 1;
    auto const execute3d = [&]() { h.run_plan(x_device, X_device, ntimes); };

    if (a.verbose) {
        std::cout << "Check" << std::endl;
    }
    bench(1, execute3d, a.verbose);

    h.copy(X_device, x, size);
    if (check(x, a.N[0], a.N[1], a.N[2])) {
        if (a.verbose) {
            std::cout << "Bench (" << a.nrepeat << "x)" << std::endl;
        }
        ntimes = a.nrepeat;
        auto time = bench(10, execute3d, a.verbose);
        time /= ntimes;
        std::cout << time << " s, " << 2 * sizeof(std::complex<T>) * size / time * 1.0e-9 << " GB/s"
                  << std::endl;
    }

    h.free(x_device);
    if (!a.inplace) {
        h.free(X_device);
    }
    delete[] x;
}
