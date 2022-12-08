// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/generator.hpp>

#include <iostream>

using namespace bbfft;

int main() {
    configuration cfg = {
        1, {8, 16, 1024}, precision::f32, direction::backward, transform_type::r2c};
    device_info info = {1024, {8}, 1, 128 * 1024};

    auto sbc = configure_small_batch_fft(cfg, info);
    generate_small_batch_fft(std::cout, "fft", sbc);
    // auto f2c = configure_factor2_slm_fft(cfg, info);
    // generate_factor2_slm_fft<float>(std::cout, "fft", f2c);

    return 0;
}
