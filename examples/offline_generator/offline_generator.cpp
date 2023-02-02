// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/generator.hpp>

#include <iostream>

using namespace bbfft;

int main() {
    configuration cfg = {
        2, {1, 4, 6, 1024}, precision::f32, direction::backward, transform_type::r2c};
    device_info info = {1024, {16, 32}, 2, 128 * 1024};

    generate_fft_kernels(std::cout, {cfg}, info);

    return 0;
}
