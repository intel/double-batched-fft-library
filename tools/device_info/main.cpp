// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <bbfft/device_info.hpp>
#include <bbfft/sycl/device.hpp>

#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
    auto q = sycl::queue{};
    auto info = bbfft::get_device_info(q.get_device());
    std::cout << info << std::endl;

    return 0;
}
