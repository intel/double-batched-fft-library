// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/generator.hpp>

#include <exception>
#include <iostream>

using namespace bbfft;

int main(int argc, char **argv) {
    args a = {};
    try {
        a = parse_args(argc, argv);
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl << std::endl;
        show_help(std::cerr);
        return -1;
    }

    if (a.help) {
        show_help(std::cout);
        return 0;
    }

    generate_fft_kernels(std::cout, a.configurations, a.info);

    return 0;
}
