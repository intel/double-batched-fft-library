// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"

#include <bbfft/bad_configuration.hpp>
#include <bbfft/configuration.hpp>
#include <bbfft/generator.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

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

    try {
        generate_fft_kernels(std::cout, a.configurations, a.info);
    } catch (bbfft::bad_configuration const &e) {
        std::cerr << "==> Bad configuration: " << e.what() << std::endl;
        return -1;
    } catch (std::exception const &e) {
        std::cerr << "==> Could not compile FFT kernels." << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
