// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "fft3d.hpp"

#include <complex>
#include <iostream>

int main(int argc, char **argv) {
    try {
        args a = parse_args(argc, argv);

        if (a.r2c) {
            std::cout << "r2c ";
            if (a.double_precision) {
                std::cout << "double" << std::endl;
                test<double>(a);
            } else {
                std::cout << "single" << std::endl;
                test<float>(a);
            }
        } else {
            std::cout << "c2c ";
            if (a.double_precision) {
                std::cout << "double" << std::endl;
                test<std::complex<double>>(a);
            } else {
                std::cout << "single" << std::endl;
                test<std::complex<float>>(a);
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Error: Could not parse command line." << std::endl;
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
