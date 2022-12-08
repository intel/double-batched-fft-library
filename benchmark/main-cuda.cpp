// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include <stdexcept>

#include <iostream>

void test(args const &a);

int main(int argc, char **argv) {
    try {
        auto a = parse_args(argc, argv);
        test(a);
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Error: Could not parse command line." << std::endl;
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
