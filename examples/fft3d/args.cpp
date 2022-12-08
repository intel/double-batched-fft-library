// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"

#include <cstdlib>
#include <stdexcept>

args parse_args(int argc, char **argv) {
    args a = {};
    a.inplace = true;
    a.double_precision = false;
    a.r2c = true;
    a.verbose = false;

    auto const help = []() {
        return
            R"HLP(Usage: ./fft3d [-iodscr] <N1> <N2> <N3>

Options:
-i/o    in-place / out-of-place [default: in-place]
-d/s    double / single precision [default: single]
-c/r    c2c / r2c [default: r2c]
-v      verbose
)HLP";
    };

    int positional_arg = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            for (char *opt = &argv[i][1]; *opt != 0; ++opt) {
                switch (*opt) {
                case 'i':
                    a.inplace = true;
                    break;
                case 'o':
                    a.inplace = false;
                    break;
                case 'd':
                    a.double_precision = true;
                    break;
                case 's':
                    a.double_precision = false;
                    break;
                case 'c':
                    a.r2c = false;
                    break;
                case 'r':
                    a.r2c = true;
                    break;
                case 'v':
                    a.verbose = true;
                    break;
                default:
                    throw std::invalid_argument(help());
                    break;
                }
            }
        } else if (positional_arg < 3) {
            a.N[positional_arg] = atoi(argv[i]);
            ++positional_arg;
        } else {
            throw std::invalid_argument(help());
        }
    }
    if (positional_arg != 3) {
        throw std::invalid_argument(help());
    }
    return a;
}
