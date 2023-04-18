// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "test.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

args parse_args(int argc, char **argv) {
    args a = {};
    a.inplace = true;
    a.inverse = false;
    a.p = 's';
    a.d = 'r';

    auto const help = []() {
        return R"HLP(Usage: ./test [-i/o] [-v] [-m <M>] [-k <K>] [-b <B>] (s|d)(r|c) <N1> <N2> ...";

Performs 1D FFT on MxNxK tensor where the FFT is taken over the second mode.

Options:
-i/o    in-place / out-of-place [default: in-place]
-v      backward transform [default: forward]
-m      Size of first mode; repeat -m to test multiple sizes
-k      Fixed size of third mode
-b      Choose size of third mode based on memory footprint of the input tensor.
        Supports common suffixes (k/K = kilo, m/M = mega, g/G = giga).
        Ignored if -k is given.
s/d     double / single precision
r/c     real / complex
Nx      Size of second mode; multiple sizes supported
)HLP";
    };

    auto const parse_bytes = [](std::string b) {
        std::size_t s = 1;
        switch (b.back()) {
        case 'k':
        case 'K':
            s = 1024;
            break;
        case 'm':
        case 'M':
            s = 1024 * 1024;
            break;
        case 'g':
        case 'G':
            s = 1024 * 1024 * 1024;
            break;
        default:
            s = 1;
            break;
        }
        if (s != 1) {
            b.pop_back();
        }
        return std::stoul(b) * s;
    };

    std::size_t bytes = 0;
    int positional_arg = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 'i':
                a.inplace = true;
                break;
            case 'o':
                a.inplace = false;
                break;
            case 'v':
                a.inverse = true;
                break;
            default:
                if (i + 1 < argc) {
                    switch (argv[i][1]) {
                    case 'm':
                        a.MM.emplace_back(std::stoi(argv[++i]));
                        break;
                    case 'k':
                        a.KK = K_fixed{std::stoul(argv[++i])};
                        break;
                    case 'b':
                        bytes = parse_bytes(argv[++i]);
                        break;
                    default:
                        throw std::invalid_argument(help());
                    }
                } else {
                    throw std::invalid_argument(help());
                }
                break;
            }
        } else if (positional_arg == 0) {
            a.p = argv[i][0];
            if (argv[i][1] != '\0') {
                a.d = argv[i][1];
            }
            ++positional_arg;
        } else {
            a.NN.emplace_back(std::stoi(argv[i]));
        }
    }
    if (a.MM.empty()) {
        a.MM.emplace_back(1);
    }
    if (!a.KK) {
        if (bytes == 0) {
            bytes = 1024 * 1024 * 1024;
        }
        a.KK = K_memory_limit{bytes, a.p, a.d};
    }
    return a;
}
