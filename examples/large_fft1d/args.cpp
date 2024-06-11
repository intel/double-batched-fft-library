// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "bbfft/parser.hpp"

#include <cstdlib>
#include <cstring>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

args arg_parser::parse_args(int argc, char **argv) {
    args a = {};
    a.min_time = 1.0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            auto const fail = [&]() {
                throw std::runtime_error("==> Error: unrecognized argument " +
                                         std::string(argv[i]));
            };
            if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
                a.help = true;
            } else if (i + 1 < argc) {
                if (std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "--min-time") == 0) {
                    a.min_time = std::atof(argv[++i]);
                } else {
                    fail();
                }
            } else {
                fail();
            }
        } else {
            a.cfgs.emplace_back(bbfft::parse_fft_descriptor(argv[i]));
        }
    }
    if (a.help) {
        return a;
    }

    return a;
}

void arg_parser::show_help(std::ostream &os) {
    os << "usage: large_fft1d fft-descriptor1 fft-descriptor2 ..." << std::endl
       << std::endl
       << R"HELP(
positional arguments:
    fft-descriptorN     1D FFT descriptor (see docs/manual/descriptor.rst)

optional arguments:
    -h, --help          Show help and quit
    -t, --min-time      Minimum test duration in seconds
)HELP";
}
