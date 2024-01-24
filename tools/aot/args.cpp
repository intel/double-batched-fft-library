// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "info.hpp"

#include "bbfft/parser.hpp"

#include <cstdlib>
#include <cstring>
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace bbfft;

args parse_args(int argc, char **argv) {
    args a = {};
    a.format = module_format::native;
    a.info = {};
    a.device = {};

    int positional_arg = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            auto const fail = [&]() {
                throw std::runtime_error("==> Error: unrecognized argument " +
                                         std::string(argv[i]));
            };
            if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
                a.help = true;
            } else if (i + 1 < argc) {
                if (std::strcmp(argv[i], "-f") == 0 || std::strcmp(argv[i], "--format") == 0) {
                    ++i;
                    if (std::strcmp(argv[i], "native") == 0) {
                        a.format = module_format::native;
                    } else if (std::strcmp(argv[i], "spirv") == 0) {
                        a.format = module_format::spirv;
                    } else {
                        fail();
                    }
                } else if (std::strcmp(argv[i], "-d") == 0 ||
                           std::strcmp(argv[i], "--device") == 0) {
                    ++i;
                    a.device = std::string(argv[i]);
                } else if (std::strcmp(argv[i], "-i") == 0 ||
                           std::strcmp(argv[i], "--device_info") == 0) {
                    ++i;
                    a.info = parse_device_info(argv[i]);
                } else {
                    fail();
                }
            } else {
                fail();
            }
        } else if (positional_arg == 0) {
            a.kernel_filename = std::string(argv[i]);
            ++positional_arg;
        } else {
            a.configurations.emplace_back(parse_fft_descriptor(argv[i]));
            ++positional_arg;
        }
    }
    if (positional_arg == 0) {
        throw std::invalid_argument("==> You need to provide the kernel file name");
    }
    if (a.configurations.empty()) {
        throw std::invalid_argument("==> You need to provide at least one FFT desciptor");
    }
    if (a.format == module_format::native && a.device.empty()) {
        throw std::invalid_argument("==> Device must be provided when using native format");
    }
    if (a.info.max_work_group_size == 0) {
        if (auto it = builtin_device_info.find(a.device); it != builtin_device_info.end()) {
            a.info = it->second;
        } else {
            throw std::invalid_argument("Device info missing for device \"" + a.device +
                                        "\". You need to provide device info via --device_info.");
        }
    }
    return a;
}

void show_help(std::ostream &os) {
    os << R"HELP(Usage: bbfft-aot-generate <kernel_filename> <fft-descriptor1> <fft-descriptor2> ...

Tool to compile FFT kernels ahead-of-time and store native device code in a binary blob.

example:
    bbfft-aot-generate -d pvc kernels.bin scfi16*1000

positional arguments:
    kernel_filename     Path to output file (e.g. kernels.bin)
    fft-descriptorN     FFT descriptor

optional arguments:
    -h, --help          Show help and quit
    -f, --format        native or spirv (default: native)
    -d, --device        Target device
    -i, --device_info   Device info
)HELP";
}
