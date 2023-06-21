// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"

#include <bbfft/configuration.hpp>
#include <bbfft/detail/compiler_options.hpp>
#include <bbfft/device_info.hpp>
#include <bbfft/generator.hpp>
#include <bbfft/ze/online_compiler.hpp>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
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

    auto kernel_file = std::ofstream(a.kernel_filename, std::ios::binary);
    if (!kernel_file) {
        std::cerr << "==> Could not open " << a.kernel_filename << " for writing." << std::endl;
        return -1;
    }

    std::ostringstream oss;
    auto kernel_names = generate_fft_kernels(oss, a.configurations, a.info);

    try {
        auto bin = a.format == module_format::native
                       ? ze::compile_to_native(oss.str(), a.device, detail::compiler_options)
                       : ze::compile_to_spirv(oss.str(), detail::compiler_options);
        kernel_file.write(reinterpret_cast<char *>(bin.data()), bin.size());
    } catch (std::exception const &e) {
        std::cerr << "==> Could not compile FFT kernels." << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
