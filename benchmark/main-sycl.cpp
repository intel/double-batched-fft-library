// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"

#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>

void test(sycl::queue Q, args const &a);

int main(int argc, char **argv) {
    try {
        auto a = parse_args(argc, argv);

        auto handle_async_error = [](sycl::exception_list elist) {
            for (std::exception_ptr const &e : elist) {
                try {
                    std::rethrow_exception(e);
                } catch (sycl::exception const &e) {
                    std::cout << e.what() << std::endl << std::flush;
                }
            }
        };

        auto Q = sycl::queue(handle_async_error);
        test(std::move(Q), a);
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Error: Could not parse command line." << std::endl;
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
