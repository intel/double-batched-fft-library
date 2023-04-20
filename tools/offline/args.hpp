// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
#ifndef ARGS_20230420_HPP
#define ARGS_20230420_HPP

#include <bbfft/configuration.hpp>
#include <bbfft/device_info.hpp>

#include <iosfwd>
#include <string>
#include <vector>

struct args {
    std::vector<bbfft::configuration> configurations;
    bool help;
    std::string device;
    bbfft::device_info info;
};

args parse_args(int argc, char **argv);
void show_help(std::ostream &os);

#endif // ARGS_20230420_HPP
