// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ARGS_20240610_HPP
#define ARGS_20240610_HPP

#include "bbfft/configuration.hpp"

#include <iosfwd>
#include <vector>

struct args {
    bool help;
    double min_time;
    std::vector<bbfft::configuration> cfgs;
};

class arg_parser {
  public:
    args parse_args(int argc, char **argv);
    void show_help(std::ostream &os);
};

#endif // ARGS_20240610_HPP
