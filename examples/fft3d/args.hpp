// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ARGS_20220713_HPP
#define ARGS_20220713_HPP

#include <array>
#include <cstddef>

struct args {
    std::array<std::size_t, 3> N;
    bool inplace;
    bool double_precision;
    bool r2c;
    bool verbose;
};

args parse_args(int argc, char **argv);

#endif // ARGS_20220713_HPP
