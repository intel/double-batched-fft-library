// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ARGS_20220428_H
#define ARGS_20220428_H

#include <functional>
#include <vector>

struct args {
    bool inplace;
    bool inverse;
    char p;
    char d;
    std::vector<unsigned int> MM;
    std::vector<unsigned int> NN;
    std::function<unsigned int(unsigned int, unsigned int)> KK;
};

args parse_args(int argc, char **argv);

#endif // ARGS_20220428_H
