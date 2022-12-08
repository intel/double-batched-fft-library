// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef RESULT20220221_H
#define RESULT20220221_H

#include <string>
#include <vector>

struct result {
    std::string precision;
    unsigned int N;
    unsigned int inplace;
    unsigned int M;
    unsigned int K;
    std::string domain;
    bool inverse;
    double time;
    double bandwidth;
    double flops;
};

std::vector<std::string> column_names() {
    return {"p", "N", "inplace", "M", "K", "domain", "inverse", "time", "bw", "flops"};
}

template <typename Printer> void print(result const &r, Printer &p) {
    p << r.precision << r.N << r.inplace << r.M << r.K << r.domain << r.inverse << r.time
      << r.bandwidth << r.flops;
}

#endif // RESULT20220221_H
