// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_20220425_H
#define TEST_20220425_H

#include <cstddef>

class K_fixed {
  public:
    K_fixed(unsigned long K) : K(K) {}
    inline auto operator()(unsigned int, unsigned int) { return K; }

  private:
    unsigned long K;
};

class K_memory_limit {
  public:
    K_memory_limit(std::size_t bytes, char p, char d) : bytes(bytes) {
        number_size = p == 'd' ? sizeof(double) : sizeof(float);
        if (d == 'c') {
            number_size *= 2;
        }
    }
    inline auto operator()(unsigned int M, unsigned int N) {
        unsigned int k = bytes / (M * N * number_size);
        return k > 0 ? k : 1u;
    };

  private:
    std::size_t bytes;
    std::size_t number_size;
};

#endif // TEST_20220425_H
