// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_SIGNAL_20240610_HPP
#define TEST_SIGNAL_20240610_HPP

#include "refdft.hpp"

#include "bbfft/configuration.hpp"

#include <iosfwd>
#include <memory>

class test_bench_1d {
  public:
    test_bench_1d(bbfft::configuration const &cfg, refdft_factory const &factory);
    void signal(void *x) const;
    bool check(void *x, std::ostream *os = nullptr) const;

  private:
    bbfft::configuration cfg_;
    std::unique_ptr<refdft> ref_;
};

#endif // TEST_SIGNAL_20240610_HPP
