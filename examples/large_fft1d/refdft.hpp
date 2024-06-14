// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REFDFT_20240614_HPP
#define REFDFT_20240614_HPP

#include "bbfft/configuration.hpp"

#include <memory>

class refdft {
  public:
    virtual ~refdft() = default;
    virtual void x(long n, long k, void *val) const = 0;
    virtual void X(long n, long k, void *val) const = 0;
    virtual double Linf() const = 0;
};

class refdft_factory {
  public:
    virtual auto make_ref(bbfft::precision fp, bbfft::transform_type type, long N, long K) const
        -> std::unique_ptr<refdft> = 0;
};

#endif // REFDFT_20240614_HPP
