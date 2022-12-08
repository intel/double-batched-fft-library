// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"
#include "fft.hpp"

#include "doctest/doctest.h"

using namespace bbfft;

TEST_CASE("unsupported fft dim") {
    auto Q = sycl::queue();

    auto cfg = configuration{};
    CHECK_THROWS_AS((make_plan(cfg, Q)), bad_configuration);

    cfg.dim = max_fft_dim + 1;
    CHECK_THROWS_AS((make_plan(cfg, Q)), bad_configuration);
}
