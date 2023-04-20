// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/parser.hpp"
#include "bbfft/device_info.hpp"

#include "doctest/doctest.h"

#include <sstream>

using namespace bbfft;

TEST_CASE("device info") {
    auto ginfo = device_info{1024, {16, 32}, 128 * 1024, device_type::gpu};
    auto cinfo = device_info{8192, {4, 8, 16, 32, 64}, 32768, device_type::cpu};

    CHECK(parse_device_info(ginfo.to_string()) == ginfo);
    CHECK(parse_device_info(cinfo.to_string()) == cinfo);
}
