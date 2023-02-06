// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "configurations.hpp"

using namespace bbfft;

std::vector<configuration> configurations() {
    configuration cfg_template = {
        1, {1, 2, 16384}, precision::f32, direction::forward, transform_type::r2c};
    auto cfgs = std::vector<configuration>{};
    for (unsigned int i = 2; i < 1024; i *= 2) {
        cfg_template.shape[1] = i;
        cfg_template.set_strides_default(true);
        cfgs.push_back(cfg_template);
    }
    return cfgs;
}
