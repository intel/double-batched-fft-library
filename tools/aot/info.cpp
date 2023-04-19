// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "info.hpp"

using namespace bbfft;

const auto builtin_device_info = std::unordered_map<std::string, device_info>{
    {"pvc", {1024, {16, 32}, 128 * 1024, device_type::gpu}}};
