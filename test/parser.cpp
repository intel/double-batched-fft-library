// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/parser.hpp"
#include "bbfft/device_info.hpp"

#include "doctest/doctest.h"

#include <sstream>

using namespace bbfft;

template <typename... T> auto make_shape(T... value) -> std::array<std::size_t, max_tensor_dim> {
    return {static_cast<std::size_t>(value)...};
}

auto default_istride(configuration const &cfg, bool inplace) {
    return default_istride(cfg.dim, cfg.shape, cfg.type, inplace);
}

auto default_ostride(configuration const &cfg, bool inplace) {
    return default_ostride(cfg.dim, cfg.shape, cfg.type, inplace);
}

TEST_CASE("device info") {
    auto ginfo = device_info{1024, {16, 32}, 128 * 1024, device_type::gpu};
    auto cinfo = device_info{8192, {4, 8, 16, 32, 64}, 32768, device_type::cpu};

    CHECK(parse_device_info(ginfo.to_string()) == ginfo);
    CHECK(parse_device_info(cinfo.to_string()) == cinfo);
}

TEST_CASE("fft descriptor parser") {
    std::string desc;
    auto cfg = parse_fft_descriptor(desc = "srfi5");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(1, 5, 1));
    CHECK(cfg.fp == precision::f32);
    CHECK(cfg.dir == direction::forward);
    CHECK(cfg.type == transform_type::r2c);
    CHECK(cfg.istride == default_istride(cfg, true));
    CHECK(cfg.ostride == default_ostride(cfg, true));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "dcbi4*5");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(1, 4, 5));
    CHECK(cfg.fp == precision::f64);
    CHECK(cfg.dir == direction::backward);
    CHECK(cfg.type == transform_type::c2c);
    CHECK(cfg.istride == default_istride(cfg, true));
    CHECK(cfg.ostride == default_ostride(cfg, true));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "dcbi4.5");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(4, 5, 1));
    CHECK(cfg.fp == precision::f64);
    CHECK(cfg.dir == direction::backward);
    CHECK(cfg.type == transform_type::c2c);
    CHECK(cfg.istride == default_istride(cfg, true));
    CHECK(cfg.ostride == default_ostride(cfg, true));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "drbo4.5*6");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(4, 5, 6));
    CHECK(cfg.fp == precision::f64);
    CHECK(cfg.dir == direction::backward);
    CHECK(cfg.type == transform_type::c2r);
    CHECK(cfg.istride == default_istride(cfg, false));
    CHECK(cfg.ostride == default_ostride(cfg, false));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "drfo5x6x7");
    CHECK(cfg.dim == 3);
    CHECK(cfg.shape == make_shape(1, 5, 6, 7, 1));
    CHECK(cfg.fp == precision::f64);
    CHECK(cfg.dir == direction::forward);
    CHECK(cfg.type == transform_type::r2c);
    CHECK(cfg.istride == default_istride(cfg, false));
    CHECK(cfg.ostride == default_ostride(cfg, false));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "srbo4.5x6*7");
    CHECK(cfg.dim == 2);
    CHECK(cfg.shape == make_shape(4, 5, 6, 7));
    CHECK(cfg.fp == precision::f32);
    CHECK(cfg.dir == direction::backward);
    CHECK(cfg.type == transform_type::c2r);
    CHECK(cfg.istride == default_istride(cfg, false));
    CHECK(cfg.ostride == default_ostride(cfg, false));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "scfo16*32i1,1,20");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(1, 16, 32));
    CHECK(cfg.fp == precision::f32);
    CHECK(cfg.dir == direction::forward);
    CHECK(cfg.type == transform_type::c2c);
    CHECK(cfg.istride == make_shape(1, 1, 20));
    CHECK(cfg.ostride == default_ostride(cfg, false));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "scfo16*32i1,1,20o1,2,32");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(1, 16, 32));
    CHECK(cfg.fp == precision::f32);
    CHECK(cfg.dir == direction::forward);
    CHECK(cfg.type == transform_type::c2c);
    CHECK(cfg.istride == make_shape(1, 1, 20));
    CHECK(cfg.ostride == make_shape(1, 2, 32));
    CHECK(cfg.to_string() == desc);

    cfg = parse_fft_descriptor(desc = "scfi16*32i1,1,20o1,1,20");
    CHECK(cfg.dim == 1);
    CHECK(cfg.shape == make_shape(1, 16, 32));
    CHECK(cfg.fp == precision::f32);
    CHECK(cfg.dir == direction::forward);
    CHECK(cfg.type == transform_type::c2c);
    CHECK(cfg.istride == make_shape(1, 1, 20));
    CHECK(cfg.ostride == make_shape(1, 1, 20));
    CHECK(cfg.to_string() == desc);

    CHECK_THROWS_AS((parse_fft_descriptor("srb4x5*6x7")), std::runtime_error);
}
