// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/tensor_indexer.hpp"

#include "doctest/doctest.h"

#include <cstddef>

using namespace bbfft;

TEST_CASE("tensor indexer") {
    SUBCASE("col major") {
        SUBCASE("packed") {
            auto x = tensor_indexer<std::size_t, 3u, layout::col_major>({3, 4, 5});
            CHECK(x(0, 0, 0) == 0);
            CHECK(x(1, 0, 0) == 1);
            CHECK(x(0, 1, 0) == 3);
            CHECK(x(0, 0, 1) == 12);
            CHECK(x(2, 2, 2) == 2 + 2 * 3 + 2 * 12);
            CHECK(x(2, 3, 4) == x.size() - 1);
            CHECK(x.size() == 60);
        }
        SUBCASE("strided") {
            auto x = tensor_indexer<std::size_t, 3u, layout::col_major>({3, 4, 5}, {1, 5, 30});
            CHECK(x(0, 0, 0) == 0);
            CHECK(x(1, 0, 0) == 1);
            CHECK(x(0, 1, 0) == 5);
            CHECK(x(0, 0, 1) == 30);
            CHECK(x(2, 2, 2) == 2 + 2 * 5 + 2 * 30);
            CHECK(x(2, 3, 4) == 137);
            CHECK(x.size() == 150);
        }
        SUBCASE("may fuse") {
            auto x1 = tensor_indexer<std::size_t, 5u, layout::col_major>({2, 3, 4, 5, 6});
            auto x2 = tensor_indexer<std::size_t, 5u, layout::col_major>({2, 3, 4, 5, 6},
                                                                         {1, 3, 9, 36, 216});
            CHECK(x1.may_fuse() == true);
            CHECK(x1.may_fuse<1, 3>() == true);
            CHECK(x1.may_fuse<4>() == true);
            CHECK(x2.may_fuse() == false);
            CHECK(x2.may_fuse<1, 3>() == true);
            CHECK(x2.may_fuse<2, 3>() == true);
            CHECK(x2.may_fuse<1, 4>() == false);
        }
        SUBCASE("fused") {
            auto x1 = tensor_indexer<std::size_t, 5u, layout::col_major>({2, 3, 4, 5, 6});
            auto f1 = x1.fused();
            auto f2 = x1.fused<1, 3>();
            REQUIRE(f1.dim() == 1);
            CHECK(f1.shape(0) == 720);
            CHECK(f1.stride(0) == 1);
            REQUIRE(f2.dim() == 3);
            CHECK(f2.shape(0) == 2);
            CHECK(f2.shape(1) == 60);
            CHECK(f2.shape(2) == 6);
            CHECK(f2.stride(0) == 1);
            CHECK(f2.stride(1) == 2);
            CHECK(f2.stride(2) == 120);
            CHECK(f2(1, 35, 3) == x1(1, 2, 3, 2, 3));
        }
    }
    SUBCASE("row major") {
        SUBCASE("packed") {
            auto x = tensor_indexer<std::size_t, 3u, layout::row_major>({3, 4, 5});
            CHECK(x(0, 0, 0) == 0);
            CHECK(x(1, 0, 0) == 20);
            CHECK(x(0, 1, 0) == 5);
            CHECK(x(0, 0, 1) == 1);
            CHECK(x(2, 2, 2) == 2 * 20 + 2 * 5 + 2);
            CHECK(x(2, 3, 4) == x.size() - 1);
            CHECK(x.size() == 60);
        }
        SUBCASE("strided") {
            auto x = tensor_indexer<std::size_t, 3u, layout::row_major>({3, 4, 5}, {30, 6, 1});
            CHECK(x(0, 0, 0) == 0);
            CHECK(x(1, 0, 0) == 30);
            CHECK(x(0, 1, 0) == 6);
            CHECK(x(0, 0, 1) == 1);
            CHECK(x(2, 2, 2) == 2 * 30 + 2 * 6 + 2);
            CHECK(x(2, 3, 4) == 82);
            CHECK(x.size() == 90);
        }
        SUBCASE("may fuse") {
            auto x1 = tensor_indexer<std::size_t, 5u, layout::row_major>({6, 5, 4, 3, 2});
            auto x2 = tensor_indexer<std::size_t, 5u, layout::row_major>({6, 5, 4, 3, 2},
                                                                         {216, 36, 9, 3, 1});
            CHECK(x1.may_fuse() == true);
            CHECK(x1.may_fuse<1, 3>() == true);
            CHECK(x1.may_fuse<4>() == true);
            CHECK(x2.may_fuse() == false);
            CHECK(x2.may_fuse<1, 3>() == true);
            CHECK(x2.may_fuse<2, 3>() == true);
            CHECK(x2.may_fuse<1, 4>() == false);
        }
    }
}
