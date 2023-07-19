// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/tensor_indexer.hpp"

#include "doctest/doctest.h"

#include <cstddef>
#include <ostream>

using namespace bbfft;

namespace std {
template <typename T, std::size_t D>
std::ostream &operator<<(std::ostream &os, const std::array<T, D> &array) {
    auto it = array.begin();
    os << "{" << *it++;
    for (; it != array.end(); ++it) {
        os << ", " << *it;
    }
    return os << "}";
}
} // namespace std

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
            CHECK(f2.shape() == std::array<std::size_t, 3u>{2, 60, 6});
            CHECK(f2.stride() == std::array<std::size_t, 3u>{1, 2, 120});
            CHECK(f2(1, 35, 3) == x1(1, 2, 3, 2, 3));
        }
        SUBCASE("may reshape mode") {
            auto x1 = tensor_indexer<std::size_t, 4u, layout::col_major>({10, 8, 5, 10},
                                                                         {1, 12, 120, 500});
            CHECK(x1.may_reshape_mode(1, std::array<std::size_t, 3>{2, 2, 2}) == true);
            CHECK(x1.may_reshape_mode(1, std::array<std::size_t, 3>{2, 3, 2}) == false);
        }
        SUBCASE("reshape mode") {
            auto x1 = tensor_indexer<std::size_t, 4u, layout::col_major>({10, 8, 5, 10},
                                                                         {1, 12, 120, 500});
            auto r1 = x1.reshape_mode(0, std::array<std::size_t, 2>{2, 5});
            CHECK(r1.dim() == 5u);
            CHECK(r1.shape() == std::array<std::size_t, 5u>{2, 5, 8, 5, 10});
            CHECK(r1.stride() == std::array<std::size_t, 5u>{1, 2, 12, 120, 500});
            auto r2 = x1.reshape_mode(1, std::array<std::size_t, 3>{2, 2, 2});
            CHECK(r2.dim() == 6u);
            CHECK(r2.shape() == std::array<std::size_t, 6u>{10, 2, 2, 2, 5, 10});
            CHECK(r2.stride() == std::array<std::size_t, 6u>{1, 12, 24, 48, 120, 500});
            auto r3 = x1.reshape_mode(3, std::array<std::size_t, 2>{5, 2});
            CHECK(r3.dim() == 5u);
            CHECK(r3.shape() == std::array<std::size_t, 5u>{10, 8, 5, 5, 2});
            CHECK(r3.stride() == std::array<std::size_t, 5u>{1, 12, 120, 500, 2500});
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
        SUBCASE("fused") {
            auto x1 = tensor_indexer<std::size_t, 5u, layout::row_major>({6, 5, 4, 3, 2});
            auto f1 = x1.fused();
            auto f2 = x1.fused<1, 3>();
            REQUIRE(f1.dim() == 1);
            CHECK(f1.shape(0) == 720);
            CHECK(f1.stride(0) == 1);
            REQUIRE(f2.dim() == 3);
            CHECK(f2.shape() == std::array<std::size_t, 3u>{6, 60, 2});
            CHECK(f2.stride() == std::array<std::size_t, 3u>{120, 2, 1});
            CHECK(f2(1, 35, 3) == x1(1, 2, 3, 2, 3));
        }
        SUBCASE("reshape mode") {
            auto x1 = tensor_indexer<std::size_t, 4u, layout::row_major>({10, 8, 5, 10});
            auto r1 = x1.reshape_mode(0, std::array<std::size_t, 2>{2, 5});
            CHECK(r1.dim() == 5u);
            CHECK(r1.shape() == std::array<std::size_t, 5u>{2, 5, 8, 5, 10});
            CHECK(r1.stride() == std::array<std::size_t, 5u>{2000, 400, 50, 10, 1});
            auto r2 = x1.reshape_mode(1, std::array<std::size_t, 3>{2, 2, 2});
            CHECK(r2.dim() == 6u);
            CHECK(r2.shape() == std::array<std::size_t, 6u>{10, 2, 2, 2, 5, 10});
            CHECK(r2.stride() == std::array<std::size_t, 6u>{400, 200, 100, 50, 10, 1});
            auto r3 = x1.reshape_mode(3, std::array<std::size_t, 2>{5, 2});
            CHECK(r3.dim() == 5u);
            CHECK(r3.shape() == std::array<std::size_t, 5u>{10, 8, 5, 5, 2});
            CHECK(r3.stride() == std::array<std::size_t, 5u>{400, 50, 10, 2, 1});
        }
    }
}
