// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "scrambler.hpp"

#include "doctest/doctest.h"
#include <functional>
#include <numeric>
#include <vector>

using namespace bbfft;

TEST_CASE("scrambler") {
    SUBCASE("manual") {
        auto factorization = std::vector<int>{3, 2, 2};
        auto p = scrambler(factorization);
        auto P = unscrambler(factorization);

        CHECK(p(0) == 0);
        CHECK(p(1) == 4);
        CHECK(p(2) == 8);
        CHECK(p(3) == 2);
        CHECK(p(4) == 6);
        CHECK(p(5) == 10);
        CHECK(p(6) == 1);
        CHECK(p(7) == 5);
        CHECK(p(8) == 9);
        CHECK(p(9) == 3);
        CHECK(p(10) == 7);
        CHECK(p(11) == 11);
        CHECK(p(12) == 12);
        CHECK(p(13) == 16);

        CHECK(0 == P(0));
        CHECK(1 == P(4));
        CHECK(2 == P(8));
        CHECK(3 == P(2));
        CHECK(4 == P(6));
        CHECK(5 == P(10));
        CHECK(6 == P(1));
        CHECK(7 == P(5));
        CHECK(8 == P(9));
        CHECK(9 == P(3));
        CHECK(10 == P(7));
        CHECK(11 == P(11));
        CHECK(12 == P(12));
        CHECK(13 == P(16));
    }

    SUBCASE("identity") {
        auto factorization = std::vector<int>{13, 5, 2, 7};
        int N = std::reduce(factorization.begin(), factorization.end(), 1, std::multiplies<int>{});
        auto p = scrambler(factorization);
        auto P = unscrambler(factorization);
        for (int i = 0; i < 2 * N; ++i) {
            CHECK(p(P(i)) == i);
            CHECK(P(p(i)) == i);
        }
    }
}
