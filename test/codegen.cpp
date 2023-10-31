// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"
#include "bbfft/detail/generator_impl.hpp"
#include "math.hpp"
#include "prime_factorization.hpp"
#include "scrambler.hpp"

#include "doctest/doctest.h"
#include <vector>

using namespace bbfft;

TEST_CASE("scrambler") {
    SUBCASE("manual") {
        auto factorization = std::vector<int>{3, 2, 2};
        auto p = scrambler(factorization);
        auto P = unscrambler(factorization);

        p.in0toN(true);
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
        p.in0toN(false);
        CHECK(p(12) == 12);
        CHECK(p(13) == 16);

        P.in0toN(true);
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
        P.in0toN(false);
        CHECK(12 == P(12));
        CHECK(13 == P(16));
    }

    SUBCASE("identity") {
        auto factorization = std::vector<int>{13, 5, 2, 7};
        int N = 1;
        for (auto f : factorization) {
            N *= f;
        }
        auto p = scrambler(factorization);
        auto P = unscrambler(factorization);
        for (int i = 0; i < 2 * N; ++i) {
            CHECK(p(P(i)) == i);
            CHECK(P(p(i)) == i);
        }
    }
}

TEST_CASE("math") {
    SUBCASE("ipow") {
        CHECK(ipow(3, 0) == 1);
        CHECK(ipow(5, 1) == 5);
        CHECK(ipow(7, 2) == 49);
        CHECK(ipow(11, 3) == 1331);
    }

    SUBCASE("iroot") {
        CHECK(iroot(0, 5) == 0);
        CHECK(iroot(3, 1) == 3);
        CHECK(iroot(25, 2) == 5);
        CHECK(iroot(1331, 3) == 11);
        CHECK(iroot(1337, 3) == 11);
        CHECK(iroot(4096, 3) == 16);
        CHECK(iroot(64539, 2) == 254);
        CHECK(iroot(64539, 3) == 40);
        CHECK(iroot(64539, 4) == 15);
    }

    SUBCASE("is_prime") {
        CHECK(!is_prime(1));
        CHECK(is_prime(2));
        CHECK(is_prime(13));
        CHECK(!is_prime(25));
        CHECK(is_prime(2689));
    }
}

TEST_CASE("prime factorization") {
    CHECK(factor(4096, 0) == std::vector<unsigned>{});
    CHECK(factor(4096, 1) == std::vector<unsigned>{4096});
    CHECK(factor(4096, 2) == std::vector<unsigned>{64, 64});
    CHECK(factor(4096, 3) == std::vector<unsigned>{16, 16, 16});

    CHECK(factor(2080, 2) == std::vector<unsigned>{40, 52});
    CHECK(factor(2080, 3) == std::vector<unsigned>{10, 13, 16});
    CHECK(factor(2080, 4) == std::vector<unsigned>{4, 5, 8, 13});

    CHECK(factor(3465, 2) == std::vector<unsigned>{55, 63});
    CHECK(factor(3465, 3) == std::vector<unsigned>{11, 15, 21});
    CHECK(factor(3465, 4) == std::vector<unsigned>{5, 7, 9, 11});

    CHECK(factor(64536, 2) == std::vector<unsigned>{24, 2689});
    CHECK(factor(64536, 3) == std::vector<unsigned>{1, 24, 2689});
    CHECK(factor(64536, 4) == std::vector<unsigned>{1, 2, 12, 2689});

    CHECK(factor(65536, 2) == std::vector<unsigned>{256, 256});
    CHECK(factor(65536, 3) == std::vector<unsigned>{32, 32, 64});
    CHECK(factor(65536, 4) == std::vector<unsigned>{16, 16, 16, 16});
}

TEST_CASE("identifier") {
    auto sbc = small_batch_configuration{
        -1,         1,          1,    32,      2,      16, precision::f32, transform_type::c2c,
        {1, 1, 32}, {1, 1, 33}, true, nullptr, nullptr};
    CHECK(sbc.identifier() == "sbfft_m1_M1_Mb1_N32_Kb2_sgs16_f32_c2c_is1_1_32_os1_1_33_in1");

    auto f2c = factor2_slm_configuration{+1,
                                         1,
                                         1,
                                         512,
                                         {16, 32},
                                         16,
                                         1,
                                         16,
                                         precision::f64,
                                         transform_type::c2c,
                                         {1, 1, 512},
                                         {1, 1, 512},
                                         true,
                                         nullptr,
                                         nullptr};
    CHECK(f2c.identifier() ==
          "f2fft_p1_M1_Mb1_N512_factorization16x32_Nb16_Kb1_sgs16_f64_c2c_is1_1_512_os1_1_512_in1");
}
