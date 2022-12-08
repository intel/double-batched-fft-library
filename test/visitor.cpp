// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builtin_function.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/var.hpp"
#include "clir/visit.hpp"
#include "clir/visitor/codegen_opencl.hpp"
#include "clir/visitor/equal_expr.hpp"
#include "clir/visitor/unsafe_simplification.hpp"

#include "doctest/doctest.h"

#include <sstream>

using namespace clir;

TEST_CASE("Code generation") {
    auto a = var("a");
    auto b = var("b");
    auto p = var("p");

    auto e2s = [](auto &&e) {
        std::stringstream s;
        generate_opencl(s, e);
        return s.str();
    };

    SUBCASE("Operator precedence and associativity") {
        CHECK(e2s(++a) == "++a");
        CHECK(e2s(a++) == "a++");
        CHECK(e2s(dereference(p)++) == "(*p)++");
        CHECK(e2s(dereference(p++)) == "*p++");
        CHECK(e2s(++dereference(p)) == "++*p");
        CHECK(e2s(dereference(++p)) == "*++p");

        CHECK(e2s(a + b * a) == "a + b * a");
        CHECK(e2s((a * b) + a) == "a * b + a");
        CHECK(e2s((a + b) * a) == "(a + b) * a");
        CHECK(e2s(a * (a + b)) == "a * (a + b)");
        CHECK(e2s((a * b) / a) == "a * b / a");
        CHECK(e2s(a * (b / a)) == "a * (b / a)");
        CHECK(e2s(a * b / a) == "a * b / a");

        CHECK(e2s(p[a + b]) == "p[a + b]");
        CHECK(e2s(address_of(p[a + b])) == "&p[a + b]");
        CHECK(e2s(dereference(p)[a + b]) == "(*p)[a + b]");

        CHECK(e2s(cast(generic_int(), p)) == "(int) p");
        CHECK(e2s(cast(pointer_to(generic_int()), p)[4]) == "((int*) p)[4]");
        CHECK(e2s(cast(generic_float(), cast(generic_int(), p))) == "(float) (int) p");
        CHECK(e2s(dereference(cast(pointer_to(generic_int()), p++))) == "*(int*) p++");
    }

    SUBCASE("printf") { CHECK(e2s(printf({"%d %d\n", a, b})) == "printf(\"%d %d\\n\", a, b)"); }
    SUBCASE("external") { CHECK(e2s(call_external("test", {a, b})) == "test(a, b)"); }
    SUBCASE("init_vector") {
        CHECK(e2s(init_vector(generic_int(3), {0, 1, 2})) == "(int3) (0, 1, 2)");
        CHECK(e2s(init_vector(generic_int(4), {1, init_vector(generic_int(2), {2, 3}), 4})) ==
              "(int4) (1, (int2) (2, 3), 4)");
    }

    SUBCASE("swizzle") {
        CHECK(e2s(a.lo()) == "a.lo");
        CHECK(e2s(a.hi()) == "a.hi");
        CHECK(e2s(a.even()) == "a.even");
        CHECK(e2s(a.odd()) == "a.odd");
        CHECK(e2s(a.lo().odd()) == "a.lo.odd");
        CHECK(e2s(a.s(2)) == "a.z");
        CHECK(e2s(a.s(3, 0)) == "a.wx");
        CHECK(e2s(a.s(0, 0, 0)) == "a.xxx");
        CHECK(e2s(a.s(0, 1, 2, 0)) == "a.xyzx");
        CHECK(e2s(a.s(0, 1, 2, 0, 5, 7, 7, 3)) == "a.s01205773");
        CHECK(e2s(a.s(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)) ==
              "a.sfedcba9876543210");
    }
}

TEST_CASE("Equal expression") {
    auto a = var("a");
    auto b = var("b");
    auto b2 = var("b");

    CHECK(is_equivalent(a, a));
    CHECK(is_equivalent(a * b, a * b));
    CHECK(!is_equivalent(a + b, a * b));
    CHECK(!is_equivalent(b + a, a + b));

    CHECK(is_equivalent(a * get_global_id(a + b), a * get_global_id(a + b)));
    CHECK(!is_equivalent(a * get_global_id(a + b), a * get_global_id(a - b)));

    CHECK(is_equivalent(generic_int(), generic_int()));
    CHECK(!is_equivalent(global_int(), private_int()));

    CHECK(!is_equivalent(b, b2));
}

TEST_CASE("Unsafe expression simplification") {
    auto a = var("a");
    auto b = var("b");

    /* Binary op tests */
    CHECK(is_equivalent(unsafe_simplify(0 + a), a));
    CHECK(is_equivalent(unsafe_simplify(0 | a), a));
    CHECK(is_equivalent(unsafe_simplify(0 ^ a), a));
    CHECK(is_equivalent(unsafe_simplify(0 - a), -a));

    CHECK(is_equivalent(unsafe_simplify(0 * a), 0));
    CHECK(is_equivalent(unsafe_simplify(0 / a), 0));
    CHECK(is_equivalent(unsafe_simplify(0 % a), 0));
    CHECK(is_equivalent(unsafe_simplify(0 & a), 0));
    CHECK(is_equivalent(unsafe_simplify(0 << a), 0));
    CHECK(is_equivalent(unsafe_simplify(0 >> a), 0));

    CHECK(is_equivalent(unsafe_simplify(a + 0), a));
    CHECK(is_equivalent(unsafe_simplify(a | 0), a));
    CHECK(is_equivalent(unsafe_simplify(a ^ 0), a));
    CHECK(is_equivalent(unsafe_simplify(a - 0), a));
    CHECK(is_equivalent(unsafe_simplify(a << 0), a));
    CHECK(is_equivalent(unsafe_simplify(a >> 0), a));

    CHECK(is_equivalent(unsafe_simplify(1 * a), a));
    CHECK(is_equivalent(unsafe_simplify(a * 1), a));
    CHECK(is_equivalent(unsafe_simplify(a / 1), a));
    CHECK(is_equivalent(unsafe_simplify(a % 1), 0));

    /* Tree tests */
    CHECK(is_equivalent(unsafe_simplify(-(b + 0)), -b));
    CHECK(is_equivalent(unsafe_simplify(a[b + 0]), a[b]));
    CHECK(is_equivalent(unsafe_simplify(a + (b + 0)), a + b));
    CHECK(is_equivalent(unsafe_simplify(a * (b * 1)), a * b));

    /* Builtin function test */
    CHECK(is_equivalent(unsafe_simplify(get_global_id(a + 0)), get_global_id(a)));
    CHECK(is_equivalent(unsafe_simplify(intel_sub_group_shuffle_xor(0, a)), 0));
}
