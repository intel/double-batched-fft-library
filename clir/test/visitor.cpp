// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/builder.hpp"
#include "clir/builtin_function.hpp"
#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/func.hpp"
#include "clir/internal/function_node.hpp"
#include "clir/visit.hpp"
#include "clir/visitor/codegen_opencl.hpp"
#include "clir/visitor/equal_expr.hpp"
#include "clir/visitor/required_extensions.hpp"
#include "clir/visitor/to_imm.hpp"
#include "clir/visitor/unsafe_simplification.hpp"

#include "doctest/doctest.h"

#include <cstdint>
#include <sstream>
#include <variant>

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

    SUBCASE("Function qualifier") {
        CHECK(to_string(function_qualifier::none) == "");
        CHECK(to_string(function_qualifier::extern_t) == "extern");
        CHECK(to_string(function_qualifier::inline_t) == "inline");
        CHECK(to_string(function_qualifier::kernel_t) == "kernel");
        CHECK(to_string(function_qualifier::extern_t | function_qualifier::inline_t) ==
              "extern inline");
        CHECK(to_string(function_qualifier::extern_t | function_qualifier::kernel_t) ==
              "extern kernel");
        CHECK(to_string(function_qualifier::inline_t | function_qualifier::kernel_t) ==
              "inline kernel");
        CHECK(to_string(function_qualifier::extern_t | function_qualifier::inline_t |
                        function_qualifier::kernel_t) == "extern inline kernel");
    }

    SUBCASE("Type qualifier") {
        CHECK(to_string(type_qualifier::none) == "");
        CHECK(to_string(type_qualifier::const_t) == "const");
        CHECK(to_string(type_qualifier::restrict_t) == "restrict");
        CHECK(to_string(type_qualifier::volatile_t) == "volatile");
        CHECK(to_string(type_qualifier::const_t | type_qualifier::volatile_t) == "const volatile");
    }

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

        CHECK(e2s(ternary_conditional(a < b, a, b)) == "a < b ? a : b");
        CHECK(e2s(5 + ternary_conditional(a < b, a + 3, b - 2)) == "5 + (a < b ? a + 3 : b - 2)");
    }

    SUBCASE("printf") { CHECK(e2s(printf({"%d %d\n", a, b})) == "printf(\"%d %d\\n\", a, b)"); }
    SUBCASE("call") { CHECK(e2s(call("test", {a, b})) == "test(a, b)"); }
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

    SUBCASE("data type") {
        CHECK(e2s(generic_float()) == "float");
        CHECK(e2s(local_float(4)) == "local float4");
        CHECK(e2s(pointer_to(global_float())) == "global float*");
        CHECK(e2s(pointer_to(pointer_to(global_float()))) == "global float**");
        CHECK(e2s(pointer_to(pointer_to(global_float()), address_space::global_t)) ==
              "global float**global");
        CHECK(e2s(array_of(global_float(), 10)) == "global float[10]");
        CHECK(e2s(array_of(pointer_to(global_float()), 10)) == "global float*[10]");
        CHECK(e2s(array_of(array_of(pointer_to(generic_float()), 10), 12)) == "float*[12][10]");
        CHECK(e2s(array_of(pointer_to(array_of(generic_float(), 10)), 12)) == "float(*[12])[10]");
        CHECK(e2s(pointer_to(array_of(array_of(generic_int(), 10), 12))) == "int(*)[12][10]");
        CHECK(e2s(pointer_to(
                  pointer_to(array_of(array_of(generic_int(), 10), 12), address_space::global_t),
                  address_space::local_t)) == "int(*global*local)[12][10]");
        CHECK(e2s(pointer_to(pointer_to(array_of(pointer_to(global_float()), 10)))) ==
              "global float*(**)[10]");
        CHECK(e2s(cast(array_of(generic_float(), 10), a)) == "(float[10]) a");
        std::stringstream oss;
        oss << "global float*(** a)[10];" << std::endl;
        CHECK(e2s(declaration(pointer_to(pointer_to(array_of(pointer_to(global_float()), 10))),
                              a)) == oss.str());

        CHECK(e2s(global_atomic_float()) == "global atomic_float");
        CHECK(e2s(global_atomic_float(type_qualifier::const_t | type_qualifier::volatile_t)) ==
              "global const volatile atomic_float");
        CHECK(e2s(generic_int(2, type_qualifier::const_t)) == "const int2");
        CHECK(e2s(pointer_to(global_atomic_int(type_qualifier::volatile_t), address_space::global_t,
                             type_qualifier::restrict_t)) ==
              "global volatile atomic_int*global restrict");
    }

    SUBCASE("prototype") {
        std::stringstream oss1;
        oss1 << "void test();" << std::endl;
        CHECK(e2s(func(std::make_shared<internal::prototype>("test"))) == oss1.str());

        auto p2 = std::make_shared<internal::prototype>("test");
        p2->qualifiers(function_qualifier::kernel_t);
        std::stringstream oss2;
        oss2 << "kernel" << std::endl << "void test();" << std::endl;
        CHECK(e2s(func(p2)) == oss2.str());
    }

    SUBCASE("program") {
        auto fb = function_builder("test");
        fb.qualifier(function_qualifier::extern_t);
        program_builder pb;
        pb.declare_assign(constant_int(), "c", 5);
        pb.add(fb.get_product());
        auto fb2 = function_builder("test2");
        fb2.body([](block_builder &) {});
        pb.add(fb2.get_product());
        std::stringstream oss;
        oss << "constant int c = 5;" << std::endl
            << "extern" << std::endl
            << "void test();" << std::endl
            << "void test2() {" << std::endl
            << '}' << std::endl;
        CHECK(e2s(pb.get_product()) == oss.str());
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

    CHECK(is_equivalent(ternary_conditional(a + b, b, a), ternary_conditional(a + b, b, a)));
    CHECK(!is_equivalent(ternary_conditional(a + b, b, a), ternary_conditional(a + b, b + a, a)));
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
    CHECK(is_equivalent(unsafe_simplify(ternary_conditional(a < b + 0, a + 0, a * (b * 1))),
                        ternary_conditional(a < b, a, a * b)));

    /* Builtin function test */
    CHECK(is_equivalent(unsafe_simplify(get_global_id(a + 0)), get_global_id(a)));
    CHECK(is_equivalent(unsafe_simplify(intel_sub_group_shuffle_xor(0, a)), 0));
}

TEST_CASE("To imm") {
    expr a = 42;
    expr b = 42.0;
    CHECK(std::holds_alternative<int64_t>(get_imm(a)));
    CHECK(std::holds_alternative<double>(get_imm(b)));
    CHECK(std::holds_alternative<std::monostate>(get_imm(a + b)));
}

TEST_CASE("Required extensions") {
    auto fb = function_builder("test");
    fb.body([](block_builder &bb) {
        bb.declare_assign(generic_int(), "a", -intel_sub_group_shuffle(0, 0));
        bb.declare_assign(generic_int(), "b", intel_sub_group_block_read_us(0));
    });
    auto ext = get_required_extensions(fb.get_product());
    CHECK(ext.size() == 2);
    CHECK(ext == std::vector<extension>{extension::cl_intel_subgroups,
                                        extension::cl_intel_subgroups_short});
}
