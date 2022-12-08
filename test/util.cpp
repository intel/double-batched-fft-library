// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/string_util.hpp"

#include "doctest/doctest.h"

using namespace clir;

TEST_CASE("Escape string") {
    CHECK(escaped_string("\'\"\?\\\a\b\f\n\r\t\v") == "\\'\\\"\\?\\\\\\a\\b\\f\\n\\r\\t\\v");
}

