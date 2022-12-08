// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/string_util.hpp"

#include <cstddef>

namespace clir {

std::string escaped_string(std::string_view str) {
    constexpr char special_escaped[] = "\\'\\\"\\?\\\\\\a\\b\\f\\n\\r\\t\\v";
    auto const is_special = [](char c) -> int {
        constexpr char special[] = "\'\"\?\\\a\b\f\n\r\t\v";
        for (std::size_t i = 0; i < sizeof(special) - 1; ++i) {
            if (c == special[i]) {
                return i;
            }
        }
        return -1;
    };
    unsigned num_special = 0;
    for (auto c : str) {
        if (is_special(c) >= 0) {
            ++num_special;
        }
    }
    auto result = std::string{};
    result.reserve(str.size() + num_special);
    for (auto c : str) {
        if (int i = is_special(c); i >= 0) {
            result.push_back(special_escaped[2 * i]);
            result.push_back(special_escaped[2 * i + 1]);
        } else {
            result.push_back(c);
        }
    }

    return result;
}

} // namespace clir

