// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef STRING_UTIL_20220509_HPP
#define STRING_UTIL_20220509_HPP

#include "clir/export.hpp"

#include <string>
#include <string_view>

namespace clir {

CLIR_EXPORT std::string escaped_string(std::string_view str);

}

#endif // STRING_UTIL_20220509_HPP
