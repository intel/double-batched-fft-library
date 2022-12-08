// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/bad_configuration.hpp"

#include <utility>

namespace bbfft {

bad_configuration::bad_configuration(std::string what) : what_(std::move(what)) {}
bad_configuration::bad_configuration(char const *what) : what_(what) {}
char const *bad_configuration::what() const noexcept { return what_.c_str(); }

} // namespace bbfft
