// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/detail/compiler_options.hpp"

namespace bbfft::detail {

const std::vector<std::string> compiler_options{"-cl-mad-enable"};
const std::vector<std::string> required_extensions{"cl_khr_fp64"};

} // namespace bbfft::detail
