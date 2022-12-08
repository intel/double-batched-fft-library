// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/user_module.hpp"

namespace bbfft {

user_module::operator bool() const noexcept {
    return data != nullptr && length > 0 && (load_function != nullptr || store_function != nullptr);
}

} // namespace bbfft
