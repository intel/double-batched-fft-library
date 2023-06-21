// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/jit_cache_all.hpp"

#include <utility>

namespace bbfft {

auto jit_cache_all::get(jit_cache_key const &key) const -> shared_handle<module_handle_t> {
    if (auto it = mods_.find(key); it != mods_.end()) {
        return it->second;
    }
    return {};
}
void jit_cache_all::store(jit_cache_key const &key, shared_handle<module_handle_t> mod) {
    mods_[key] = std::move(mod);
}

auto jit_cache_all::kernel_names() const -> std::vector<std::string> {
    auto result = std::vector<std::string>{};
    for (auto const &[key, value] : mods_) {
        result.push_back(key.kernel_name);
    }
    return result;
}

} // namespace bbfft
