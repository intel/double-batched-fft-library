// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/aot_cache.hpp"

#include <utility>

#include <iostream>

namespace bbfft {

auto aot_cache::get(jit_cache_key const &key) const -> shared_handle<module_handle_t> {
    for (auto const &aot_mod : aot_modules_) {
        if (key.device_id == aot_mod.device_id) {
            if (auto it = aot_mod.kernel_names.find(key.kernel_name);
                it != aot_mod.kernel_names.end()) {
                return aot_mod.mod;
            }
        }
    }
    return {};
}

void aot_cache::store(jit_cache_key const &, shared_handle<module_handle_t>) {}

void aot_cache::register_module(aot_module aot_mod) {
    aot_modules_.emplace_back(std::move(aot_mod));
}

} // namespace bbfft
