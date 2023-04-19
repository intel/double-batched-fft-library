// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/aot_cache.hpp"

#include <utility>

#include <iostream>

namespace bbfft {

auto aot_cache::get(jit_cache_key const &key) const -> shared_handle<module_handle_t> {
    for (auto const &mod : modules_) {
        if (auto it = mod.kernel_names.find(key.kernel_name); it != mod.kernel_names.end()) {
            return mod.module;
        }
    }
    return {};
}

void aot_cache::store(jit_cache_key const &, shared_handle<module_handle_t>) {}

void aot_cache::register_module(aot_module mod) { modules_.emplace_back(std::move(mod)); }

} // namespace bbfft
