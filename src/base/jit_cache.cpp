// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/jit_cache.hpp"

#include <functional>
#include <unordered_map>
#include <utility>

namespace bbfft {

bool jit_cache_key::operator==(jit_cache_key const &other) const {
    return kernel_name == other.kernel_name && device_id == other.device_id;
}

std::size_t jit_cache_key_hash::operator()(jit_cache_key const &key) const noexcept {
    std::size_t hash = std::hash<std::string>()(key.kernel_name);
    std::size_t hash2 = std::hash<uint64_t>()(key.device_id);
    hash ^= hash2 + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

jit_cache::~jit_cache() {}

} // namespace bbfft
