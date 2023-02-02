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

class jit_cache_all::impl {
  public:
    ~impl() = default;
    auto get_binary(jit_cache_key const &key) const -> std::pair<uint8_t const *, std::size_t> {
        if (auto it = binary_.find(key); it != binary_.end()) {
            return {it->second.data(), it->second.size()};
        }
        return {nullptr, 0};
    }
    void store_binary(jit_cache_key const &key, std::vector<uint8_t> binary) {
        binary_[key] = binary;
    }
    auto kernel_names() const {
        auto result = std::vector<std::string>{};
        for (auto const &[key, value] : binary_) {
            result.push_back(key.kernel_name);
        }
        return result;
    }

  private:
    std::unordered_map<jit_cache_key, std::vector<uint8_t>, jit_cache_key_hash> binary_;
};

jit_cache_all::jit_cache_all() : pimpl_(std::make_unique<impl>()) {}
jit_cache_all::~jit_cache_all() {}

auto jit_cache_all::get_binary(jit_cache_key const &key) const
    -> std::pair<uint8_t const *, std::size_t> {
    return pimpl_->get_binary(key);
}
void jit_cache_all::store_binary(jit_cache_key const &key, std::vector<uint8_t> binary) {
    pimpl_->store_binary(key, std::move(binary));
}

std::vector<std::string> jit_cache_all::kernel_names() const { return pimpl_->kernel_names(); }

} // namespace bbfft
