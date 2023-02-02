// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/cache.hpp"

#include <functional>
#include <unordered_map>
#include <utility>

namespace bbfft {

bool cache_key::operator==(cache_key const &other) const {
    return cfg == other.cfg && device_id == other.device_id;
}

struct cache_key_hash {
    std::size_t operator()(cache_key const &key) const noexcept {
        std::size_t hash = std::hash<uint64_t>()(key.device_id);
        std::hash<uint8_t> hasher;
        for (auto c : key.cfg) {
            hash ^= hasher(c) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

cache::~cache() {}

class cache_all::impl {
  public:
    ~impl() = default;
    auto get_binary(cache_key const &key) const -> std::pair<uint8_t const *, std::size_t> {
        if (auto it = binary_.find(key); it != binary_.end()) {
            return {it->second.data(), it->second.size()};
        }
        return {nullptr, 0};
    }
    void store_binary(cache_key const &key, std::vector<uint8_t> binary) { binary_[key] = binary; }

  private:
    std::unordered_map<cache_key, std::vector<uint8_t>, cache_key_hash> binary_;
};

cache_all::cache_all() : pimpl_(std::make_unique<impl>()) {}
cache_all::~cache_all() {}

auto cache_all::get_binary(cache_key const &key) const -> std::pair<uint8_t const *, std::size_t> {
    return pimpl_->get_binary(key);
}
void cache_all::store_binary(cache_key const &key, std::vector<uint8_t> binary) {
    pimpl_->store_binary(key, std::move(binary));
}

} // namespace bbfft
