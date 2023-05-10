// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef JIT_CACHE_ALL_20230202_HPP
#define JIT_CACHE_ALL_20230202_HPP

#include "bbfft/export.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace bbfft {

/**
 * @brief Cache that stores all kernels
 */
class BBFFT_EXPORT jit_cache_all : public jit_cache {
  public:
    /**
     * @copydoc jit_cache::get
     */
    auto get(jit_cache_key const &key) const -> shared_handle<module_handle_t> override;
    /**
     * @copydoc jit_cache::store
     */
    void store(jit_cache_key const &key, shared_handle<module_handle_t> mod) override;
    /**
     * @brief Get all kernel names stored in this cache
     */
    auto kernel_names() const -> std::vector<std::string>;

  private:
    std::unordered_map<jit_cache_key, shared_handle<module_handle_t>, jit_cache_key_hash> mods_;
};

} // namespace bbfft

#endif // JIT_CACHE_ALL_20230202_HPP
