// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AOT_CACHE_20230202_HPP
#define AOT_CACHE_20230202_HPP

#include "bbfft/export.hpp"
#include "bbfft/jit_cache.hpp"
#include "bbfft/shared_handle.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace bbfft {

struct BBFFT_EXPORT aot_module {
    shared_handle<module_handle_t> module;
    std::unordered_set<std::string> kernel_names;
    std::uint64_t device_id;
};

class BBFFT_EXPORT aot_cache : public jit_cache {
  public:
    /**
     * @copydoc jit_cache::get
     */
    auto get(jit_cache_key const &key) const -> shared_handle<module_handle_t> override;
    /**
     * @copydoc jit_cache::store
     */
    void store(jit_cache_key const &key, shared_handle<module_handle_t> module) override;

    /**
     * @brief register module with this cache
     */
    void register_module(aot_module mod);

  private:
    std::vector<aot_module> modules_;
};

} // namespace bbfft

#endif // AOT_CACHE_20230202_HPP
