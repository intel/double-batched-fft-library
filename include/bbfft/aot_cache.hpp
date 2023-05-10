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

/**
 * @brief Collection of native module handle, set of kernel names stored in the native module, and a
 * device identifier.
 */
struct BBFFT_EXPORT aot_module {
    shared_handle<module_handle_t> mod;           ///< Native module handle
    std::unordered_set<std::string> kernel_names; ///< Set of kernel names
    std::uint64_t device_id;                      ///< Device id for mod
};

/**
 * @brief Cache to look up ahead-of-time compiled FFT kernels
 */
class BBFFT_EXPORT aot_cache : public jit_cache {
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
     * @brief register module with this cache
     */
    void register_module(aot_module aot_mod);

  private:
    std::vector<aot_module> aot_modules_;
};

} // namespace bbfft

#endif // AOT_CACHE_20230202_HPP
