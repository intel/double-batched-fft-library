// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef JIT_CACHE_20230202_HPP
#define JIT_CACHE_20230202_HPP

#include "bbfft/export.hpp"
#include "bbfft/shared_handle.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace bbfft {

using device_handle_t = std::uintptr_t;
using module_handle_t = std::uintptr_t;

template <class To, class From> To cast(From v);
template <class To, class From> To cast(From v) {
    static_assert(sizeof(To) == sizeof(From));
    static_assert(alignof(To) == alignof(From));
    return reinterpret_cast<To>(v);
}

/**
 * @brief Unique identifier for fft kernel
 */
struct BBFFT_EXPORT jit_cache_key {
    std::string kernel_name = {}; ///< Name of the OpenCL kernel
    std::uint64_t device_id = std::numeric_limits<std::uint64_t>::max(); ///< Unique device id

    bool operator==(jit_cache_key const &other) const;
};
/**
 * @brief Hash function for jit_cache_key
 */
struct jit_cache_key_hash {
    /**
     * @brief Compute hash
     *
     * @param key cache key
     *
     * @return hash
     */
    std::size_t operator()(jit_cache_key const &key) const noexcept;
};

/**
 * @brief Interface for jit_caches
 *
 * @tparam backend-specific kernel bundle type
 */
class BBFFT_EXPORT jit_cache {
  public:
    /**
     * @brief Destructor
     */
    virtual ~jit_cache();
    /**
     * @brief Get FFT kernel bundle
     *
     * @param key FFT kernel identifier
     *
     * @return kernel bundle
     */
    virtual auto get(jit_cache_key const &key) const -> shared_handle<module_handle_t> = 0;
    /**
     * @brief Store  FFT kernel bundle
     *
     * @param key FFT kernel identifier
     * @param mod kernel bundle
     */
    virtual void store(jit_cache_key const &key, shared_handle<module_handle_t> mod) = 0;
};

} // namespace bbfft

#endif // JIT_CACHE_20230202_HPP
