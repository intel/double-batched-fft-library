// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef JIT_CACHE_20230202_HPP
#define JIT_CACHE_20230202_HPP

#include "bbfft/detail/generator_impl.hpp"
#include "bbfft/export.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <vector>

namespace bbfft {

/**
 * @brief Unique identifier for fft kernel
 */
struct BBFFT_EXPORT jit_cache_key {
    std::array<uint8_t, max_configuration_size> cfg = {};      ///< Binary dump of configuration
    uint64_t device_id = std::numeric_limits<uint64_t>::max(); ///< Unique device identifier

    bool operator==(jit_cache_key const &other) const;
};

/**
 * @brief Interface for jit_caches
 */
class BBFFT_EXPORT jit_cache {
  public:
    /**
     * @brief Destructor
     */
    virtual ~jit_cache();
    /**
     * @brief Get binary FFT kernel
     *
     * @param key FFT kernel identifier
     *
     * @return Pointer to native binary blob and size of binary blob
     */
    virtual auto get_binary(jit_cache_key const &key) const
        -> std::pair<uint8_t const *, std::size_t> = 0;
    /**
     * @brief Store binary FFT kernel
     *
     * @param key FFT kernel identifier
     * @param binary Pointer to native binary blob
     * @param binary_size Size of binary blob
     */
    virtual void store_binary(jit_cache_key const &key, std::vector<uint8_t> binary) = 0;
};

/**
 * @brief Cache that stores all kernels
 */
class BBFFT_EXPORT jit_cache_all : public jit_cache {
  public:
    /**
     * @brief Constructor
     */
    jit_cache_all();
    /**
     * @brief Destructor
     *
     * Note: necessary for forward declaration of impl with unique_ptr
     */
    ~jit_cache_all();
    /**
     * @copydoc jit_cache::get_binary
     */
    auto get_binary(jit_cache_key const &key) const
        -> std::pair<uint8_t const *, std::size_t> override;
    /**
     * @copydoc cache::store_binary
     */
    void store_binary(jit_cache_key const &key, std::vector<uint8_t> binary) override;

  private:
    class impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace bbfft

#endif // JIT_CACHE_20230202_HPP
