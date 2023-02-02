// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CACHE_20230131_HPP
#define CACHE_20230131_HPP

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
struct BBFFT_EXPORT cache_key {
    std::array<uint8_t, max_configuration_size> cfg = {};      ///< Binary dump of configuration
    uint64_t device_id = std::numeric_limits<uint64_t>::max(); ///< Unique device identifier

    bool operator==(cache_key const &other) const;
};

/**
 * @brief Interface for caches
 */
class BBFFT_EXPORT cache {
  public:
    /**
     * @brief Destructor
     */
    virtual ~cache();
    /**
     * @brief Get binary FFT kernel
     *
     * @param key FFT kernel identifier
     *
     * @return Pointer to native binary blob and size of binary blob
     */
    virtual auto get_binary(cache_key const &key) const
        -> std::pair<uint8_t const *, std::size_t> = 0;
    /**
     * @brief Store binary FFT kernel
     *
     * @param key FFT kernel identifier
     * @param binary Pointer to native binary blob
     * @param binary_size Size of binary blob
     */
    virtual void store_binary(cache_key const &key, std::vector<uint8_t> binary) = 0;
};

/**
 * @brief Cache that stores all kernels
 */
class BBFFT_EXPORT cache_all : public cache {
  public:
    /**
     * @brief Constructor
     */
    cache_all();
    /**
     * @brief Destructor
     *
     * Note: necessary for forward declaration of impl with unique_ptr
     */
    ~cache_all();
    /**
     * @copydoc cache::get_binary
     */
    auto get_binary(cache_key const &key) const -> std::pair<uint8_t const *, std::size_t> override;
    /**
     * @copydoc cache::store_binary
     */
    void store_binary(cache_key const &key, std::vector<uint8_t> binary) override;

  private:
    class impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace bbfft

#endif // CACHE_20230131_HPP
