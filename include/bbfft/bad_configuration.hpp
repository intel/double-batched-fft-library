// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BAD_CONFIGURATION_20220506_HPP
#define BAD_CONFIGURATION_20220506_HPP

#include "bbfft/export.hpp"

#include <exception>
#include <string>

namespace bbfft {

/**
 * @brief Exception type for faulty or unsupported configurations
 */
class BBFFT_EXPORT bad_configuration : public std::exception {
  public:
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     */
    bad_configuration(std::string what);
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     */
    bad_configuration(char const *what);
    /**
     * @brief Explanation of exception
     *
     * @return Explanatory string
     */
    char const *what() const noexcept override;

  private:
    std::string what_;
};

} // namespace bbfft

#endif // BAD_CONFIGURATION_20220506_HPP
