// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_ERROR_20220412_HPP
#define ZE_ERROR_20220412_HPP

#include "bbfft/export.hpp"

#include <level_zero/ze_api.h>

#include <cstdio>
#include <exception>
#include <string>

/**
 * @brief Checks Level Zero call for error and throws exception if call was not successful
 *
 * @param X Level Zero call or ze_result_t status code
 */
#define ZE_CHECK(X)                                                                                \
    [](ze_result_t status) {                                                                       \
        if (status != ZE_RESULT_SUCCESS) {                                                         \
            char what[256];                                                                        \
            snprintf(what, sizeof(what), "%s in %s on line %d returned %s (%d).\n", #X, __FILE__,  \
                     __LINE__, ::bbfft::ze::ze_result_to_string(status), status);                  \
            throw ::bbfft::ze::error(what, status);                                                \
        }                                                                                          \
    }(X)

namespace bbfft::ze {

/**
 * @brief Level Zero error
 */
class BBFFT_EXPORT error : public std::exception {
  public:
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     * @param status Status code returned by Level Zero
     */
    error(std::string what, ze_result_t status);
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     * @param status Status code returned by Level Zero
     */
    error(char const *what, ze_result_t status);
    /**
     * @brief Explanation
     *
     * @return Explanatory string
     */
    char const *what() const noexcept override;
    /**
     * @brief Level Zero status code
     *
     * @return Status code
     */
    ze_result_t error_code() const noexcept;

  private:
    std::string what_;
    ze_result_t status_;
};

BBFFT_EXPORT char const *ze_result_to_string(ze_result_t status);

} // namespace bbfft::ze

#endif // ZE_ERROR_20220412_HPP
