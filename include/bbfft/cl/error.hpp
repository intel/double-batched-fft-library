// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ERROR_20220412_HPP
#define CL_ERROR_20220412_HPP

#include "bbfft/export.hpp"

#include <CL/cl.h>

#include <cstdio>
#include <exception>
#include <string>

/**
 * @brief Checks OpenCL call for error and throws exception if call was not successful
 *
 * @param X OpenCL call or cl_int status code
 */
#define CL_CHECK(X)                                                                                \
    [](cl_int status) {                                                                            \
        if (status != CL_SUCCESS) {                                                                \
            char what[256];                                                                        \
            snprintf(what, sizeof(what), "%s in %s on line %d returned %s (%d).\n", #X, __FILE__,  \
                     __LINE__, ::bbfft::cl::cl_status_to_string(status), status);                  \
            throw ::bbfft::cl::error(what, status);                                                \
        }                                                                                          \
    }(X)

namespace bbfft::cl {

/**
 * @brief OpenCL error
 */
class BBFFT_EXPORT error : public std::exception {
  public:
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     * @param status Status code returned by OpenCL
     */
    error(std::string what, cl_int status);
    /**
     * @brief Constructor
     *
     * @param what Explanatory string
     * @param status Status code returned by OpenCL
     */
    error(char const *what, cl_int status);
    /**
     * @brief Explanation
     *
     * @return Explanatory string
     */
    char const *what() const noexcept override;
    /**
     * @brief OpenCL status code
     *
     * @return Status code
     */
    cl_int status_code() const noexcept;

  private:
    std::string what_;
    cl_int status_;
};

BBFFT_EXPORT char const *cl_status_to_string(cl_int status);

} // namespace bbfft::cl

#endif // CL_ERROR_20220412_HPP
