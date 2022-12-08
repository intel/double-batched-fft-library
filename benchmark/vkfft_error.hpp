// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VKFFT_ERROR_20220601_HPP
#define VKFFT_ERROR_20220601_HPP

#include <vkFFT.h>

#include <cstdio>
#include <exception>
#include <string>

#define VKFFT_CHECK(X)                                                                             \
    [](VkFFTResult status) {                                                                       \
        if (status != VKFFT_SUCCESS) {                                                             \
            char what[256];                                                                        \
            snprintf(what, sizeof(what), "%s in %s on line %d returned %s (%d).\n", #X, __FILE__,  \
                     __LINE__, vkfft_result_to_string(status), status);                            \
            throw vkfft_error(what, status);                                                       \
        }                                                                                          \
    }(X)

class vkfft_error : public std::exception {
  public:
    vkfft_error(std::string what, VkFFTResult status);
    vkfft_error(char const *what, VkFFTResult status);
    char const *what() const noexcept override { return what_.c_str(); }
    VkFFTResult status_code() const noexcept { return status_; }

  private:
    std::string what_;
    VkFFTResult status_;
};

char const *vkfft_result_to_string(VkFFTResult status);

#endif // VKFFT_ERROR_20220601_HPP
