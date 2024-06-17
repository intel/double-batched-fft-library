// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FFT1D_CUSTOM2_20240616_HPP
#define FFT1D_CUSTOM2_20240616_HPP

#include "bbfft/configuration.hpp"

#include <CL/cl.h>
#include <array>
#include <cstddef>
#include <vector>

class fft1d_custom2 {
  public:
    fft1d_custom2(bbfft::configuration const &cfg, cl_command_queue queue, cl_context context,
                  cl_device_id device);
    ~fft1d_custom2();

    fft1d_custom2(fft1d_custom2 const &other) = delete;
    fft1d_custom2(fft1d_custom2 &&other) = delete;
    fft1d_custom2 &operator=(fft1d_custom2 const &other) = delete;
    fft1d_custom2 &operator=(fft1d_custom2 &&other) = delete;

    auto execute(cl_mem in, cl_mem out, std::vector<cl_event> const &dep_events) -> cl_event;

  private:
    struct plan {
        cl_kernel kernel;
        std::array<std::size_t, 3u> gws;
        std::array<std::size_t, 3u> lws;
    };

    cl_command_queue queue_;
    std::vector<plan> plans_;
    cl_mem twiddle_;
    cl_mem buffer_;
    cl_program program_;
    cl_kernel r2c_post_;
    std::array<std::size_t, 3u> r2c_post_gws_;
};

#endif // FFT1D_CUSTOM2_20240616_HPP
