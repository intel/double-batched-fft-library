#ifndef FFT1D_20240610_HPP
#define FFT1D_20240610_HPP

#include "bbfft/cl/make_plan.hpp"
#include "bbfft/cl/mem.hpp"
#include "bbfft/configuration.hpp"
#include "bbfft/mem.hpp"
#include "bbfft/prime_factorization.hpp"

#include <CL/cl.h>
#include <array>
#include <cstddef>
#include <vector>

/*
 * y(k) = sum_j x(j) w_N(jk)
 * j = j0 + j1 N0 + j2 N0 N1
 * k = k2 + k1 N2 + k0 N2 N1
 *
 * y(k2,k1,k0) = sum_j0 sum_j1 sum_j2
 *                  x(j0,j1,j2) w_N((j0 + j1 N0 + j2 N0 N1)(k2 + k1 N2 + k0 N2 N1))
 *             = sum_j0 sum_j1 sum_j2
 *                  x(j0,j1,j2) w_N2(j2k2) w_N1(j1k1)) w_N12(j1k2) w_N0(j0k0) w_N(j0(k2 + k1 N2))
 *             = sum_j0 w_N0(j0k0) w_N(j0(k2 + k1 N2))
 *                  sum_j1 w_N1(j1k1)) w_N12(j1k2)
 *                      sum_j2 x(j0,j1,j2) w_N2(j2k2)
 *
 * x1(j0,j1,k2) = sum_j2 x(j0,j1,j2) w_N2(j2k2)
 * x2(j0,j1,k2) = x1(j0,j1,k2) w_N12(j1k2)
 * x3(j0,k1,k2) = sum_j1 x2(j0,j1,k2) w_N1(j1k1)
 * x4(j0,k1,k2) = x3(j0,k1,k2) w_N(j0(k2 + k1 N2))
 * y(k1,k1,k2) = sum_j0 x4(j0,k1,k2) w_N1(j0k0)
 */

class fft1d {
  public:
    fft1d(bbfft::configuration const &cfg, cl_command_queue queue, cl_context context,
          cl_device_id device);
    ~fft1d();

    fft1d(fft1d const &other) = delete;
    fft1d(fft1d &&other) = delete;
    fft1d &operator=(fft1d const &other) = delete;
    fft1d &operator=(fft1d &&other) = delete;

    auto execute(cl_mem in, cl_mem out, std::vector<cl_event> const &dep_events) -> cl_event;

  private:
    static auto get_real_type(bbfft::precision p) -> char const *;

    cl_command_queue queue_;
    std::vector<bbfft::opencl_plan> plans_;
    cl_mem buffer_;
    cl_program program_;
    cl_kernel bit_reversal_, r2c_post_;
    std::array<std::size_t, 3u> bit_reversal_gws_, bit_reversal_lws_, r2c_post_gws_;
};

#endif // FFT1D_20240610_HPP
