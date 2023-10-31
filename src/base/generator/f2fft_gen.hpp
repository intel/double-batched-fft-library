// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef F2FFT_GEN_20230811_HPP
#define F2FFT_GEN_20230811_HPP

#include "bbfft/detail/generator_impl.hpp"
#include "generator/snippet.hpp"
#include "generator/tensor_accessor.hpp"
#include "generator/tensor_view.hpp"
#include "generator/utility.hpp"
#include "scrambler.hpp"

#include "clir/builder.hpp"
#include "clir/expr.hpp"

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string_view>

namespace bbfft {

class f2fft_gen {
  public:
    struct gen_cfg {
        std::size_t N_in, N_out, N_slm, N_fft;
        short in_components, out_components;
    };
    f2fft_gen(gen_cfg p) : p_(p) {}
    virtual ~f2fft_gen() {}

    void generate(std::ostream &os, factor2_slm_configuration const &cfg,
                  std::string_view name) const;
    inline auto const &p() const { return p_; }

  protected:
    struct copy_params {
        factor2_slm_configuration const &cfg;
        precision_helper fph;
        tensor_view<3u> const &view;
        tensor_view<1u> const &X1_view;
        tensor_view<1u> const &x_view;
        std::shared_ptr<array_accessor> x_acc;
        clir::expr mm;
        clir::expr kk;
        clir::expr K;
        clir::expr j1;
    };
    struct prepost_params {
        factor2_slm_configuration const &cfg;
        precision_helper fph;
        tensor_view<3u> const &view;
        tensor_view<1u> const &X1_view;
        clir::expr mm;
        clir::expr n_local;
        clir::expr kk;
        clir::expr K;
        clir::expr twiddle = nullptr;
        unscrambler<clir::expr> unscramble = unscrambler<clir::expr>({});
    };

    virtual void preprocess(clir::block_builder &, prepost_params) const {}
    virtual void load(clir::block_builder &bb, copy_params cp) const = 0;
    virtual void postprocess(clir::block_builder &, prepost_params) const = 0;

    void global_load(clir::block_builder &bb, copy_params const &cp, clir::expr k,
                     tensor_view<3u> const &view) const;

  private:
    gen_cfg p_;
};

class f2fft_gen_c2c : public f2fft_gen {
  public:
    f2fft_gen_c2c(std::size_t N) : f2fft_gen(gen_cfg{N, N, N, N, 2, 2}) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void postprocess(clir::block_builder &bb, prepost_params pp) const override;
};

class f2fft_gen_r2c : public f2fft_gen {
  public:
    f2fft_gen_r2c(std::size_t N, short n_stride)
        : f2fft_gen(gen_cfg{N, N / 2 + 1, N / n_stride + 1, N / n_stride, 1, 2}) {}
};

class f2fft_gen_r2c_half : public f2fft_gen_r2c {
  public:
    f2fft_gen_r2c_half(std::size_t N) : f2fft_gen_r2c(N, 2) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void postprocess(clir::block_builder &bb, prepost_params pp) const override;

  private:
    static void postprocess_i(clir::block_builder &bb, precision_helper fph, clir::expr i,
                              clir::expr twiddle, std::size_t N, tensor_view<1u> const &y,
                              tensor_view<1u> const &X, unscrambler<clir::expr> const &unscramble);
};

class f2fft_gen_r2c_double : public f2fft_gen_r2c {
  public:
    f2fft_gen_r2c_double(std::size_t N) : f2fft_gen_r2c(N, 1) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void postprocess(clir::block_builder &bb, prepost_params pp) const override;

  private:
    static void postprocess_i(clir::block_builder &bb, precision_helper fph, clir::expr i,
                              std::size_t N, tensor_view<1u> const &x, tensor_view<1u> const &ya,
                              tensor_view<1u> const &yb, unscrambler<clir::expr> const &unscramble);
};

class f2fft_gen_c2r : public f2fft_gen {
  public:
    f2fft_gen_c2r(std::size_t N, short n_stride)
        : f2fft_gen(gen_cfg{N / 2 + 1, N, N / n_stride + 1, N / n_stride, 2, 1}) {}
};

class f2fft_gen_c2r_half : public f2fft_gen_c2r {
  public:
    f2fft_gen_c2r_half(std::size_t N) : f2fft_gen_c2r(N, 2) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void preprocess(clir::block_builder &bb, prepost_params pp) const override;
    void postprocess(clir::block_builder &bb, prepost_params pp) const override;

  private:
    static void preprocess_i(clir::block_builder &bb, precision_helper fph, clir::expr i,
                             clir::expr twiddle, std::size_t N, tensor_view<1u> const &x,
                             tensor_view<1u> const &X1);
};

class f2fft_gen_c2r_double : public f2fft_gen_c2r {
  public:
    f2fft_gen_c2r_double(std::size_t N) : f2fft_gen_c2r(N, 1) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void preprocess(clir::block_builder &bb, prepost_params pp) const override;
    void postprocess(clir::block_builder &bb, prepost_params pp) const override;

  private:
    static void preprocess_i(clir::block_builder &bb, precision_helper fph, clir::expr i,
                             std::size_t N, tensor_view<1u> const &xa, tensor_view<1u> const &xb,
                             tensor_view<1u> const &y);
};

} // namespace bbfft

#endif // F2FFT_GEN_20230811_HPP
