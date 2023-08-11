// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SBFFT_GEN_20230811_HPP
#define SBFFT_GEN_20230811_HPP

#include "bbfft/detail/generator_impl.hpp"
#include "generator/snippet.hpp"
#include "generator/tensor_accessor.hpp"
#include "generator/tensor_view.hpp"
#include "generator/utility.hpp"

#include "clir/builder.hpp"
#include "clir/expr.hpp"

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string_view>

namespace bbfft {

class sbfft_gen {
  public:
    struct gen_cfg {
        std::size_t N_in, N_out, N_slm, N_fft;
        short in_components, out_components, k_stride;
    };
    sbfft_gen(gen_cfg p) : p_(p) {}
    virtual ~sbfft_gen() {}

    void generate(std::ostream &os, small_batch_configuration const &cfg,
                  std::string_view name) const;
    inline auto const &p() const { return p_; }

  protected:
    struct copy_params {
        small_batch_configuration const &cfg;
        precision_helper fph;
        tensor_view<3u> const &view;
        tensor_view<3u> const &X1_view;
        tensor_view<1u> const &X1_1d;
        tensor_view<1u> const &x_view;
        std::shared_ptr<array_accessor> x_acc;
        clir::expr mb;
        clir::expr K;
        clir::expr kb;
        clir::expr kb_odd = nullptr;
        permutation_fun P = identity;
    };

    virtual void load(clir::block_builder &bb, copy_params cp) const = 0;
    virtual void store(clir::block_builder &bb, copy_params cp) const = 0;

    void double_load(clir::block_builder &bb, copy_params cp, int k_offset) const;
    void double_store(clir::block_builder &bb, copy_params cp, int k_offset) const;

  private:
    gen_cfg p_;
};

class sbfft_gen_c2c : public sbfft_gen {
  public:
    sbfft_gen_c2c(std::size_t N) : sbfft_gen(gen_cfg{N, N, N, N, 2, 2, 1}) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void store(clir::block_builder &bb, copy_params cp) const override;
};

class sbfft_gen_r2c : public sbfft_gen {
  public:
    sbfft_gen_r2c(std::size_t N, short n_stride, short k_stride)
        : sbfft_gen(gen_cfg{N, N / 2 + 1, N / 2 + 1, N / n_stride, 1, 2, k_stride}) {}
};

class sbfft_gen_r2c_half : public sbfft_gen_r2c {
  public:
    sbfft_gen_r2c_half(std::size_t N) : sbfft_gen_r2c(N, 2, 1) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void store(clir::block_builder &bb, copy_params cp) const override;

  private:
    static void postprocess(clir::block_builder &bb, precision_helper fph, tensor_view<1u> const &y,
                            tensor_view<1u> const &X1, std::size_t N, permutation_fun P = identity);
};

class sbfft_gen_r2c_double : public sbfft_gen_r2c {
  public:
    sbfft_gen_r2c_double(std::size_t N) : sbfft_gen_r2c(N, 1, 2) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void store(clir::block_builder &bb, copy_params cp) const override;

  private:
    static void postprocess(clir::block_builder &bb, precision_helper fph, tensor_view<1u> const &y,
                            tensor_view<1u> const &X1, std::size_t N, int component,
                            permutation_fun P = identity);
};

class sbfft_gen_c2r : public sbfft_gen {
  public:
    sbfft_gen_c2r(std::size_t N, short n_stride, short k_stride)
        : sbfft_gen(gen_cfg{N / 2 + 1, N, N / 2 + 1, N / n_stride, 2, 1, k_stride}) {}
};

class sbfft_gen_c2r_half : public sbfft_gen_c2r {
  public:
    sbfft_gen_c2r_half(std::size_t N) : sbfft_gen_c2r(N, 2, 1) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void store(clir::block_builder &bb, copy_params cp) const override;

  private:
    static void preprocess(clir::block_builder &bb, precision_helper fph, tensor_view<1u> const &X1,
                           tensor_view<1u> const &x, std::size_t N);
};

class sbfft_gen_c2r_double : public sbfft_gen_c2r {
  public:
    sbfft_gen_c2r_double(std::size_t N) : sbfft_gen_c2r(N, 1, 2) {}

  protected:
    void load(clir::block_builder &bb, copy_params cp) const override;
    void store(clir::block_builder &bb, copy_params cp) const override;

  private:
    static void preprocess(clir::block_builder &bb, precision_helper fph, tensor_view<1u> const &X1,
                           tensor_view<1u> const &x, std::size_t N, int component);
};

} // namespace bbfft

#endif // SBFFT_GEN_20230811_HPP
