// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "mixed_radix_fft.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "math.hpp"
#include "root_of_unity.hpp"
#include "scrambler.hpp"

#include "clir/builder.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

using namespace clir;

namespace bbfft {

expr complex_mul::operator()(expr c, std::complex<double> const &w) {
    auto wr = fph_.constant(w.real());
    auto wi = fph_.constant(w.imag());
    return init_vector(fph_.type(2), {c.s(0) * wr - c.s(1) * wi, c.s(0) * wi + c.s(1) * wr});
}

expr complex_mul::operator()(expr c, expr w) {
    return init_vector(fph_.type(2),
                       {c.s(0) * w.s(0) - c.s(1) * w.s(1), c.s(0) * w.s(1) + c.s(1) * w.s(0)});
}

expr complex_mul::pair_real(clir::expr x1, clir::expr x2, std::complex<double> const &w) {
    auto wr = fph_.constant(w.real());
    return wr * (x1 + x2);
}

expr complex_mul::pair_imag(clir::expr x1, clir::expr x2, std::complex<double> const &w) {
    auto wi = fph_.constant(w.imag());
    return wi * init_vector(fph_.type(2), {x2.s(1) - x1.s(1), x1.s(0) - x2.s(0)});
}

clir::expr basic_esum::operator()(block_builder &, int kf) {
    expr esum = x_(0);
    for (int jf = 1; jf < Nf_; ++jf) {
        auto w = power_of_w(direction_ * kf * jf, Nf_);
        esum = esum + cmul_(x_(jf), w);
    }
    return esum;
}

/*
 * Suppose we have a pair
 *
 * p1 = w_N(k) * x(j1) + w_N(-k) * x(j2)
 *
 * and let wr_N(k) = Re(w_N(k)), wi_N(j) = Im(w_N(k))
 *
 * then we have
 *
 * p1 = wr_N(k) * x(j1) + wi_N(k) * i * x(j1) + wr_N(k) * x(j2) - wi_N(k) * i * x(j2)
 *    = wr_N(k) * (x(j1) + x(j2)) + wi_N(k) * i * (x(j1) - x(j2))
 *    = p11 + p12
 *
 * where p11 := wr_N(k) * (x(j1) + x(j2)), p12 :=  wi_N(k) * i * (x(j1) - x(j2))
 *
 * Hence, instead of 8 mults and 10 adds we only need 6 adds and 4 mults.
 *
 * Moreover, we have
 *
 * p2 = w_N(-k) * x(j1) + w_N(k) * x(j2)
 *    = wr_N(k) * (x(j1) + x(j2)) + wi_N(k) * i * (x(j2) - x(j1))
 *    = p11 - p12
 */
clir::expr pair_optimization_esum::operator()(block_builder &bb, int kf) {
    if (kf == 0) {
        return basic_esum::operator()(bb, 0);
    }
    expr esum = x_(0);
    auto singletons = std::vector<product>{};
    auto pairs = std::vector<std::pair<product, product>>{};
    for (int jf = 1; jf < Nf_; ++jf) {
        auto w_arg = simplify_power_of_w(direction_ * kf * jf, Nf_);
        auto w_arg_other = std::make_pair(-w_arg.first, w_arg.second);
        if (auto it = std::find_if(std::begin(singletons), std::end(singletons),
                                   [&](product const &p) { return w_arg_other == p.w_arg; });
            it != std::end(singletons)) {
            pairs.push_back(std::make_pair(*it, product{w_arg, jf}));
            singletons.erase(it);
        } else {
            singletons.push_back(product{w_arg, jf});
        }
    }

    for (auto const &s : singletons) {
        auto w = power_of_w(s.w_arg);
        esum = esum + cmul_(x_(s.jf), w);
    }
    for (auto const &p : pairs) {
        auto p1 = p.first;
        auto p2 = p.second;
        p1.w_arg.first *= -1;
        p2.w_arg.first *= -1;
        auto other_pair = std::make_pair(p1, p2);
        if (auto it = available_pairs_.find(other_pair); it != available_pairs_.end()) {
            esum = esum + it->second.first - it->second.second;
        } else {
            p1 = p.first;
            p2 = p.second;
            if (p1.w_arg.first < 0) {
                std::swap(p1, p2);
            }
            auto x1 = x_(p1.jf);
            auto x2 = x_(p2.jf);
            auto w = power_of_w(p1.w_arg);
            auto v1 = bb.declare_assign(cmul_.fph().type(2), "p1", cmul_.pair_real(x1, x2, w));
            auto v2 = bb.declare_assign(cmul_.fph().type(2), "p2", cmul_.pair_imag(x1, x2, w));
            available_pairs_[p] = std::make_pair(v1, v2);
            esum = esum + v1 + v2;
        }
    }
    return esum;
}

expr multiply_imaginary_unit(precision_helper fph, expr x, expr is_odd) {
    return select(intel_sub_group_shuffle_down(-x, -x, 1), intel_sub_group_shuffle_up(x, x, 1),
                  cast(fph.select_type(), is_odd));
}

expr sub_group_xc(precision_helper fph, expr x, std::complex<double> y) {
    return x * fph.constant(y.real());
}
expr sub_group_xs(precision_helper fph, expr x, std::complex<double> y, expr &is_odd) {
    return multiply_imaginary_unit(fph, x * fph.constant(y.imag()), is_odd);
}

expr add_copy_sign(expr x, expr y, int sign) {
    if (sign < 0) {
        return x - y;
    }
    return x + y;
}

expr sub_group_xy(precision_helper fph, expr x, std::complex<double> y, expr &is_odd) {
    auto xc = sub_group_xc(fph, x, y);
    auto xs = sub_group_xs(fph, x, y, is_odd);
    return xc + xs;
}

void generate_fft::basic(block_builder &bb, precision fp, int direction,
                         std::vector<int> factorization, var &x, var &y, expr is_odd,
                         expr twiddle) {
    int L = factorization.size();
    int N = product(factorization.begin(), factorization.end(), 1);
    int J = N;
    int K = 1;
    auto fph = precision_helper(fp);

    for (int f = L - 1; f >= 0; --f) {
        auto Nf = factorization[f];
        J /= Nf;
        auto source = tensor_indexer<int, 3, layout::col_major>({K, J, Nf});
        auto target = tensor_indexer<int, 3, layout::col_major>({K, Nf, J});
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                for (int kf = 0; kf < Nf; ++kf) {
                    expr ersum = x[source(k, j, 0)];
                    expr eisum = fph.zero();
                    for (int jf = 1; jf < Nf; ++jf) {
                        auto w = power_of_w(direction * kf * jf, Nf);
                        ersum = ersum + x[source(k, j, jf)] * fph.constant(w.real());
                        eisum = eisum + x[source(k, j, jf)] * fph.constant(w.imag());
                    }
                    auto re = var("re");
                    auto im = var("im");
                    bb.declare_assign(fph.type(), re, ersum);
                    bb.declare_assign(fph.type(), im, eisum);
                    auto tw = power_of_w(direction * kf * j, J * Nf);
                    auto const twiddle_and_store = [&](expr tw_re, expr tw_im) {
                        auto tmp = var("tmp");
                        bb.declare_assign(fph.type(), tmp, re * tw_im + im * tw_re);
                        bb.assign(y[target(k, kf, j)],
                                  re * tw_re - im * tw_im +
                                      multiply_imaginary_unit(fph, tmp, is_odd));
                    };
                    auto tw_real = fph.constant(tw.real());
                    auto tw_imag = fph.constant(tw.imag());
                    if (bool(twiddle) && f == 0) {
                        auto t1 = var("tw_re");
                        auto t2 = var("tw_im");
                        auto tw_idx = target(k, kf, j);
                        bb.declare_assign(fph.type(), t1,
                                          tw_real * twiddle[tw_idx][0] -
                                              tw_imag * twiddle[tw_idx][1]);
                        bb.declare_assign(fph.type(), t2,
                                          tw_real * twiddle[tw_idx][1] +
                                              tw_imag * twiddle[tw_idx][0]);
                        twiddle_and_store(t1, t2);
                    } else {
                        twiddle_and_store(tw_real, tw_imag);
                    }
                }
            }
        }
        K *= Nf;
        std::swap(x, y);
    }
    // ensure x=input and y=output
    std::swap(x, y);
}

template <typename ESum>
void inplace(clir::block_builder &bb, precision fp, int direction, std::vector<int> factorization,
             clir::var x, clir::expr twiddle = nullptr) {
    int L = factorization.size();
    int N = product(factorization.begin(), factorization.end(), 1);
    int J = N;
    int K = 1;
    auto fph = precision_helper(fp);
    auto scramble = scrambler(factorization);

    auto cmul = complex_mul(fph);

    for (int f = L - 1; f >= 0; --f) {
        auto Nf = factorization[f];
        J /= Nf;
        auto y = bb.declare(array_of(fph.type(2), Nf), "y");
        auto indexer = tensor_indexer<int, 3, layout::col_major>({J, Nf, K});
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                auto esum = ESum(fph, direction, Nf, [&](int jf) { return x[indexer(j, jf, k)]; });
                for (int kf = 0; kf < Nf; ++kf) {
                    bb.assign(y[kf], esum(bb, kf));
                    auto tw = power_of_w(direction * kf * j, J * Nf);
                    if (bool(twiddle) && f == 0) {
                        auto tw_idx = scramble(indexer(j, kf, k));
                        auto tw_tmp =
                            bb.declare_assign(fph.type(2), "tw_tmp", cmul(twiddle[tw_idx], tw));
                        bb.assign(y[kf], cmul(y[kf], tw_tmp));
                    } else {
                        bb.assign(y[kf], cmul(y[kf], tw));
                    }
                }
                for (int kf = 0; kf < Nf; ++kf) {
                    bb.assign(x[indexer(j, kf, k)], y[kf]);
                }
            }
        }
        K *= Nf;
    }
}

void generate_fft::basic_inplace(block_builder &bb, precision fp, int direction,
                                 std::vector<int> factorization, var x, expr twiddle) {
    inplace<basic_esum>(bb, std::move(fp), std::move(direction), std::move(factorization),
                        std::move(x), std::move(twiddle));
}

void generate_fft::pair_optimization_inplace(block_builder &bb, precision fp, int direction,
                                             std::vector<int> factorization, var x, expr twiddle) {
    inplace<pair_optimization_esum>(bb, std::move(fp), std::move(direction),
                                    std::move(factorization), std::move(x), std::move(twiddle));
}

void generate_fft::basic_inplace_subgroup(block_builder &bb, precision fp, int direction,
                                          std::vector<int> factorization, var x, expr is_odd,
                                          expr twiddle) {
    int L = factorization.size();
    int N = product(factorization.begin(), factorization.end(), 1);
    int J = N;
    int K = 1;
    auto fph = precision_helper(fp);
    auto scramble = scrambler(factorization);

    for (int f = L - 1; f >= 0; --f) {
        auto Nf = factorization[f];
        J /= Nf;
        auto y = bb.declare(array_of(fph.type(), Nf), "y");
        auto indexer = tensor_indexer<int, 3, layout::col_major>({J, Nf, K});
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                for (int kf = 0; kf < Nf; ++kf) {
                    expr ersum = x[indexer(j, 0, k)];
                    expr eisum = 0.0f;
                    for (int jf = 1; jf < Nf; ++jf) {
                        auto w = power_of_w(direction * kf * jf, Nf);
                        ersum = ersum + x[indexer(j, jf, k)] * fph.constant(w.real());
                        eisum = eisum + x[indexer(j, jf, k)] * fph.constant(w.imag());
                    }
                    auto re = var("re");
                    auto im = var("im");
                    bb.declare_assign(fph.type(), re, ersum);
                    bb.declare_assign(fph.type(), im, eisum);
                    auto tw = power_of_w(direction * kf * j, J * Nf);
                    auto const twiddle_and_store = [&](expr tw_re, expr tw_im) {
                        auto tmp = bb.declare_assign(fph.type(), "tmp", re * tw_im + im * tw_re);
                        bb.assign(y[kf], re * tw_re - im * tw_im +
                                             multiply_imaginary_unit(fph, tmp, is_odd));
                    };
                    auto tw_real = fph.constant(tw.real());
                    auto tw_imag = fph.constant(tw.imag());
                    if (bool(twiddle) && f == 0) {
                        auto t1 = var("tw_re");
                        auto t2 = var("tw_im");
                        auto tw_idx = scramble(indexer(j, kf, k));
                        bb.declare_assign(fph.type(), t1,
                                          tw_real * twiddle[tw_idx][0] -
                                              tw_imag * twiddle[tw_idx][1]);
                        bb.declare_assign(fph.type(), t2,
                                          tw_real * twiddle[tw_idx][1] +
                                              tw_imag * twiddle[tw_idx][0]);
                        twiddle_and_store(t1, t2);
                    } else {
                        twiddle_and_store(tw_real, tw_imag);
                    }
                }
                for (int kf = 0; kf < Nf; ++kf) {
                    bb.assign(x[indexer(j, kf, k)], y[kf]);
                }
            }
        }
        K *= Nf;
    }
}

} // namespace bbfft
