// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "mixed_radix_fft.hpp"
#include "bbfft/tensor_indexer.hpp"
#include "root_of_unity.hpp"
#include "scrambler.hpp"

#include "clir/builder.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"
#include "clir/stmt.hpp"
#include "clir/var.hpp"

#include <cstddef>
#include <type_traits>
#include <unordered_map>

using namespace clir;

namespace bbfft {

expr complex_mul::operator()(expr c, std::complex<double> w) {
    auto wr = fph_.constant(w.real());
    auto wi = fph_.constant(w.imag());
    return init_vector(fph_.type(2), {c.s(0) * wr - c.s(1) * wi, c.s(0) * wi + c.s(1) * wr});
}

expr complex_mul::operator()(expr c, expr w) {
    return init_vector(fph_.type(2),
                       {c.s(0) * w.s(0) - c.s(1) * w.s(1), c.s(0) * w.s(1) + c.s(1) * w.s(0)});
}

int product(std::vector<int> const &factorization) {
    int N = 1;
    for (auto f : factorization) {
        N *= f;
    }
    return N;
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

void generate_fft::with_cse(block_builder &bb, precision fp, int direction,
                            std::vector<int> factorization, var &x, var &y, expr is_odd) {
    int L = factorization.size();
    int N = product(factorization);
    int J = N;
    int K = 1;
    auto fph = precision_helper(fp);

    for (int f = L - 1; f >= 0; --f) {
        auto Nf = factorization[f];
        J /= Nf;
        auto source = tensor_indexer<int, 3>({Nf, J, K});
        auto target = tensor_indexer<int, 3>({J, Nf, K});
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {

                for (int kf = 0; kf < Nf; ++kf) {
                    bb.assign(y[target(j, kf, k)], x[source(0, j, k)]);
                }
                for (int jf = 1; jf < Nf; ++jf) {
                    auto pair_hash = [Nf](std::pair<int, int> const &p) {
                        return p.first * Nf + p.second;
                    };
                    auto unique_products =
                        std::unordered_map<std::pair<int, int>, std::pair<var, var>,
                                           decltype(pair_hash)>(Nf * Nf, pair_hash);

                    for (int kf = 0; kf < Nf; ++kf) {
                        auto [zz, NN] = simplify_power_of_w(direction * kf * jf, Nf);
                        auto absZZ = std::make_pair(abs(zz), NN);

                        auto ref = unique_products.find(absZZ);
                        if (ref == unique_products.end()) {
                            auto w = power_of_w(absZZ);
                            auto xc = var("xc");
                            auto xs = var("xs");
                            bb.declare_assign(fph.type(), xc,
                                              sub_group_xc(fph, x[source(jf, j, k)], w));
                            bb.declare_assign(fph.type(), xs,
                                              sub_group_xs(fph, x[source(jf, j, k)], w, is_odd));

                            unique_products[absZZ] = std::make_pair(std::move(xc), std::move(xs));
                            ref = unique_products.find(absZZ);
                        }
                        bb.add(add_into(y[target(j, kf, k)],
                                        add_copy_sign(ref->second.first, ref->second.second, zz)));
                    }
                }
                for (int kf = 0; kf < Nf; ++kf) {
                    auto w = power_of_w(direction * kf * j, J * Nf);
                    if (w.real() != 1.0f || w.imag() != 0.0f) {
                        bb.assign(y[target(j, kf, k)],
                                  sub_group_xy(fph, y[target(j, kf, k)], w, is_odd));
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

void generate_fft::basic(block_builder &bb, precision fp, int direction,
                         std::vector<int> factorization, var &x, var &y, expr is_odd,
                         expr twiddle) {
    int L = factorization.size();
    int N = product(factorization);
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

void generate_fft::basic_inplace(block_builder &bb, precision fp, int direction,
                                 std::vector<int> factorization, var x, expr twiddle) {
    int L = factorization.size();
    int N = product(factorization);
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
                for (int kf = 0; kf < Nf; ++kf) {
                    expr esum = x[indexer(j, 0, k)];
                    for (int jf = 1; jf < Nf; ++jf) {
                        auto w = power_of_w(direction * kf * jf, Nf);
                        esum = esum + cmul(x[indexer(j, jf, k)], w);
                    }
                    bb.assign(y[kf], esum);
                    auto tw = power_of_w(direction * kf * j, J * Nf);
                    if (bool(twiddle) && f == 0) {
                        auto tw_idx = scramble(indexer(j, kf, k));
                        auto tw_tmp = bb.declare_assign(fph.type(2), "tw_tmp",
                                                        cmul(twiddle[tw_idx], tw));
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

void generate_fft::basic_inplace_subgroup(block_builder &bb, precision fp, int direction,
                                          std::vector<int> factorization, var x, expr is_odd,
                                          expr twiddle) {
    int L = factorization.size();
    int N = product(factorization);
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
