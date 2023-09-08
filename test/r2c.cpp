// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "fft.hpp"

#include "bbfft/configuration.hpp"
#include "bbfft/sycl/make_plan.hpp"
#include "bbfft/tensor_indexer.hpp"

#include <complex>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

using namespace bbfft;
using namespace sycl;

template <std::size_t D>
auto extract_shape(tensor_indexer<std::size_t, 2u + D, layout::col_major> const &xi) {
    std::array<std::size_t, D> shape;
    for (std::size_t d = 0; d < D; ++d) {
        shape[d] = xi.shape(d + 1);
    }
    return shape;
}

template <typename T, std::size_t D> class harness {
  public:
    constexpr static std::size_t tensor_dim = 2u + D;
    using indexer = tensor_indexer<std::size_t, tensor_dim, layout::col_major>;

    harness(std::array<std::size_t, D> N) : N_(N) {}

    static auto scale(std::size_t k, std::size_t K) { return T(1.0) + k / T(K); };
    static auto basis_no(std::size_t m, std::size_t k, std::size_t N) -> long {
        return (m + k) % N;
    };

    void init_forward(queue Q, T *x, indexer const &xi) {
        auto N = N_;
        REQUIRE(xi.template may_fuse<1u, D>());
        auto xi_f = xi.template fused<1u, D>();
        auto shape = extract_shape<D>(xi);
        std::size_t M = xi_f.shape(0), NN = xi_f.shape(1), K = xi_f.shape(2);
        Q.parallel_for(range{K, NN, M}, [=](id<3> idx) {
             auto k = idx[0], n = idx[1], m = idx[2];
             T arg = outer_product(
                 [&m, &k](auto nd, auto Nd) {
                     return std::cos((T(tau) / Nd) * basis_no(m, k, Nd) * nd) / T(Nd);
                 },
                 unflatten(n, shape), N);
             x[xi_f(m, n, k)] = scale(k, K) * arg;
         }).wait();
    }

    void init_backward(queue Q, std::complex<T> *X, indexer const &Xi,
                       bool pollute_0_imaginary = false) {
        auto N = N_;
        REQUIRE(Xi.template may_fuse<1u, D>());
        auto shape = extract_shape<D>(Xi);
        auto Xi_f = Xi.template fused<1u, D>();
        std::size_t M = Xi_f.shape(0), NN = Xi_f.shape(1), K = Xi_f.shape(2);
        Q.parallel_for(range{K, NN, M}, [=](id<3> idx) {
             auto k = idx[0], n = idx[1], m = idx[2];
             T delta = outer_product(
                 [&m, &k](auto nd, auto Nd) {
                     long ndl = static_cast<long>(nd);
                     return (periodic_delta<T>(ndl - basis_no(m, k, Nd), Nd) +
                             periodic_delta<T>(ndl + basis_no(m, k, Nd), Nd)) /
                            T(2.0);
                 },
                 unflatten(n, shape), N);
             X[Xi_f(m, n, k)] = {scale(k, K) * delta, T(0.0)};
         }).wait();
        if (pollute_0_imaginary) {
            Q.parallel_for(range{K, M}, [=](id<2> idx) {
                 auto k = idx[0], m = idx[1];
                 X[Xi_f(m, 0, k)].imag(T(1.0) + m + k);
             }).wait();
        }
    }

    void check_forward(std::complex<T> *X, indexer const &Xi) {
        REQUIRE(Xi.template may_fuse<1u, D>());
        auto Xi_f = Xi.template fused<1u, D>();
        auto shape = extract_shape<D>(Xi);
        std::size_t M = Xi_f.shape(0), NN = Xi_f.shape(1), K = Xi_f.shape(2);
        double eps = tol<T>(N_);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t n = 0; n < NN; ++n) {
                for (std::size_t m = 0; m < M; ++m) {
                    T ref = outer_product(
                        [&m, &k](auto nd, auto Nd) {
                            long ndl = static_cast<long>(nd);
                            return (periodic_delta<T>(ndl - basis_no(m, k, Nd), Nd) +
                                    periodic_delta<T>(ndl + basis_no(m, k, Nd), Nd)) /
                                   T(2.0);
                        },
                        unflatten(n, shape), N_);
                    ref *= scale(k, K);
                    REQUIRE(X[Xi_f(m, n, k)].real() == doctest::Approx(ref).epsilon(eps));
                    REQUIRE(X[Xi_f(m, n, k)].imag() == doctest::Approx(T(0.0)).epsilon(eps));
                }
            }
        }
    }

    void check_backward(T *x, indexer const &xi) {
        REQUIRE(xi.template may_fuse<1u, D>());
        auto xi_f = xi.template fused<1u, D>();
        auto shape = extract_shape<D>(xi);
        std::size_t M = xi_f.shape(0), NN = xi_f.shape(1), K = xi_f.shape(2);
        double eps = tol<T>(N_);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t n = 0; n < NN; ++n) {
                for (std::size_t m = 0; m < M; ++m) {
                    auto nds = unflatten(n, shape);
                    if (nds[0] < N_[0]) {
                        T ref = outer_product(
                            [&m, &k](auto nd, auto Nd) {
                                return std::cos((T(tau) / Nd) * basis_no(m, k, Nd) * nd);
                            },
                            nds, N_);
                        ref *= scale(k, K);
                        REQUIRE(x[xi_f(m, n, k)] == doctest::Approx(ref).epsilon(eps));
                    }
                }
            }
        }
    }

  private:
    std::array<std::size_t, D> N_;
};

template <typename T, std::size_t D, bool Inplace = false>
void r2c_forward(std::size_t M, std::array<std::size_t, D> N, std::size_t K) {
    auto Q = queue();

    auto h = harness<T, D>(N);

    std::array<std::size_t, bbfft::max_tensor_dim> shape = {};
    shape[0] = M;
    for (std::size_t d = 0; d < D; ++d) {
        shape[d + 1] = N[d];
    }
    shape[D + 1] = K;
    configuration cfg = {
        D,                  // dim
        shape,              // shape
        to_precision_v<T>,  // precision
        direction::forward, // direction
        transform_type::r2c // r2c
    };
    cfg.set_strides_default(Inplace);
    shape[1] = Inplace ? 2 * (N[0] / 2 + 1) : N[0];
    auto xi = tensor_indexer<std::size_t, h.tensor_dim, layout::col_major>(
        fit_array<h.tensor_dim>(shape), fit_array<h.tensor_dim>(cfg.istride));
    shape[1] = N[0] / 2 + 1;
    auto Xi = tensor_indexer<std::size_t, h.tensor_dim, layout::col_major>(
        fit_array<h.tensor_dim>(shape), fit_array<h.tensor_dim>(cfg.ostride));

    auto x = malloc_device<T>(xi.size(), Q);
    auto X_device = Inplace ? reinterpret_cast<std::complex<T> *>(x)
                            : malloc_device<std::complex<T>>(Xi.size(), Q);
    auto X = new std::complex<T>[Xi.size()];
    auto plan = make_plan(cfg, Q);

    h.init_forward(Q, x, xi);
    plan.execute(x, X_device).wait();
    Q.copy(X_device, X, Xi.size()).wait();
    h.check_forward(X, Xi);

    free(x, Q);
    if (!Inplace) {
        free(X_device, Q);
    }
    delete[] X;
}

TEST_CASE_TEMPLATE("r2c 1d out-of-place", T, TEST_PRECISIONS) {
    auto Q = queue();

    auto KK = std::vector<std::size_t>{1, 33};
    auto MM = std::vector<std::size_t>{1, 3, 32};
    auto NN = std::vector<std::size_t>{2, 4, 8, 16, 32, 128, 256, 512, 102, 220, 10, 26};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 1u, false>(M, {N}, K);
}

TEST_CASE_TEMPLATE("r2c 1d in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{65};
    auto MM = std::vector<std::size_t>{1, 3};
    auto NN = std::vector<std::size_t>{4, 12, 110};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 1u, true>(M, {N}, K);
}

TEST_CASE_TEMPLATE("r2c 2d in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 33};
    auto MM = std::vector<std::size_t>{1, 3};
    auto NN = std::vector<std::array<std::size_t, 2u>>{{4, 8}, {8, 5}};

    std::size_t M, K;
    std::array<std::size_t, 2> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 2u, true>(M, N, K);
}

TEST_CASE_TEMPLATE("r2c 3d in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 33};
    auto MM = std::vector<std::size_t>{1, 3};
    auto NN = std::vector<std::array<std::size_t, 3u>>{{4, 8, 2}, {8, 256, 5}};

    std::size_t M, K;
    std::array<std::size_t, 3> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 3u, true>(M, N, K);
}

TEST_CASE_TEMPLATE("r2c 2d out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 65};
    auto MM = std::vector<std::size_t>{1, 7};
    auto NN = std::vector<std::array<std::size_t, 2u>>{{5, 4}, {10, 3}};

    std::size_t M, K;
    std::array<std::size_t, 2> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 2u, false>(M, N, K);
}

TEST_CASE_TEMPLATE("r2c 3d out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 65};
    auto MM = std::vector<std::size_t>{1, 7};
    auto NN = std::vector<std::array<std::size_t, 3u>>{{5, 4, 6}, {10, 286, 3}};

    std::size_t M, K;
    std::array<std::size_t, 3> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    r2c_forward<T, 3u, false>(M, N, K);
}

template <typename T, std::size_t D, bool Inplace = false>
void c2r_backward(std::size_t M, std::array<std::size_t, D> N, std::size_t K,
                  bool pollute_0 = false) {
    auto Q = queue();

    auto h = harness<T, D>(N);

    std::array<std::size_t, bbfft::max_tensor_dim> shape = {};
    shape[0] = M;
    for (std::size_t d = 0; d < D; ++d) {
        shape[d + 1] = N[d];
    }
    shape[D + 1] = K;
    configuration cfg = {
        D,                   // dim
        shape,               // shape
        to_precision_v<T>,   // precision
        direction::backward, // direction
        transform_type::c2r  // c2r
    };
    cfg.set_strides_default(Inplace);
    shape[1] = N[0] / 2 + 1;
    auto Xi = tensor_indexer<std::size_t, h.tensor_dim, layout::col_major>(
        fit_array<h.tensor_dim>(shape), fit_array<h.tensor_dim>(cfg.istride));
    shape[1] = Inplace ? 2 * (N[0] / 2 + 1) : N[0];
    auto xi = tensor_indexer<std::size_t, h.tensor_dim, layout::col_major>(
        fit_array<h.tensor_dim>(shape), fit_array<h.tensor_dim>(cfg.ostride));

    auto X = malloc_device<std::complex<T>>(Xi.size(), Q);
    auto x_device = Inplace ? reinterpret_cast<T *>(X) : malloc_device<T>(xi.size(), Q);
    auto x = new T[xi.size()];
    auto plan = make_plan(cfg, Q);

    h.init_backward(Q, X, Xi, pollute_0);
    plan.execute(X, x_device).wait();
    Q.copy(x_device, x, xi.size()).wait();
    h.check_backward(x, xi);

    free(X, Q);
    if (!Inplace) {
        free(x_device, Q);
    }
    delete[] x;
}

TEST_CASE_TEMPLATE("c2r 1d out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{33};
    auto MM = std::vector<std::size_t>{1, 5, 32};
    auto NN = std::vector<std::size_t>{16, 96, 256, 300};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 1u, false>(M, {N}, K);
}

TEST_CASE_TEMPLATE("c2r 1d in-place polluted 0", T, TEST_PRECISIONS) {
    /* The DFT of a real sequence x has conjugate-even symmetry, hence imag(X[0]) = 0.
     * Here we inject a non-zero imaginary part to test whether it is ignored by the FFT.
     * (Note that a non-zero imaginary part is faulty usage but we want to match the behaviour
     * of other FFT libraries.)
     */
    auto KK = std::vector<std::size_t>{33};
    auto MM = std::vector<std::size_t>{1};
    auto NN = std::vector<std::size_t>{16, 48, 512};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 1u, true>(M, {N}, K, true);
}

TEST_CASE_TEMPLATE("c2r 2d in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 54};
    auto MM = std::vector<std::size_t>{1, 3};
    auto NN = std::vector<std::array<std::size_t, 2u>>{{4, 2}, {96, 96}};

    std::size_t M, K;
    std::array<std::size_t, 2> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 2u, true>(M, N, K);
}

TEST_CASE_TEMPLATE("c2r 3d in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 54};
    auto MM = std::vector<std::size_t>{1, 3};
    auto NN = std::vector<std::array<std::size_t, 3u>>{{4, 8, 2}, {96, 96, 80}};

    std::size_t M, K;
    std::array<std::size_t, 3> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 3u, true>(M, N, K);
}

TEST_CASE_TEMPLATE("c2r 2d out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 65};
    auto MM = std::vector<std::size_t>{1, 7};
    auto NN = std::vector<std::array<std::size_t, 2u>>{{5, 4}, {10, 3}};

    std::size_t M, K;
    std::array<std::size_t, 2> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 2u, false>(M, N, K);
}

TEST_CASE_TEMPLATE("c2r 3d out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 65};
    auto MM = std::vector<std::size_t>{1, 7};
    auto NN = std::vector<std::array<std::size_t, 3u>>{{5, 4, 6}, {10, 286, 3}};

    std::size_t M, K;
    std::array<std::size_t, 3> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    c2r_backward<T, 3u, false>(M, N, K);
}


template <typename T, std::size_t D, bool Inplace = false>
void identity_test(std::size_t M, std::array<std::size_t, D> N, std::size_t K) {
    auto Q = queue();

    auto rd = std::random_device{};
    auto gen = std::mt19937(rd());
    auto Y = std::uniform_real_distribution<T>(0.0, 1.0);

    constexpr std::size_t tensor_dim = 2u + D;

    std::array<std::size_t, tensor_dim> xi_shape = {};
    std::array<std::size_t, tensor_dim> Xi_shape = {};
    xi_shape[0] = M;
    xi_shape[1] = Inplace ? 2 * (N[0] / 2 + 1) : N[0];
    for (std::size_t d = 1; d < D; ++d) {
        xi_shape[d + 1] = N[d];
    }
    xi_shape[tensor_dim - 1] = K;
    Xi_shape[0] = M;
    Xi_shape[1] = N[0] / 2 + 1;
    for (std::size_t d = 1; d < D; ++d) {
        Xi_shape[d + 1] = N[d];
    }
    Xi_shape[tensor_dim - 1] = K;
    auto xi = tensor_indexer<std::size_t, tensor_dim, layout::col_major>(xi_shape);
    auto Xi = tensor_indexer<std::size_t, tensor_dim, layout::col_major>(Xi_shape);
    auto x_ref = new T[xi.size()];
    auto x_device = malloc_device<T>(xi.size(), Q);
    auto X_device = Inplace ? reinterpret_cast<std::complex<T> *>(x_device)
                            : malloc_device<std::complex<T>>(Xi.size(), Q);
    auto x_host = new T[xi.size()];

    xi_shape[1] = N[0];
    configuration cfg = {D,
                         fit_array<bbfft::max_tensor_dim>(xi_shape),
                         to_precision_v<T>,
                         direction::forward,
                         transform_type::r2c,
                         fit_array<bbfft::max_tensor_dim>(xi.stride()),
                         fit_array<bbfft::max_tensor_dim>(Xi.stride())};
    auto plan = make_plan(cfg, Q);

    configuration icfg = {D,
                          fit_array<bbfft::max_tensor_dim>(xi_shape),
                          to_precision_v<T>,
                          direction::backward,
                          transform_type::c2r,
                          fit_array<bbfft::max_tensor_dim>(Xi.stride()),
                          fit_array<bbfft::max_tensor_dim>(xi.stride())};
    auto iplan = make_plan(icfg, Q);

    for (std::size_t j = 0; j < xi.size(); ++j) {
        x_ref[j] = Y(gen);
    }

    Q.copy(x_ref, x_device, xi.size()).wait();
    plan.execute(x_device, X_device).wait();
    iplan.execute(X_device, x_device).wait();
    Q.copy(x_device, x_host, xi.size()).wait();

    std::size_t normalization = 1;
    for (std::size_t d = 0; d < D; ++d) {
        normalization *= N[d];
    }

    double eps = tol<T, D>(N);
    REQUIRE(xi.template may_fuse<1u, D>());
    auto xi_f = xi.template fused<1u, D>();
    auto shape = extract_shape<D>(xi);
    std::size_t NN = xi_f.shape(1);
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < NN; ++n) {
            for (std::size_t m = 0; m < M; ++m) {
                auto nds = unflatten(n, shape);
                if (nds[0] < N[0]) {
                    REQUIRE(x_ref[xi_f(m, n, k)] ==
                            doctest::Approx(x_host[xi_f(m, n, k)] / normalization).epsilon(eps));
                }
            }
        }
    }

    free(x_device, Q);
    if (!Inplace) {
        free(X_device, Q);
    }
    delete[] x_host;
    delete[] x_ref;
}

TEST_CASE_TEMPLATE("r2c 1d identity in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{16};
    auto MM = std::vector<std::size_t>{1, 8};
    auto NN = std::vector<std::size_t>{8, 64, 212};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    identity_test<T, 1u, true>(M, {N}, K);
}

TEST_CASE_TEMPLATE("r2c 1d identity out-of-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{16};
    auto MM = std::vector<std::size_t>{1, 8};
    auto NN = std::vector<std::size_t>{8, 64, 212};

    std::size_t M, N, K;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    identity_test<T, 1u, false>(M, {N}, K);
}

TEST_CASE_TEMPLATE("r2c 3d identity in-place", T, TEST_PRECISIONS) {
    auto KK = std::vector<std::size_t>{1, 11};
    auto MM = std::vector<std::size_t>{1, 7};
    auto NN = std::vector<std::array<std::size_t, 3u>>{{8, 8, 8}, {96, 96, 80}};

    std::size_t M, K;
    std::array<std::size_t, 3> N;
    DOCTEST_TENSOR3_TEST(MM, NN, KK);

    identity_test<T, 3u, true>(M, N, K);
}
