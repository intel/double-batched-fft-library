// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TENSOR_INDEXER_20220407_HPP
#define TENSOR_INDEXER_20220407_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

namespace bbfft {

/**
 * @brief Tensor storage layout
 *
 * Row major format: The mode varying fastest in memory is specified last
 *
 * Column major format: The mode varying fastest in memory is specified first
 */
enum class layout {
    row_major, ///< Row major layout
    col_major  ///< Column major layout
};

/**
 * @brief Truncate or extend std::array
 *
 * @tparam Dout Length of output array
 * @tparam IdxT Array value type
 * @tparam Din Length of input array
 * @param in Input array
 * @param fill_value Fill value if extension is needed
 *
 * @return Truncated or extended array
 */
template <std::size_t Dout, typename IdxT, std::size_t Din>
constexpr auto fit_array(std::array<IdxT, Din> const &in, IdxT fill_value = IdxT(0)) {
    std::array<IdxT, Dout> out;
    constexpr unsigned int d_min = std::min(Din, Dout);
    for (unsigned int d = 0; d < d_min; ++d) {
        out[d] = in[d];
    }
    for (unsigned int d = d_min; d < Dout; ++d) {
        out[d] = fill_value;
    }
    return out;
}

/**
 * @brief Utility class to compute addresses for multi-dimensional data
 *
 * @tparam IdxT Index type
 * @tparam D Tensor dimension
 * @tparam L Storage layout
 */
template <typename IdxT, unsigned int D, layout L = layout::row_major> class tensor_indexer {
  public:
    using multi_idx_t = std::array<IdxT, D>; ///< Data type of multi-indices

    /**
     * @brief Construct empty indexer
     */
    tensor_indexer() : shape_{}, stride_{} {}

    /**
     * @brief Construct indexer for \f$N_1\times\dots\times N_D\f$ tensor
     *
     * Strides are computed automatically assuming packed data as following:
     *
     * Row-major layout: \f$s = (N_{D-1}\cdot\ldots\cdot N_2, \dots, N_{D-1}, 1)\f$
     *
     * Column-major layout: \f$s = (1, N_1, \dots, N_1\cdot\ldots\cdot N_{D-1})\f$
     *
     * @param shape The numbers \f$N_1,\dots,N_D\f$
     */
    tensor_indexer(multi_idx_t shape) : shape_(std::move(shape)) {
        if constexpr (L == layout::row_major) {
            stride_[D - 1] = 1;
            for (int i = D - 2; i >= 0; --i) {
                stride_[i] = stride_[i + 1] * shape_[i + 1];
            }
        } else {
            stride_[0] = 1;
            for (unsigned int i = 1; i < D; ++i) {
                stride_[i] = stride_[i - 1] * shape_[i - 1];
            }
        }
    }

    /**
     * @brief Construct indexer for \f$N_1\times\dots\times N_D\f$ tensor with manual strides
     *
     * @param shape The numbers \f$N_1,\dots,N_D\f$
     * @param stride Custom strides
     */
    tensor_indexer(multi_idx_t shape, multi_idx_t stride)
        : shape_(std::move(shape)), stride_(std::move(stride)) {}

    /**
     * @brief Compute linear index for entry \f$(i_1, ..., i_D)\f$.
     *
     * Indices are computed as following:
     * \f[a = \sum_{j=1}^D i_j s_j,\f]
     * where \f$s_j\f$ are the entries of the stride array.
     *
     * @tparam Indices Index types of arguments
     * @param ...is Indices
     *
     * @return Linear index
     */
    template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == D, int>>
    IdxT operator()(Indices &&...is) const {
        return linear_index(is...);
    }

    /**
     * @brief Compute linear index
     *
     * @param idx Multi-index
     *
     * @return Linear index
     */
    IdxT operator()(multi_idx_t const &idx) const {
        IdxT r = 0;
        for (unsigned int i = 0; i < D; ++i) {
            r = r + idx[i] * stride_[i];
        }
        return r;
    }

    /**
     * @brief Tensor shape
     *
     * @return Numbers \f$N_1,\dots,N_D\f$
     */
    auto shape() const { return shape_; }
    /**
     * @brief Tensor shape
     *
     * @param d mode
     *
     * @return Number \f$N_d\f$
     */
    auto shape(unsigned int d) const { return shape_[d]; }
    /**
     * @brief Strides
     *
     * @return Stride array
     */
    auto stride() const { return stride_; }
    /**
     * @brief Stride for d-th mode
     *
     * @param d mode
     *
     * @return Stride
     */
    auto stride(unsigned int d) const { return stride_[d]; }
    /**
     * @brief Compute number of elements in tensor
     *
     * @return Size (multiply with element type to get number of bytes)
     */
    IdxT size() const {
        if constexpr (L == layout::row_major) {
            return stride_.front() * shape_.front();
        } else {
            return stride_.back() * shape_.back();
        }
    }
    /**
     * @brief Dimension
     *
     * @return Dimension
     */
    constexpr auto dim() const { return D; }

    /**
     * @brief Checks whether indices may be fused
     *
     * "Fusing" means that one may treat 2 or more neighbouring indices as a single super-index.
     * Modes may only be fused it they are packed in memory.
     *
     * @tparam Dfrom First fused mode
     * @tparam Dto Last fused mode
     *
     * @return True if modes may be fused
     */
    template <unsigned int Dfrom = 0, unsigned int Dto = D - 1> bool may_fuse() const {
        static_assert(Dfrom <= D - 1);
        static_assert(Dfrom <= Dto);
        static_assert(Dto <= D - 1);
        if constexpr (Dfrom < Dto) {
            if constexpr (L == layout::row_major) {
                for (unsigned int d = Dfrom + 1u; d <= Dto; ++d) {
                    bool ok = stride_[d] * shape_[d] == stride_[d - 1];
                    if (!ok) {
                        return false;
                    }
                }
            } else {
                for (unsigned int d = Dfrom; d < Dto; ++d) {
                    bool ok = stride_[d] * shape_[d] == stride_[d + 1];
                    if (!ok) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Returns a tensor indexer with fused modes
     *
     * @see may_fuse()
     *
     * @tparam Dfrom First fused mode
     * @tparam Dto Last fused mode
     *
     * @return True if modes may be fused
     */
    template <unsigned int Dfrom = 0, unsigned int Dto = D - 1> auto fused() const {
        static_assert(Dfrom <= D - 1);
        static_assert(Dfrom <= Dto);
        static_assert(Dto <= D - 1);
        auto new_shape = std::array<IdxT, D - (Dto - Dfrom)>{};
        auto new_stride = std::array<IdxT, D - (Dto - Dfrom)>{};
        std::copy(shape_.begin(), shape_.begin() + Dfrom, new_shape.begin());
        std::copy(stride_.begin(), stride_.begin() + Dfrom + 1, new_stride.begin());
        std::copy(shape_.begin() + Dto + 1, shape_.begin() + D, new_shape.begin() + Dfrom + 1);
        std::copy(stride_.begin() + Dto + 1, stride_.begin() + D, new_stride.begin() + Dfrom + 1);
        auto N = std::reduce(shape_.begin() + Dfrom, shape_.begin() + Dto + 1, 1,
                             std::multiplies<unsigned int>{});
        new_shape[Dfrom] = N;
        return tensor_indexer<IdxT, D - (Dto - Dfrom), L>(new_shape, new_stride);
    }

  private:
    template <typename Head> IdxT linear_index(Head head) const { return head * stride_[D - 1u]; }
    template <typename Head, typename... Tail> IdxT linear_index(Head head, Tail... tail) const {
        constexpr auto d = (D - 1u) - sizeof...(Tail);
        return linear_index(tail...) + head * stride_[d];
    }

    multi_idx_t shape_;
    multi_idx_t stride_;
};

/**
 * @brief Shortcut for 1-D tensor_indexer
 *
 * @tparam IdxT Index type
 * @tparam L Storage layout
 */
template <typename IdxT, layout L = layout::row_major>
using vector_indexer = tensor_indexer<IdxT, 1u, L>;
/**
 * @brief Shortcut for 2-D tensor_indexer
 *
 * @tparam IdxT Index type
 * @tparam L Storage layout
 */
template <typename IdxT, layout L = layout::row_major>
using matrix_indexer = tensor_indexer<IdxT, 2u, L>;

} // namespace bbfft

#endif // TENSOR_INDEXER_20220407_HPP
