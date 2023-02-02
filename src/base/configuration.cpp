// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"

namespace bbfft {

char const *to_string(transform_type type) {
    switch (type) {
    case transform_type::c2c:
        return "c2c";
    case transform_type::r2c:
        return "r2c";
    case transform_type::c2r:
        return "c2r";
    };
    return "unknown";
}

auto default_istride(unsigned dim, std::array<std::size_t, max_tensor_dim> const &shape,
                     transform_type type, bool inplace) -> std::array<std::size_t, max_tensor_dim> {
    std::size_t shape1 = shape[1];
    switch (type) {
    case transform_type::r2c:
        shape1 = inplace ? 2 * (shape[1] / 2 + 1) : shape[1];
        break;
    case transform_type::c2r:
        shape1 = shape[1] / 2 + 1;
        break;
    case transform_type::c2c:
        shape1 = shape[1];
        break;
    }
    std::array<std::size_t, max_tensor_dim> stride = {1, shape[0], shape1 * shape[0]};
    for (unsigned d = 1; d < dim; ++d) {
        stride[d + 2] = shape[d + 1] * stride[d + 1];
    }
    return stride;
}

auto default_ostride(unsigned dim, std::array<std::size_t, max_tensor_dim> const &shape,
                     transform_type type, bool inplace) -> std::array<std::size_t, max_tensor_dim> {
    switch (type) {
    case transform_type::r2c:
        return default_istride(dim, shape, transform_type::c2r, inplace);
    case transform_type::c2r:
        return default_istride(dim, shape, transform_type::r2c, inplace);
    case transform_type::c2c:
        return default_istride(dim, shape, transform_type::c2c, inplace);
    }
    return {};
}

void configuration::set_strides_default(bool inplace) {
    istride = default_istride(dim, shape, type, inplace);
    ostride = default_ostride(dim, shape, type, inplace);
}

} // namespace bbfft
