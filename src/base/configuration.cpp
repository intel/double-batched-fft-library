// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"

#include <ostream>
#include <sstream>
#include <stdexcept>

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

std::string configuration::to_string() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, configuration const &cfg) {
    // precision
    switch (cfg.fp) {
    case precision::f32:
        os << 's';
        break;
    case precision::f64:
        os << 'd';
        break;
    default:
        throw std::runtime_error("Unsupported precision");
        break;
    }
    // domain
    switch (cfg.type) {
    case transform_type::c2c:
        os << 'c';
        break;
    case transform_type::r2c:
    case transform_type::c2r:
        os << 'r';
        break;
    default:
        throw std::runtime_error("Unsupported transform type");
        break;
    }
    // direction
    switch (cfg.dir) {
    case direction::forward:
        os << 'f';
        break;
    case direction::backward:
        os << 'b';
        break;
    default:
        throw std::runtime_error("Unsupported direction");
        break;
    }
    if ((cfg.type == transform_type::r2c && cfg.dir == direction::backward) ||
        (cfg.type == transform_type::c2r && cfg.dir == direction::forward)) {
        throw std::runtime_error(
            "r2c direction must be forward and c2r direction must be backward");
    }

    // placement
    auto istride = cfg.istride;
    auto ostride = cfg.ostride;
    if (cfg.type == transform_type::r2c) {
        for (unsigned d = 2; d < max_tensor_dim; ++d) {
            ostride[d] *= 2;
        }
    } else if (cfg.type == transform_type::c2r) {
        for (unsigned d = 2; d < max_tensor_dim; ++d) {
            istride[d] *= 2;
        }
    }
    bool inplace = true;
    for (unsigned d = 0; d < cfg.dim + 2; ++d) {
        inplace = inplace && (istride[d] == ostride[d]);
    }
    os << (inplace ? 'i' : 'o');

    // shape
    if (cfg.shape[0] != 1u) {
        os << cfg.shape[0] << '.';
    }
    os << cfg.shape[1];
    for (unsigned d = 1; d < cfg.dim; ++d) {
        os << 'x' << cfg.shape[1 + d];
    }
    if (cfg.shape[cfg.dim + 1] != 1u) {
        os << '*' << cfg.shape[cfg.dim + 1];
    }

    // istride
    if (cfg.istride != default_istride(cfg.dim, cfg.shape, cfg.type, inplace)) {
        os << 'i' << cfg.istride[0];
        for (unsigned d = 1; d < cfg.dim + 2; ++d) {
            os << ',' << cfg.istride[d];
        }
    }

    // ostride
    if (cfg.ostride != default_ostride(cfg.dim, cfg.shape, cfg.type, inplace)) {
        os << 'o' << cfg.ostride[0];
        for (unsigned d = 1; d < cfg.dim + 2; ++d) {
            os << ',' << cfg.ostride[d];
        }
    }

    return os;
}

} // namespace bbfft
