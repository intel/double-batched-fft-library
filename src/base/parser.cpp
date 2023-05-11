// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/parser.hpp"

#include <cctype>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bbfft {

class error_formatter {
  public:
    error_formatter(std::string_view &desc, std::string_view::const_iterator &it)
        : desc(desc), it(it) {}
    auto operator()(std::string const &message, bool pos_indicator = true) {
        std::ostringstream oss;
        oss << "==> " << desc << " is malformed: " << message;
        if (pos_indicator) {
            auto pos = 4 + std::distance(desc.cbegin(), it);
            oss << std::endl << std::string(pos, ' ') << "^";
        }
        return oss.str();
    }

  private:
    std::string_view &desc;
    std::string_view::const_iterator &it;
};

class number_parser {
  public:
    number_parser(std::string_view &desc, std::string_view::const_iterator &it,
                  bool ignore_leading_whitespace = false)
        : desc(desc), it(it), ignore_leading_whitespace(ignore_leading_whitespace), err(desc, it) {}
    auto operator()() {
        while (ignore_leading_whitespace && it != desc.cend() && std::isblank(*it)) {
            ++it;
        }
        auto begin = it;
        while (it != desc.cend() && '0' <= *it && *it <= '9') {
            ++it;
        }
        if (begin == it) {
            throw std::runtime_error(err("expected number"));
        }
        return std::stoi(std::string(begin, it));
    }

  private:
    std::string_view &desc;
    std::string_view::const_iterator &it;
    bool ignore_leading_whitespace;
    error_formatter err;
};

configuration parse_fft_descriptor(std::string_view desc) {
    configuration cfg = {};
    bool inplace = false;

    auto it = desc.cbegin();
    auto format_error = error_formatter(desc, it);
    auto parse_number = number_parser(desc, it);
    bool was_at_end = false;
    auto const advance = [&]() {
        was_at_end = it == desc.cend();
        char val = '\0';
        if (it != desc.cend()) {
            val = *it++;
        }
        return val;
    };
    auto const fail = [&](std::string const &message) {
        if (!was_at_end) {
            --it;
        }
        throw std::runtime_error(format_error(message));
    };
    auto const expected = [&](const char *c) { fail("expected " + std::string(c)); };

    switch (advance()) {
    case 's':
        cfg.fp = precision::f32;
        break;
    case 'd':
        cfg.fp = precision::f64;
        break;
    default:
        expected("'s' (single) or 'd' (double)");
        break;
    }
    switch (advance()) {
    case 'c':
        cfg.type = transform_type::c2c;
        break;
    case 'r':
        cfg.type = transform_type::r2c;
        break;
    default:
        expected("'c' (complex) or 'r' (real)");
        break;
    }
    switch (advance()) {
    case 'f':
        cfg.dir = direction::forward;
        break;
    case 'b':
        cfg.dir = direction::backward;
        if (cfg.type == transform_type::r2c) {
            cfg.type = transform_type::c2r;
        }
        break;
    default:
        expected("'f' (forward) or 'b' (backward)");
        break;
    }
    switch (advance()) {
    case 'i':
        inplace = true;
        break;
    case 'o':
        inplace = false;
        break;
    default:
        expected("'i' (in-place) or 'o' (out-of-place)");
        break;
    }

    std::array<char, max_tensor_dim - 1> ops = {};
    int dim = 0;
    cfg.shape[dim++] = parse_number();
    while (it != desc.cend()) {
        if (dim >= static_cast<int>(max_tensor_dim)) {
            throw std::runtime_error(format_error("tensor dimension must not be larger than " +
                                                      std::to_string(max_tensor_dim),
                                                  false));
        }
        auto val = advance();
        if (val != '.' && val != 'x' && val != '*') {
            --it;
            break;
        }
        ops[dim - 1] = val;
        cfg.shape[dim++] = parse_number();
    }

    bool has_M = dim >= 2 && ops[0] == '.';
    bool has_K = dim >= 2 && ops[dim - 2] == '*';
    unsigned num_x = 0;
    for (int i = (has_M ? 1 : 0); i < dim - 1 - (has_K ? 1 : 0); ++i) {
        if (ops[i] == 'x') {
            ++num_x;
        } else {
            throw std::runtime_error(format_error("'.' or '*' must only appear at the beginning or "
                                                  "end of the tensor shape, respectively",
                                                  false));
        }
    }
    cfg.dim = 1 + num_x;

    if (cfg.dim > max_fft_dim) {
        throw std::runtime_error(
            format_error("only " + std::to_string(max_fft_dim - 1) + " 'x' are supported", false));
    }
    if (!has_K) {
        cfg.shape[dim++] = 1u;
    }
    if (!has_M) {
        for (int i = dim - 1; i >= 0; --i) {
            cfg.shape[i + 1] = cfg.shape[i];
        }
        cfg.shape[0] = 1u;
    }

    auto const parse_stride = [&](std::array<std::size_t, max_tensor_dim> &stride) {
        for (unsigned d = 0; d < cfg.dim + 2; ++d) {
            stride[d] = parse_number();
            if (d < cfg.dim + 1 && advance() != ',') {
                expected(",");
            }
        }
    };

    bool custom_istride = false, custom_ostride = false;
    while (it != desc.cend()) {
        switch (advance()) {
        case 'i':
            parse_stride(cfg.istride);
            custom_istride = true;
            break;
        case 'o':
            parse_stride(cfg.ostride);
            custom_ostride = true;
            break;
        default:
            expected("'i' (istride) or 'o' (ostride)");
            break;
        }
    }

    if (!custom_istride) {
        cfg.istride = default_istride(cfg.dim, cfg.shape, cfg.type, inplace);
    }
    if (!custom_ostride) {
        cfg.ostride = default_ostride(cfg.dim, cfg.shape, cfg.type, inplace);
    }

    return cfg;
}

device_info parse_device_info(std::string_view desc) {
    device_info info = {};

    auto it = desc.cbegin();
    auto format_error = error_formatter(desc, it);
    auto parse_number = number_parser(desc, it, true);
    char val;
    bool was_at_end = false;
    auto const advance = [&]() {
        was_at_end = it == desc.cend();
        char val = ' ';
        while (it != desc.cend() && std::isblank(val)) {
            val = *it++;
        }
        return val;
    };
    auto const expect = [&](char c) {
        if (advance() != c) {
            if (!was_at_end) {
                --it;
            }
            throw std::runtime_error(format_error("expected '" + std::string(1, c) + "'"));
        }
    };

    expect('{');
    info.max_work_group_size = parse_number();
    expect(',');
    expect('{');
    info.subgroup_sizes.emplace_back(parse_number());
    while ((val = advance()) && val == ',') {
        info.subgroup_sizes.emplace_back(parse_number());
    }
    if (val != '}') {
        if (!was_at_end) {
            --it;
        }
        throw std::runtime_error(format_error("expected ']'"));
    }
    expect(',');
    info.local_memory_size = parse_number();
    expect(',');
    val = advance();
    if (val == 'g') {
        expect('p');
        expect('u');
        info.type = device_type::gpu;
    } else if (val == 'c') {
        expect('p');
        expect('u');
        info.type = device_type::cpu;
    } else {
        throw std::runtime_error(format_error("expected gpu or cpu"));
    }
    expect('}');

    return info;
}

} // namespace bbfft
