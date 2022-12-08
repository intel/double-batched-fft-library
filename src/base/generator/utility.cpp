// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "utility.hpp"

using namespace clir;

namespace bbfft {

builtin_type precision_to_builtin_type(precision fp) {
    builtin_type t = builtin_type::void_t;
    switch (fp) {
    case precision::f32:
        t = builtin_type::float_t;
        break;
    case precision::f64:
        t = builtin_type::double_t;
        break;
    }
    return t;
}

short precision_to_bits(precision fp) { return static_cast<int>(fp) * 8; }

precision_helper::precision_helper(precision fp) : fp_(fp) {}
builtin_type precision_helper::cl_type() const {
    builtin_type t = builtin_type::void_t;
    switch (fp_) {
    case precision::f32:
        t = builtin_type::float_t;
        break;
    case precision::f64:
        t = builtin_type::double_t;
        break;
    }
    return t;
}
short precision_helper::bits() const { return static_cast<int>(fp_) * 8; }
data_type precision_helper::type(address_space as) const { return data_type(cl_type(), as); }
data_type precision_helper::type(short size, address_space as) const {
    return data_type(cl_type(), size, as);
}
data_type precision_helper::select_type() const {
    data_type cast_type = nullptr;
    switch (fp_) {
    case precision::f32:
        cast_type = generic_uint();
        break;
    case precision::f64:
        cast_type = generic_ulong();
        break;
    }
    return cast_type;
}
expr precision_helper::constant(double value) const { return expr(value, bits()); }
expr precision_helper::zero() const { return constant(0.0); }

} // namespace bbfft
