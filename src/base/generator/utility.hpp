// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UTILITY_20221205_HPP
#define UTILITY_20221205_HPP

#include "bbfft/configuration.hpp"
#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/expr.hpp"

namespace bbfft {

clir::builtin_type precision_to_builtin_type(precision fp);
short precision_to_bits(precision fp);

class precision_helper {
  public:
    precision_helper(precision fp);
    clir::builtin_type cl_type() const;
    short bits() const;
    clir::data_type type(clir::address_space as = clir::address_space::generic_t) const;
    clir::data_type type(short size, clir::address_space as = clir::address_space::generic_t) const;
    clir::data_type select_type() const;
    clir::expr constant(double value) const;
    clir::expr zero() const;

  private:
    precision fp_;
};

} // namespace bbfft

#endif // UTILITY_20221205_HPP
