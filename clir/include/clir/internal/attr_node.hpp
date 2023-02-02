// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ATTR_NODE_20220405_HPP
#define ATTR_NODE_20220405_HPP

#include "clir/builtin_type.hpp"
#include "clir/export.hpp"
#include "clir/internal/args_impl.hpp"
#include "clir/virtual_type_list.hpp"

#include <array>
#include <optional>
#include <ostream>
#include <utility>

namespace clir::internal {

template <class T> class attribute;
class CLIR_EXPORT attr_node
    : public virtual_type_list<
          attribute<class work_group_size_hint>, attribute<class reqd_work_group_size>,
          attribute<class intel_reqd_sub_group_size>, attribute<class opencl_unroll_hint>,
          attribute<class aligned>, attribute<class packed>, attribute<class endian>> {
  public:
    virtual void print(std::ostream &os) = 0;
};

class CLIR_EXPORT work_group_size_hint {
  public:
    using arg_type = std::array<int, 3>;
    constexpr static const char *name = "work_group_size_hint";
};

class CLIR_EXPORT reqd_work_group_size {
  public:
    using arg_type = std::array<int, 3>;
    constexpr static const char *name = "reqd_work_group_size";
};

class CLIR_EXPORT intel_reqd_sub_group_size {
  public:
    using arg_type = int;
    constexpr static int default_arg = 8;
    constexpr static const char *name = "intel_reqd_sub_group_size";
};

class CLIR_EXPORT opencl_unroll_hint {
  public:
    using arg_type = std::optional<int>;
    constexpr static const char *name = "opencl_unroll_hint";
};

class CLIR_EXPORT aligned {
  public:
    using arg_type = std::optional<int>;
    constexpr static const char *name = "aligned";
};

class CLIR_EXPORT packed {
  public:
    using arg_type = void;
    constexpr static const char *name = "packed";
};

class CLIR_EXPORT endian {
  public:
    using arg_type = std::optional<endianess>;
    constexpr static const char *name = "endian";
};

template <class T> class attribute : public visitable<attribute<T>, attr_node> {
  public:
    template <typename... Args> attribute(Args &&...arg) : args_{std::forward<Args>(arg)...} {}
    auto args() const { return args_; }
    void print(std::ostream &os) override { os << *this; }

  private:
    args_impl<typename T::arg_type> args_;
};
template <class T> std::ostream &operator<<(std::ostream &os, attribute<T> const &attr) {
    os << T::name << attr.args();
    return os;
}

} // namespace clir::internal

#endif // ATTR_NODE_20220405_HPP
