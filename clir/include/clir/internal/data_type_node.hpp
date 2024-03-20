// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DATA_TYPE_NODE_20220405_HPP
#define DATA_TYPE_NODE_20220405_HPP

#include "clir/builtin_type.hpp"
#include "clir/data_type.hpp"
#include "clir/export.hpp"
#include "clir/virtual_type_list.hpp"

#include <cstddef>

namespace clir::internal {

class CLIR_EXPORT data_type_node
    : public virtual_type_list<class scalar_data_type, class vector_data_type, class pointer,
                               class array> {};

class CLIR_EXPORT scalar_data_type : public visitable<scalar_data_type, data_type_node> {
  public:
    inline scalar_data_type(builtin_type type, address_space space = address_space::generic_t,
                            type_qualifier qualifiers = type_qualifier::none)
        : type_(type), space_(space), qualifiers_(qualifiers) {}

    inline builtin_type type() { return type_; }
    inline address_space space() { return space_; }
    inline type_qualifier qualifiers() { return qualifiers_; }

  private:
    builtin_type type_;
    address_space space_;
    type_qualifier qualifiers_;
};

class CLIR_EXPORT vector_data_type : public visitable<vector_data_type, data_type_node> {
  public:
    inline vector_data_type(builtin_type type, short size,
                            address_space space = address_space::generic_t,
                            type_qualifier qualifiers = type_qualifier::none)
        : type_(type), size_(size), space_(space), qualifiers_(qualifiers) {}

    inline builtin_type type() { return type_; }
    inline short size() const { return size_; }
    inline address_space space() { return space_; }
    inline type_qualifier qualifiers() { return qualifiers_; }

  private:
    builtin_type type_;
    short size_;
    address_space space_;
    type_qualifier qualifiers_;
};

class CLIR_EXPORT pointer : public visitable<pointer, data_type_node> {
  public:
    inline pointer(data_type ty, address_space space = address_space::generic_t,
                   type_qualifier qualifiers = type_qualifier::none)
        : ty_(std::move(ty)), space_(space), qualifiers_(qualifiers) {}

    inline data_type &ty() { return ty_; }
    inline address_space space() { return space_; }
    inline type_qualifier qualifiers() { return qualifiers_; }

  private:
    data_type ty_;
    address_space space_;
    type_qualifier qualifiers_;
};

class CLIR_EXPORT array : public visitable<array, data_type_node> {
  public:
    inline array(data_type ty, std::size_t size) : ty_(std::move(ty)), size_(size) {}

    inline data_type &ty() { return ty_; }
    inline std::size_t size() const { return size_; }

  private:
    data_type ty_;
    std::size_t size_;
};

} // namespace clir::internal

#endif // DATA_TYPE_NODE_20220405_HPP
