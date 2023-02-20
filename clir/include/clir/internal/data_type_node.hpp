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
    scalar_data_type(builtin_type type, address_space space = address_space::generic_t)
        : type_(type), space_(space) {}

    builtin_type type() { return type_; }
    address_space space() { return space_; }

  private:
    builtin_type type_;
    address_space space_;
};

class CLIR_EXPORT vector_data_type : public visitable<vector_data_type, data_type_node> {
  public:
    vector_data_type(builtin_type type, short size, address_space space = address_space::generic_t)
        : type_(type), size_(size), space_(space) {}

    builtin_type type() { return type_; }
    short size() const { return size_; }
    address_space space() { return space_; }

  private:
    builtin_type type_;
    short size_;
    address_space space_;
};

class CLIR_EXPORT pointer : public visitable<pointer, data_type_node> {
  public:
    pointer(data_type ty, address_space space = address_space::generic_t)
        : ty_(std::move(ty)), space_(space) {}

    data_type &ty() { return ty_; }
    address_space space() { return space_; }

  private:
    data_type ty_;
    address_space space_;
};

class CLIR_EXPORT array : public visitable<array, data_type_node> {
  public:
    array(data_type ty, std::size_t size) : ty_(std::move(ty)), size_(size) {}

    data_type &ty() { return ty_; }
    std::size_t size() const { return size_; }

  private:
    data_type ty_;
    std::size_t size_;
};

} // namespace clir::internal

#endif // DATA_TYPE_NODE_20220405_HPP
