// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "clir/data_type.hpp"
#include "clir/internal/data_type_node.hpp"

#include <memory>
#include <utility>

namespace clir {

data_type::data_type(builtin_type basic_data_type, address_space as)
    : handle(make_type(basic_data_type, 1, as)) {}
data_type::data_type(builtin_type basic_data_type, short size, address_space as)
    : handle(make_type(basic_data_type, size, as)) {}

auto data_type::make_type(builtin_type basic_data_type, short size, address_space as)
    -> std::shared_ptr<internal::data_type_node> {
    if (size > 1) {
        return std::make_shared<internal::vector_data_type>(basic_data_type, size, as);
    }
    return std::make_shared<internal::scalar_data_type>(basic_data_type, as);
}

data_type pointer_to(data_type ty, address_space as) {
    return data_type(std::make_shared<internal::pointer>(std::move(ty), as));
}

data_type array_of(data_type ty, std::size_t size) {
    return data_type(std::make_shared<internal::array>(std::move(ty), size));
}

} // namespace clir
