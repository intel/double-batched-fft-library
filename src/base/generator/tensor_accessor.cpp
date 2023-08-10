// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "tensor_accessor.hpp"

using namespace clir;

namespace bbfft {

zero_accessor::zero_accessor(precision fp) : fph_(fp) {}
expr zero_accessor::operator()(expr const &) const { return fph_.zero(); }
expr zero_accessor::store(expr, expr const &) const { return nullptr; }
auto zero_accessor::subview(block_builder &, expr const &) const
    -> std::shared_ptr<tensor_accessor> {
    return std::make_shared<zero_accessor>(*this);
}

array_accessor::array_accessor(expr x, data_type type, int component)
    : x_(std::move(x)), type_(std::move(type)), component_(component) {}

expr array_accessor::operator()(expr const &offset) const {
    auto e = x_[offset];
    return component_ >= 0 ? e.s(component_) : e;
}
expr array_accessor::store(expr value, expr const &offset) const {
    return assignment(this->operator()(offset), std::move(value));
}
auto array_accessor::subview(block_builder &bb, expr const &offset) const
    -> std::shared_ptr<tensor_accessor> {
    auto e = bb.declare_assign(pointer_to(type_), "sub", x_ + offset);
    return std::make_shared<array_accessor>(e, type_);
}

callback_accessor::callback_accessor(expr x, data_type type, char const *load, char const *store,
                                     expr offset)
    : x_(std::move(x)), type_(std::move(type)), load_(load), store_(store),
      offset_(std::move(offset)) {}

expr callback_accessor::operator()(expr const &offset) const {
    if (load_) {
        return call(load_, {x_, offset});
    }
    return x_[offset];
}
expr callback_accessor::store(expr value, expr const &offset) const {
    if (store_) {
        return call(store_, {x_, offset, std::move(value)});
    }
    return assignment(x_[offset], std::move(value));
}
auto callback_accessor::subview(block_builder &bb, expr const &offset) const
    -> std::shared_ptr<tensor_accessor> {
    auto off = bb.declare_assign(generic_size(), "offset", offset_ + offset);
    return std::make_shared<callback_accessor>(x_, type_, load_, store_, std::move(off));
}

} // namespace bbfft
