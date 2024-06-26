// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DATA_TYPE_20220405_HPP
#define DATA_TYPE_20220405_HPP

#include "clir/builtin_type.hpp"
#include "clir/export.hpp"
#include "clir/handle.hpp"

#include <cstddef>
#include <memory>

#define CLIR_MAKE_DATA_TYPE_SHORTCUTS(TYPE)                                                        \
    inline data_type generic_##TYPE(type_qualifier qs = type_qualifier::none) {                    \
        return data_type(builtin_type::TYPE##_t, address_space::generic_t, qs);                    \
    }                                                                                              \
    inline data_type generic_##TYPE(short size, type_qualifier qs = type_qualifier::none) {        \
        return data_type(builtin_type::TYPE##_t, size, address_space::generic_t, qs);              \
    }                                                                                              \
    inline data_type global_##TYPE(type_qualifier qs = type_qualifier::none) {                     \
        return data_type(builtin_type::TYPE##_t, address_space::global_t, qs);                     \
    }                                                                                              \
    inline data_type global_##TYPE(short size, type_qualifier qs = type_qualifier::none) {         \
        return data_type(builtin_type::TYPE##_t, size, address_space::global_t, qs);               \
    }                                                                                              \
    inline data_type local_##TYPE(type_qualifier qs = type_qualifier::none) {                      \
        return data_type(builtin_type::TYPE##_t, address_space::local_t, qs);                      \
    }                                                                                              \
    inline data_type local_##TYPE(short size, type_qualifier qs = type_qualifier::none) {          \
        return data_type(builtin_type::TYPE##_t, size, address_space::local_t, qs);                \
    }                                                                                              \
    inline data_type constant_##TYPE(type_qualifier qs = type_qualifier::none) {                   \
        return data_type(builtin_type::TYPE##_t, address_space::constant_t, qs);                   \
    }                                                                                              \
    inline data_type constant_##TYPE(short size, type_qualifier qs = type_qualifier::none) {       \
        return data_type(builtin_type::TYPE##_t, size, address_space::constant_t, qs);             \
    }                                                                                              \
    inline data_type private_##TYPE(type_qualifier qs = type_qualifier::none) {                    \
        return data_type(builtin_type::TYPE##_t, address_space::private_t, qs);                    \
    }                                                                                              \
    inline data_type private_##TYPE(short size, type_qualifier qs = type_qualifier::none) {        \
        return data_type(builtin_type::TYPE##_t, size, address_space::private_t, qs);              \
    }

namespace clir {

namespace internal {
class CLIR_EXPORT data_type_node;
} // namespace internal

class CLIR_EXPORT data_type : public handle<internal::data_type_node> {
  public:
    using handle<internal::data_type_node>::handle;

    data_type(builtin_type basic_data_type, address_space as = address_space::generic_t,
              type_qualifier qs = type_qualifier::none);
    data_type(builtin_type basic_data_type, short size, address_space as = address_space::generic_t,
              type_qualifier qs = type_qualifier::none);

  private:
    static auto make_type(builtin_type basic_data_type, short size, address_space as,
                          type_qualifier qs) -> std::shared_ptr<internal::data_type_node>;
};

CLIR_EXPORT data_type pointer_to(data_type ty, address_space as = address_space::generic_t,
                                 type_qualifier qs = type_qualifier::none);
CLIR_EXPORT data_type array_of(data_type ty, std::size_t size);

CLIR_MAKE_DATA_TYPE_SHORTCUTS(bool)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(char)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(uchar)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(short)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(ushort)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(int)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(uint)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(long)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(ulong)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(float)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(double)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(half)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(size)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(ptrdiff)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(intptr)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(uintptr)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(void)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_int)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_uint)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_long)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_ulong)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_float)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_double)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_intptr)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_uintptr)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_size)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_ptrdiff)
CLIR_MAKE_DATA_TYPE_SHORTCUTS(atomic_half)

template <typename T> inline data_type generic_number(type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, address_space::generic_t, qs);
}
template <typename T>
inline data_type generic_number(short size, type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, size, address_space::generic_t, qs);
}
template <typename T> inline data_type global_number(type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, address_space::global_t, qs);
}
template <typename T>
inline data_type global_number(short size, type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, size, address_space::global_t, qs);
}
template <typename T> inline data_type local_number(type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, address_space::local_t, qs);
}
template <typename T>
inline data_type local_number(short size, type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, size, address_space::local_t, qs);
}
template <typename T> inline data_type constant_number(type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, address_space::constant_t, qs);
}
template <typename T>
inline data_type constant_number(short size, type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, size, address_space::constant_t, qs);
}
template <typename T> inline data_type private_number(type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, address_space::private_t, qs);
}
template <typename T>
inline data_type private_number(short size, type_qualifier qs = type_qualifier::none) {
    return data_type(to_builtin_type_v<T>, size, address_space::private_t, qs);
}

} // namespace clir

#endif // DATA_TYPE_20220405_HPP
