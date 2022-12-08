// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REAL_TYPE_20220513_H
#define REAL_TYPE_20220513_H

template <typename T> struct real_type {
    using type = T;
};
template <typename T> struct real_type<std::complex<T>> {
    using type = T;
};

#endif // REAL_TYPE_20220513_H
