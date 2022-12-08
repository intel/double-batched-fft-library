// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef INITIALIZE_20220429_H
#define INITIALIZE_20220429_H

#include "tensor.hpp"

template <typename U, typename V> void initialize(tensor<U, 3u> x, tensor<V, 3u> X, bool inverse);

#endif // INITIALIZE_20220429_H
