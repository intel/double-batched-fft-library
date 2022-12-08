// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ALLOCATOR_20220318_H
#define ALLOCATOR_20220318_H

#include <cstddef>

class allocator {
  public:
    virtual ~allocator() {}

    virtual void *malloc_shared(std::size_t numBytes) = 0;
    virtual void free(void *ptr) = 0;
};

template <typename T> T *malloc_shared(std::size_t count, allocator &alloc) {
    return reinterpret_cast<T *>(alloc.malloc_shared(count * sizeof(T)));
}

#endif // ALLOCATOR_20220318_H
