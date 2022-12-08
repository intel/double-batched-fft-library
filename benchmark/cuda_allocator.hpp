// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUDA_ALLOCATOR_20220318_H
#define CUDA_ALLOCATOR_20220318_H

#include "allocator.hpp"

#include <cuda_runtime_api.h>

class cuda_allocator : public allocator {
  public:
    void *malloc_shared(std::size_t numBytes) override {
        void *ptr = nullptr;
        cudaMallocManaged(&ptr, numBytes);
        return ptr;
    }
    void free(void *ptr) override { cudaFree(ptr); }
};

#endif // CUDA_ALLOCATOR_20220318_H
