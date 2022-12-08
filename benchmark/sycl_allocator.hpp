// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SYCL_ALLOCATOR_20220318_H
#define SYCL_ALLOCATOR_20220318_H

#include "allocator.hpp"

#include <CL/sycl.hpp>
#include <utility>

class sycl_allocator : public allocator {
  public:
    sycl_allocator(sycl::queue Q) : queue(std::move(Q)) {}

    void *malloc_shared(std::size_t numBytes) override {
        return sycl::aligned_alloc_shared(64, numBytes, queue);
    }
    void free(void *ptr) override { sycl::free(ptr, queue); }

  private:
    sycl::queue queue;
};

#endif // SYCL_ALLOCATOR_20220318_H
