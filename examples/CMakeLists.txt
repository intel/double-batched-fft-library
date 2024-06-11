# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_SYCL)
    add_subdirectory(aot)
    add_subdirectory(cache)
endif()
if(BUILD_OPENCL)
    add_subdirectory(large_fft1d)
endif()
add_subdirectory(fft3d)
add_subdirectory(test_bench)
add_subdirectory(transpose_fft_transpose)
