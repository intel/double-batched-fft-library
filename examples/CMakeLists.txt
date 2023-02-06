# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_SYCL)
    add_subdirectory(aot)
    add_subdirectory(cache)
endif()
add_subdirectory(fft3d)
add_subdirectory(offline_generator)
add_subdirectory(test_bench)
add_subdirectory(transpose_fft_transpose)
