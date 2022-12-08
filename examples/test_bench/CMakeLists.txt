# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_OPENCL)
    add_library(test-bench-opencl STATIC test_bench_opencl.cpp)
    target_include_directories(test-bench-opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-bench-opencl PUBLIC bbfft-opencl)
endif()

if(BUILD_LEVEL_ZERO)
    add_library(test-bench-level-zero STATIC test_bench_level_zero.cpp)
    target_include_directories(test-bench-level-zero PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-bench-level-zero PUBLIC bbfft-level-zero)
endif()

if(BUILD_SYCL)
    find_package(SYCL REQUIRED)

    add_library(test-bench-sycl STATIC test_bench_sycl.cpp)
    target_include_directories(test-bench-sycl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-bench-sycl PUBLIC bbfft-sycl SYCL::SYCL)
endif()
