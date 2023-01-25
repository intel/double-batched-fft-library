# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

project(double_batched_fft_library VERSION 0.3.5 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)
option(NO_DOUBLE_PRECISION "Disable double precision benchmark and tests" OFF)
option(USE_CUDA "Compile CUDA tests" OFF)
option(USE_MKL "Compile MKL benchmark" OFF)
option(USE_VKFFT "Compile VkFFT benchmark" OFF)
option(ENABLE_WARNINGS "Enable compiler warnings" OFF)
option(BUILD_DOCUMENTATION "Generate documentation" OFF)
option(BUILD_BENCHMARK "Build the benchmark" ON)
option(BUILD_EXAMPLE "Build the examples" ON)
option(BUILD_TESTING "Build unit tests" ON)
option(BUILD_SYCL "Build Double-Batched FFT Library for SYCL" ON)
cmake_dependent_option(BUILD_LEVEL_ZERO
    "Build Double-Batched FFT Library for Level Zero; required when SYCL build is enabled"
    ON "NOT BUILD_SYCL" ON)
cmake_dependent_option(BUILD_OPENCL
    "Build Double-Batched FFT Library for Level Zero; required when SYCL build is enabled"
    ON "NOT BUILD_SYCL" ON)

if(DEFINED clir_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${clir_SHARED_LIBS})
endif()
if(DEFINED bbfft_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${bbfft_SHARED_LIBS}")
endif()
if (NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
if(BUILD_EXAMPLE)
    add_subdirectory(examples)
endif()

if(BUILD_DOCUMENTATION)
    add_subdirectory(docs)
endif()
add_subdirectory(src)

enable_testing()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
