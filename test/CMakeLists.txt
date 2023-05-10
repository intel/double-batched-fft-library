# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)
include(${PROJECT_SOURCE_DIR}/external/doctest/cmake/doctest.cmake)

add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/external)
target_compile_features(test-lib PUBLIC cxx_std_17)
add_common_flags(test-lib)
if (NO_DOUBLE_PRECISION)
    target_compile_definitions(test-lib PUBLIC NO_DOUBLE_PRECISION)
endif()

# fft tests

add_executable(test-codegen codegen.cpp)
target_link_libraries(test-codegen PRIVATE test-lib clir::clir bbfft-private-test bbfft-base)
doctest_discover_tests(test-codegen)

add_executable(test-parser parser.cpp)
target_link_libraries(test-parser PRIVATE test-lib bbfft-base)
doctest_discover_tests(test-parser)

add_executable(test-tensor tensor.cpp)
target_link_libraries(test-tensor PRIVATE test-lib bbfft-base)
doctest_discover_tests(test-tensor)

if(BUILD_SYCL)
    find_package(SYCL)

    add_executable(test-c2c c2c.cpp)
    target_link_libraries(test-c2c PRIVATE test-lib bbfft-sycl SYCL::SYCL)
    doctest_discover_tests(test-c2c)

    add_executable(test-r2c r2c.cpp)
    target_link_libraries(test-r2c PRIVATE test-lib bbfft-sycl SYCL::SYCL)
    doctest_discover_tests(test-r2c)

    add_executable(test-callback callback.cpp)
    target_link_libraries(test-callback PRIVATE test-lib bbfft-sycl SYCL::SYCL)
    doctest_discover_tests(test-callback)

    add_executable(test-error error.cpp)
    target_link_libraries(test-error PRIVATE test-lib bbfft-sycl SYCL::SYCL)
    doctest_discover_tests(test-error)
endif()
