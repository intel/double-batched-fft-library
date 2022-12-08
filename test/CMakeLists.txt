# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)
include(${CMAKE_SOURCE_DIR}/external/doctest/cmake/doctest.cmake)

add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${CMAKE_SOURCE_DIR}/external)
target_compile_features(test-lib PUBLIC cxx_std_17)
if(ENABLE_WARNINGS)
    target_compile_options(test-lib PUBLIC -Wall -Wextra -Wpedantic -Wundefined-func-template)
endif()
if (NO_DOUBLE_PRECISION)
    target_compile_definitions(test-lib PUBLIC NO_DOUBLE_PRECISION)
endif()

# clir tests

add_executable(test-util util.cpp)
target_link_libraries(test-util PRIVATE test-lib clir::clir)
doctest_discover_tests(test-util)

add_executable(test-visitor visitor.cpp)
target_link_libraries(test-visitor PRIVATE test-lib clir::clir)
doctest_discover_tests(test-visitor)

# fft tests

add_executable(test-codegen codegen.cpp)
target_link_libraries(test-codegen PRIVATE test-lib clir::clir bbfft-private-test)
doctest_discover_tests(test-codegen)

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
