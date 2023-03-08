# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)
include(${PROJECT_SOURCE_DIR}/external/doctest/cmake/doctest.cmake)

add_library(clir-test-lib STATIC main.cpp)
target_include_directories(clir-test-lib PUBLIC ${PROJECT_SOURCE_DIR}/external)
target_compile_features(clir-test-lib PUBLIC cxx_std_17)
if(ENABLE_WARNINGS)
    target_compile_options(clir-test-lib PUBLIC -Wall -Wextra -Wpedantic -Wundefined-func-template)
endif()

# clir tests

add_executable(test-util util.cpp)
target_link_libraries(test-util PRIVATE clir-test-lib clir::clir)
doctest_discover_tests(test-util)

add_executable(test-visitor visitor.cpp)
target_link_libraries(test-visitor PRIVATE clir-test-lib clir::clir)
doctest_discover_tests(test-visitor)

