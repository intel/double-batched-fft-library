# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

project(clir VERSION 0.5.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(BUILD_TESTING "Build unit tests" ON)

add_subdirectory(src)

enable_testing()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
