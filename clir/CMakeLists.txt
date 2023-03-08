# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

project(clir VERSION 0.3.0 LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests" ON)

add_subdirectory(src)

enable_testing()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
