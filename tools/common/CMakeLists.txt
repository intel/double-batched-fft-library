# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

add_library(tools-common STATIC info.cpp)
target_link_libraries(tools-common PRIVATE bbfft-base)
target_include_directories(tools-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
