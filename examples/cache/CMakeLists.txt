# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(SYCL REQUIRED)

add_executable(cache main.cpp)
add_sycl_to_target(TARGET cache SOURCES main.cpp)
target_compile_features(cache PRIVATE cxx_std_17)
target_link_libraries(cache PRIVATE bbfft-sycl)
