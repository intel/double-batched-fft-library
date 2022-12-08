# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_SYCL)
    find_package(SYCL REQUIRED)

    add_executable(tft tft.cpp)
    target_link_libraries(tft PRIVATE bbfft-sycl SYCL::SYCL)
    target_compile_features(tft PRIVATE cxx_std_17)
endif()
