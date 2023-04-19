# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(GNUInstallDirs)

find_package(SYCL REQUIRED)

add_executable(bbfft-device-info main.cpp)
add_executable(bbfft::bbfft-device-info ALIAS bbfft-device-info)
add_sycl_to_target(TARGET bbfft-device-info SOURCES main.cpp)
target_compile_features(bbfft-device-info PRIVATE cxx_std_17)
target_link_libraries(bbfft-device-info PRIVATE bbfft-base bbfft-sycl)

install(TARGETS bbfft-device-info
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
