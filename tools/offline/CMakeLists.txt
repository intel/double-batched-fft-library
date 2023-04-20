# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(GNUInstallDirs)

add_executable(bbfft-offline-generate main.cpp args.cpp)
target_link_libraries(bbfft-offline-generate PRIVATE bbfft-base tools-common)
target_compile_features(bbfft-offline-generate PRIVATE cxx_std_17)

install(TARGETS bbfft-offline-generate
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

