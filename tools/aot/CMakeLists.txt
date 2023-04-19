# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

add_executable(bbfft-aot-generate main.cpp args.cpp info.cpp)
add_executable(bbfft::bbfft-aot-generate ALIAS bbfft-aot-generate)
target_compile_features(bbfft-aot-generate PRIVATE cxx_std_17)
target_link_libraries(bbfft-aot-generate PRIVATE bbfft-base bbfft-level-zero)
