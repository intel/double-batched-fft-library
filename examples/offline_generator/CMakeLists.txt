# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

add_executable(offline_generator offline_generator.cpp)
target_link_libraries(offline_generator PRIVATE bbfft-base)
target_compile_features(offline_generator PRIVATE cxx_std_17)

