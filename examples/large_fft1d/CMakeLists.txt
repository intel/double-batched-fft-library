# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(OpenMP)

add_executable(large_fft1d main.cpp args.cpp fft1d.cpp fft1d_custom.cpp fft1d_custom2.cpp test_signal.cpp)
target_compile_features(large_fft1d PRIVATE cxx_std_17)
target_link_libraries(large_fft1d PRIVATE bbfft-base bbfft-opencl)

if(OpenMP_FOUND)
    target_link_libraries(large_fft1d PRIVATE OpenMP::OpenMP_CXX)
endif()

