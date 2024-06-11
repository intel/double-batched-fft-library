# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

add_executable(large_fft1d main.cpp args.cpp fft1d.cpp test_signal.cpp)
target_compile_features(large_fft1d PRIVATE cxx_std_17)
target_link_libraries(large_fft1d PRIVATE bbfft-base bbfft-opencl)

