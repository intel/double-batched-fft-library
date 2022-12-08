# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "-std=c++17")
    add_executable(fft3d-cufft fft3d-cufft.cpp main.cpp args.cpp)
    target_compile_features(fft3d-cufft PRIVATE cxx_std_17)
    target_link_libraries(fft3d-cufft PRIVATE CUDA::cudart CUDA::cufftw)
    set_property(TARGET fft3d-cufft PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

if(BUILD_SYCL)
    find_package(SYCL REQUIRED)

    add_executable(fft3d-sycl fft3d-sycl.cpp main.cpp args.cpp)
    add_sycl_to_target(TARGET fft3d-sycl SOURCES main.cpp)
    target_link_libraries(fft3d-sycl PRIVATE test-bench-sycl)
    target_compile_features(fft3d-sycl PRIVATE cxx_std_17)
endif()

if(BUILD_LEVEL_ZERO)
    add_executable(fft3d-level-zero fft3d-level-zero.cpp main.cpp args.cpp)
    target_link_libraries(fft3d-level-zero PRIVATE test-bench-level-zero)
    target_compile_features(fft3d-level-zero PRIVATE cxx_std_17)
endif()

if(BUILD_OPENCL)
    add_executable(fft3d-opencl fft3d-opencl.cpp main.cpp args.cpp)
    target_link_libraries(fft3d-opencl PRIVATE test-bench-opencl)
    target_compile_features(fft3d-opencl PRIVATE cxx_std_17)
endif()

