# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function(add_test name)
    add_executable(${name} ${name}.cpp args.cpp main-sycl.cpp ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_17)
    if (NO_DOUBLE_PRECISION)
        target_compile_definitions(${name} PRIVATE NO_DOUBLE_PRECISION)
    endif()
endfunction()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "-std=c++17")
    add_executable(test-cufft test-cufft.cpp initialize.cu args.cpp main-cuda.cpp)
    target_compile_features(test-cufft PRIVATE cxx_std_17)
    target_include_directories(test-cufft PRIVATE ${BBFFT_SOURCE_DIR}/include)
    target_link_libraries(test-cufft PRIVATE CUDA::cudart CUDA::cufftw)
    set_property(TARGET test-cufft PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

if(USE_MKL)
    find_package(SYCL REQUIRED)

    add_test(test-mkl)
    target_compile_options(test-mkl PRIVATE -qmkl)
    target_link_options(test-mkl PRIVATE -qmkl)
    target_link_libraries(test-mkl PRIVATE SYCL::SYCL)
    target_include_directories(test-mkl PRIVATE ${BBFFT_SOURCE_DIR}/include)
endif()

if(BUILD_SYCL)
    find_package(SYCL REQUIRED)

    add_test(test-double-batched-fft)
    target_link_libraries(test-double-batched-fft PRIVATE bbfft-sycl SYCL::SYCL)
endif()

if(USE_VKFFT)
    find_package(SYCL REQUIRED)
    find_package(OpenCL REQUIRED)
    find_package(VkFFT REQUIRED)
    add_test(test-vkfft vkfft_error.cpp)
    target_link_libraries(test-vkfft PRIVATE bbfft-opencl OpenCL::OpenCL VkFFT::VkFFT SYCL::SYCL)
    target_compile_definitions(test-vkfft PRIVATE -DVKFFT_BACKEND=3 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS)
endif()
