# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(SYCL REQUIRED)

include(AddAotKernelsToTarget)

set(N 2 4 8 16 32 64 128 256 512 1024)
foreach(n IN LISTS N)
    list(APPEND descriptor_list "srfi${n}*16384")
endforeach()

add_executable(aot main.cpp)
add_sycl_to_target(TARGET aot SOURCES main.cpp)
target_compile_features(aot PRIVATE cxx_std_17)
#target_link_libraries(aot PRIVATE bbfft-sycl kernels)
target_link_libraries(aot PRIVATE bbfft-sycl)
add_aot_kernels_to_target(TARGET aot PREFIX kernels DEVICE pvc LIST ${descriptor_list})
