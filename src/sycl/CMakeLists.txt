# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_SYCL)
    include(CommonOptions)
    include(InstallLib)

    find_package(SYCL REQUIRED)

    set(SOURCES
        api.cpp
        cl_backend_bundle.cpp
        kernel_bundle.cpp
        plan.cpp
        ze_backend_bundle.cpp
    )
    set(PUBLIC_HEADERS
        sycl/make_plan.hpp
    )
    list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/bbfft/")

    add_library(bbfft-sycl ${SOURCES})
    add_library(bbfft::bbfft-sycl ALIAS bbfft-sycl)
    set_common_options(bbfft-sycl)
    add_sycl_to_target(TARGET bbfft-sycl SOURCES ${SOURCES})
    target_compile_definitions(bbfft-sycl PRIVATE -DSYCL2020_CONFORMANT_APIS)
    target_link_libraries(bbfft-sycl PUBLIC bbfft-base)
    target_link_libraries(bbfft-sycl PRIVATE bbfft-opencl bbfft-level-zero)
    target_include_directories(bbfft-sycl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    target_sources(bbfft-sycl PUBLIC FILE_SET HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/include
        FILES ${PUBLIC_HEADERS})

    install_lib(bbfft-sycl bbfft)
endif()
