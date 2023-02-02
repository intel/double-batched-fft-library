# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_OPENCL)
    include(CommonOptions)
    include(InstallLib)

    set(SOURCES
        api.cpp
        device.cpp
        error.cpp
        online_compiler.cpp
        plan.cpp
    )
    set(PUBLIC_HEADERS
        cl/device.hpp
        cl/error.hpp
        cl/make_plan.hpp
        cl/online_compiler.hpp
    )
    list(TRANSFORM PUBLIC_HEADERS PREPEND "${CMAKE_SOURCE_DIR}/include/bbfft/")

    find_package(OpenCL REQUIRED)

    add_library(bbfft-opencl ${SOURCES})
    add_library(bbfft::bbfft-opencl ALIAS bbfft-opencl)
    set_common_options(bbfft-opencl)
    target_link_libraries(bbfft-opencl PUBLIC bbfft-base ${OpenCL_LIBRARY})
    target_compile_definitions(bbfft-opencl PUBLIC ${OpenCL_COMPILE_DEFINITIONS})
    target_include_directories(bbfft-opencl PUBLIC ${OpenCL_INCLUDE_DIR})
    target_include_directories(bbfft-opencl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    target_sources(bbfft-opencl PUBLIC FILE_SET HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/include
        FILES ${PUBLIC_HEADERS})

    install_lib(bbfft-opencl bbfft)
endif()
