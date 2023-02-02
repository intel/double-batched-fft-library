# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_LEVEL_ZERO)
    include(CommonOptions)
    include(InstallLib)

    find_package(LevelZero REQUIRED)
    find_package(ocloc REQUIRED)

    set(SOURCES
        api.cpp
        device.cpp
        error.cpp
        event_pool.cpp
        kernel.cpp
        kernel_bundle.cpp
        online_compiler.cpp
        plan.cpp
    )
    set(PUBLIC_HEADERS
        ze/device.hpp
        ze/error.hpp
        ze/make_plan.hpp
        ze/online_compiler.hpp
    )
    list(TRANSFORM PUBLIC_HEADERS PREPEND "${CMAKE_SOURCE_DIR}/include/bbfft/")

    add_library(bbfft-level-zero ${SOURCES})
    add_library(bbfft::bbfft-level-zero ALIAS bbfft-level-zero)
    set_common_options(bbfft-level-zero)
    target_link_libraries(bbfft-level-zero PUBLIC bbfft-base ${LevelZero_LIBRARY})
    target_link_libraries(bbfft-level-zero PRIVATE ${ocloc_LIBRARY})
    target_include_directories(bbfft-level-zero PUBLIC ${LevelZero_INCLUDE_DIR})
    target_include_directories(bbfft-level-zero PRIVATE ${ocloc_INCLUDE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    target_sources(bbfft-level-zero PUBLIC FILE_SET HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/include
        FILES ${PUBLIC_HEADERS})

    install_lib(bbfft-level-zero bbfft)
endif()
