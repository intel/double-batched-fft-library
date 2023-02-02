# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(GenerateExportHeader)
include(CommonOptions)
include(InstallLib)

set(SOURCES_PRIVATE_TEST
    scrambler.cpp
)

set(SOURCES
    bad_configuration.cpp
    configuration.cpp
    device_info.cpp
    prime_factorization.cpp
    root_of_unity.cpp
    math.cpp
    mixed_radix_fft.cpp
    user_module.cpp
    generator/factor2_slm_fft.cpp
    generator/small_batch_fft.cpp
    generator/utility.cpp
)
set(PUBLIC_HEADERS
    generator.hpp
    configuration.hpp
    plan.hpp
    bad_configuration.hpp
    device_info.hpp
    tensor_indexer.hpp
    user_module.hpp
    detail/plan_impl.hpp
)
list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/bbfft/")

add_library(bbfft-private-test OBJECT ${SOURCES_PRIVATE_TEST})
target_include_directories(bbfft-private-test PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

add_library(bbfft-base ${SOURCES} $<TARGET_OBJECTS:bbfft-private-test>)
add_library(bbfft::bbfft-base ALIAS bbfft-base)
set_common_options(bbfft-base)
target_link_libraries(bbfft-base PRIVATE clir::clir)
set(bbfft_export_header "${PROJECT_BINARY_DIR}/include/bbfft/export.hpp")
generate_export_header(bbfft-base BASE_NAME BBFFT
    EXPORT_FILE_NAME "${bbfft_export_header}")
target_compile_definitions(bbfft-base PUBLIC
    "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:BBFFT_STATIC_DEFINE>")
target_include_directories(bbfft-base PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
)
target_include_directories(bbfft-base PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(bbfft-base PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
              ${PROJECT_BINARY_DIR}/include
    FILES ${PUBLIC_HEADERS}
          "${bbfft_export_header}")
install_lib(bbfft-base bbfft)
