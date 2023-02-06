# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(SYCL REQUIRED)
find_program(OCLOC ocloc REQUIRED)

add_executable(aot-generate generate.cpp configurations.cpp)
target_compile_features(aot-generate PRIVATE cxx_std_17)
target_link_libraries(aot-generate PRIVATE bbfft-base)

set(SRC_FILE "kernels.cl")
set(NAMES_FILE "aot_compiled_kernels.cpp")
set(BIN_FILE "kernels_XE_HPC_COREpvc.bin")
set(OBJ_FILE "kernels.o") 
add_custom_command(
    OUTPUT ${OBJ_FILE}
    BYPRODUCTS ${SRC_FILE} ${NAMES_FILE} ${BIN_FILE}
    COMMAND aot-generate ${SRC_FILE} ${NAMES_FILE}
    COMMAND ${OCLOC} compile -file ${SRC_FILE} -device pvc -internal_options "-cl-ext=+cl_khr_fp64"
    COMMAND ${CMAKE_LINKER} -r -b binary -o ${OBJ_FILE} ${BIN_FILE}
    DEPENDS aot-generate
    COMMENT "Ahead-of-time compilation of FFT kernels"
)
add_library(kernels OBJECT IMPORTED)
set_property(TARGET kernels PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_FILE}")

add_executable(aot main.cpp configurations.cpp aot_cache.cpp "${CMAKE_CURRENT_BINARY_DIR}/${NAMES_FILE}")
add_sycl_to_target(TARGET aot SOURCES main.cpp aot_cache.cpp)
target_compile_features(aot PRIVATE cxx_std_17)
target_link_libraries(aot PRIVATE bbfft-sycl kernels)
