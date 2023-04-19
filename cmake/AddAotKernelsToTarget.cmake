# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function(add_aot_kernels_to_target)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;PREFIX;DEVICE" "LIST")
    set(BIN_FILE "${ARG_PREFIX}.bin")
    set(OBJ_FILE "${ARG_PREFIX}.o")
    add_custom_command(
        OUTPUT ${OBJ_FILE}
        BYPRODUCTS ${BIN_FILE}
        COMMAND bbfft::bbfft-aot-generate -d ${ARG_DEVICE} ${BIN_FILE} ${ARG_LIST}
        COMMAND ${CMAKE_LINKER} -r -b binary -o ${OBJ_FILE} ${BIN_FILE}
        DEPENDS bbfft::bbfft-aot-generate
        COMMENT "FFT ahead-of-time compilation: ${ARG_PREFIX}"
    )
    add_library(fft-aot-${PREFIX} OBJECT IMPORTED)
    set_property(TARGET fft-aot-${PREFIX}
                 PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_FILE}")

    target_link_libraries(${ARG_TARGET} PRIVATE fft-aot-${PREFIX})
endfunction()

