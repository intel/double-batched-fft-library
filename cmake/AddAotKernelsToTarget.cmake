# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function(add_aot_kernels_to_target)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "NO_LINK_SCOPE" "TARGET;PREFIX;DEVICE;LINK_SCOPE" "LIST")
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

    set(SUPPORTED_LINK_SCOPES PRIVATE INTERFACE PUBLIC)

    if(ARG_NO_LINK_SCOPE)
        target_link_libraries(${ARG_TARGET} fft-aot-${PREFIX})
    else()
        if(NOT DEFINED ARG_LINK_SCOPE)
            set(ARG_LINK_SCOPE PRIVATE)
        endif()
        if(NOT ${ARG_LINK_SCOPE} IN_LIST SUPPORTED_LINK_SCOPES)
            message(SEND_ERROR "Link scope must be PRIVATE, INTERFACE, or PUBLIC. (${ARG_LINK_SCOPE} given)")
        endif()
        target_link_libraries(${ARG_TARGET} ${ARG_LINK_SCOPE} fft-aot-${PREFIX})
    endif()
endfunction()

