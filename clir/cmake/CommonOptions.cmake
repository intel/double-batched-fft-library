# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function(add_flag_if_available target flag)
    string(MAKE_C_IDENTIFIER ${flag} flag_c)
    check_cxx_compiler_flag(${flag} HAVE_FLAG${flag_c})
    if(HAVE_FLAG${flag_c})
        target_compile_options(${target} PRIVATE ${flag})
    endif()
endfunction()

function(add_common_flags target)
    if(ENABLE_WARNINGS)
        add_flag_if_available(${target} -Wall)
        add_flag_if_available(${target} -Wextra)
        add_flag_if_available(${target} -Wpedantic)
        add_flag_if_available(${target} -Wundefined-func-template)
    endif()
endfunction()
