# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function(set_common_options target)
    set_target_properties(${target} PROPERTIES
                          VERSION ${double_batched_fft_library_VERSION}
                          SOVERSION ${double_batched_fft_library_VERSION_MAJOR})
    target_compile_features(${target} PUBLIC cxx_std_17)
    if(ENABLE_WARNINGS)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wundefined-func-template)
    endif()
endfunction()
