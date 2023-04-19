# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/AddAotKernelsToTarget.cmake")

set(targets "${CMAKE_CURRENT_LIST_DIR}/${CMAKE_FIND_PACKAGE_NAME}-targets.cmake")

if (NOT EXISTS "${targets}")
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "Requested libraries not found. (Missing: ${targets}.)")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endif ()
include("${targets}")

unset(targets)
