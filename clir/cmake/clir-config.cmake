# Copyright (C) 2021 Alex Reinking
# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

@PACKAGE_INIT@

set(clir_known_comps static shared)
set(clir_comp_static NO)
set(clir_comp_shared NO)

foreach (clir_comp IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    if (clir_comp IN_LIST clir_known_comps)
        set(clir_comp_${clir_comp} YES)
    else ()
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
            "Unknown clir component `${clir_comp}`.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        return()
    endif ()
endforeach ()

if (clir_comp_static AND clir_comp_shared)
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "The clir components `static` and `shared` are mutually exclusive.")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endif ()

set(clir_static_targets "${CMAKE_CURRENT_LIST_DIR}/clir-static-targets.cmake")
set(clir_shared_targets "${CMAKE_CURRENT_LIST_DIR}/clir-shared-targets.cmake")

macro(clir_load_targets type)
    if (NOT EXISTS "${clir_${type}_targets}")
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
            "Requested `${type}` libraries for clir not found. (Missing: ${clir_${type}_targets}.)")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        return()
    endif ()
    include("${clir_${type}_targets}")
endmacro()

if (clir_comp_static)
    clir_load_targets(static)
elseif (clir_comp_shared)
    clir_load_targets(shared)
elseif (DEFINED clir_SHARED_LIBS AND clir_SHARED_LIBS)
    clir_load_targets(shared)
elseif (DEFINED clir_SHARED_LIBS AND NOT clir_SHARED_LIBS)
    clir_load_targets(static)
elseif (BUILD_SHARED_LIBS)
    if (EXISTS "${clir_shared_targets}")
        clir_load_targets(shared)
    else ()
        clir_load_targets(static)
    endif ()
else ()
    if (EXISTS "${clir_static_targets}")
        clir_load_targets(static)
    else ()
        clir_load_targets(shared)
    endif ()
endif ()

