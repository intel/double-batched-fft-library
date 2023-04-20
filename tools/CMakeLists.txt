# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

if(BUILD_LEVEL_ZERO)
    add_subdirectory(aot)
endif()
add_subdirectory(common)
if(BUILD_SYCL)
    add_subdirectory(device_info)
endif()
add_subdirectory(offline)
