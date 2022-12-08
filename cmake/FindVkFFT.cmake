# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Try to find VkFFT

# The following definitions are added on success
# 
#  VkFFT_FOUND - VkFFT was found
#  VkFFT_INCLUDE_DIR - VkFFT include directory
#
# and the following imported target:
#
#  VkFFT::VkFFT - VkFFT library
#
# The followings hints may be passed in the environment:
#
# VkFFT_ROOT
# VKFFT_ROOT
#

if(VkFFT_INCLUDE_DIR)
    set(VkFFT_FOUND TRUE)
else()
    find_path(VkFFT_INCLUDE_DIR NAMES vkFFT.h
        HINTS
            ENV VkFFT_ROOT
            ENV VKFFT_ROOT
            ENV CPATH
        PATH_SUFFIXES
            include
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(VkFFT DEFAULT_MSG VkFFT_INCLUDE_DIR)

    mark_as_advanced(VkFFT_INCLUDE_DIR)
endif()

if(VkFFT_FOUND AND NOT TARGET VkFFT::VkFFT)
    add_library(VkFFT::VkFFT INTERFACE IMPORTED)
    set_target_properties(VkFFT::VkFFT PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${VkFFT_INCLUDE_DIR}")
endif()
