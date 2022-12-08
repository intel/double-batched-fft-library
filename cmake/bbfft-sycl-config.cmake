# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

find_dependency(bbfft-base REQUIRED)
find_dependency(bbfft-level-zero REQUIRED)
find_dependency(bbfft-opencl REQUIRED)

@SHARED_STATIC_TEMPLATE@
