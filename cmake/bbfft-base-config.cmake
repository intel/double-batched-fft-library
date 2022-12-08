# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

find_dependency(clir REQUIRED)

@SHARED_STATIC_TEMPLATE@
