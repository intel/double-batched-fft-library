#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

types = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float', 'double']
lens = ['', 2, 3, 4, 8, 16]

for t in types:
    for l in lens:
        print(f'X(as_{t}{l}, 1, 1) \\')
