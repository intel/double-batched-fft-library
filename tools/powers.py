#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('-e', action='store_true')
parser.add_argument('min', type=int, default=2)
parser.add_argument('max', type=int, default=512)
args = parser.parse_args()

def ilog(base, x):
    return int(math.ceil(math.log(x, base)))

powers = []
for i in range(1 if args.e else 0, ilog(2, args.max) + 1):
    for j in range(ilog(3, args.max) + 1):
        for k in range(ilog(5, args.max) + 1):
            for m in range(ilog(7, args.max) + 1):
                for n in range(ilog(11, args.max) + 1):
                    for o in range(ilog(13, args.max) + 1):
                        p = 2**i * 3**j * 5**k * 7**m * 11**n * 13**o
                        if args.min <= p and p <= args.max:
                            powers.append(p)
print(' '.join([str(i) for i in sorted(powers)]))
