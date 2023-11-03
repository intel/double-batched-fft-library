#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import itertools
import math

primes = [2, 3, 5, 7, 11, 13]

parser = argparse.ArgumentParser()
parser.add_argument('-e', action='store_true')
parser.add_argument('-p', type=int, default=len(primes), choices=range(1,len(primes)+1))
parser.add_argument('min', type=int, default=2)
parser.add_argument('max', type=int, default=512)
args = parser.parse_args()

def ilog(base, x):
    return int(math.ceil(math.log(x, base)))

powers = []
exponents = [list(range(ilog(primes[i], args.max + 1))) for i in range(args.p)]
if args.e:
    exponents[0] = list(range(1, ilog(2, args.max) + 1))
for es in itertools.product(*exponents):
    p = 1
    for i, e in enumerate(es):
        p *= pow(primes[i], e)
    if args.min <= p and p <= args.max:
        powers.append(p)
print(' '.join([str(i) for i in sorted(powers)]))
