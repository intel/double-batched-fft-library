#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import re
from collections import OrderedDict

table_begin = re.compile(r'<table[^>]*>', re.ASCII)
table_end = re.compile(r'</table>', re.ASCII)
caption = re.compile(r'<caption[^>]*>\s*Table\s*([0-9]+)', re.ASCII)
signature = re.compile(
    r'<strong>((?:\w|<em>\w</em>|&lt;op&gt;)+)</strong>\s*\(([^)]*)\)',
    re.ASCII)
clean_tags = re.compile(r'<[^>]*>')
has_op = re.compile(r'&lt;op&gt;')

extract_tables = {
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 42
}
skip_2nd_variant = {
    "get_global_size", "get_global_id", "get_local_size",
    "get_enqueued_local_size", "get_local_id", "get_num_groups",
    "get_group_id", "get_global_offset", "fmax", "fmin", "isordered",
    "isunordered", "nextafter"
}
builtin = OrderedDict()

tables = dict()
active_key = None
current_table = ''

with open('OpenCL_C.html') as spec:
    for line in spec:
        b = re.search(table_begin, line)
        if b:
            active_key = 'unknown'
            current_table = ''
        if active_key:
            c = re.search(caption, line)
            if c:
                active_key = int(c.group(1))
            current_table += line
        e = re.search(table_end, line)
        if e:
            if active_key in extract_tables:
                tables[active_key] = current_table
            active_key = None

for key, tab in tables.items():
    matches = re.finditer(signature, tab)
    for fun in matches:

        def add_fun(name):
            args = fun.group(2)
            num_args = args.count(',') + 1 if args != '' else 0
            if name not in builtin:
                builtin[name] = (num_args, num_args)
            elif name not in skip_2nd_variant:
                t = builtin[name]
                builtin[name] = (min(t[0], num_args), max(t[0], num_args))

        name = re.sub(clean_tags, '', fun.group(1))
        if re.search(has_op, name):
            for op in ('add', 'min', 'max'):
                add_fun(re.sub(has_op, op, name))
        else:
            add_fun(name)

for key, val in builtin.items():
    print(f'X({key}, {val[0]}, {val[1]}) \\')
