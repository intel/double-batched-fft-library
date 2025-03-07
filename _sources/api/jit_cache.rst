.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

.. _jit-cache-api:

==============
Kernel caching
==============

It might be necessary to recreate the same plan multiple times.
To facilitate fast plan creation, kernels might be cached and looked up at plan creation.

JIT cache interface
===================

The general interface of a JIT cache is defined below.
Users may derive from jit_cache in order to implement their own caching strategies should
the provided caching strategies be insufficent.

.. doxygenclass:: bbfft::jit_cache
   :members:

Cache keys
----------

.. doxygenstruct:: bbfft::jit_cache_key
   :members:

.. doxygenstruct:: bbfft::jit_cache_key_hash
   :members:

JIT cache all
=============

Simple cache that stores all encountered kernels.

.. doxygenclass:: bbfft::jit_cache_all
   :members:

Ahead-of-time cache
===================

Kernels may be compiled ahead-of-time.
The ahead-of-time "cache" is used to look-up kernels at plan creation time.

.. doxygenclass:: bbfft::aot_cache
   :members:

.. doxygenstruct:: bbfft::aot_module
   :members:

