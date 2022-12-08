.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

=========
Utilities
=========

Tensor indexer
==============

.. doxygenclass:: bbfft::tensor_indexer
   :members:

Storage layout
==============

.. doxygenenum:: bbfft::layout

Functions
=========

.. doxygenfunction:: bbfft::fit_array

Online compiler
===============

These functions JIT-compile OpenCL-C.

OpenCL
------

.. doxygenfunction:: build_module(std::string, cl_context, cl_device_id)

.. doxygenfunction:: create_kernel(cl_program, std::string)

Level Zero
----------

.. doxygenfunction:: build_module(std::string, ze_context_handle_t, ze_device_handle_t)

.. doxygenfunction:: create_kernel(ze_module_handle_t, std::string)
