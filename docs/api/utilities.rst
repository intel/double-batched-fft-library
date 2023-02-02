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

.. doxygenfunction:: bbfft::build_kernel_bundle(std::string, cl_context, cl_device_id)

.. doxygenfunction:: bbfft::build_kernel_bundle(uint8_t const*, std::size_t, cl_context, cl_device_id)

.. doxygenfunction:: bbfft::create_kernel(cl_program, std::string)

.. doxygenfunction:: bbfft::get_native_binary(cl_program, cl_device_id)

Level Zero
----------

.. doxygenfunction:: bbfft::build_kernel_bundle(std::string, ze_context_handle_t, ze_device_handle_t)

.. doxygenfunction:: bbfft::build_kernel_bundle(uint8_t const*, std::size_t, ze_context_handle_t, ze_device_handle_t)

.. doxygenfunction:: bbfft::create_kernel(ze_module_handle_t, std::string)

.. doxygenfunction:: bbfft::get_native_binary(ze_module_handle_t)
