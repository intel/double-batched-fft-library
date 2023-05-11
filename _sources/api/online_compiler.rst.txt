.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

===============
Online compiler
===============

The double-batched FFT library uses online ("just-in-time") compilation due to the vast number of
FFT configurations.
The online compilation API is described here.

OpenCL
======

.. doxygenfunction:: bbfft::cl::build_kernel_bundle(std::string const&, cl_context, cl_device_id)

.. doxygenfunction:: bbfft::cl::build_kernel_bundle(uint8_t const*, std::size_t, module_format, cl_context, cl_device_id)

.. doxygenfunction:: bbfft::cl::create_kernel

.. doxygenfunction:: bbfft::cl::create_aot_module

Level Zero
==========

.. doxygenfunction:: bbfft::ze::build_kernel_bundle(std::string const&, ze_context_handle_t, ze_device_handle_t)

.. doxygenfunction:: bbfft::ze::build_kernel_bundle(uint8_t const*, std::size_t, module_format, ze_context_handle_t, ze_device_handle_t)

.. doxygenfunction:: bbfft::ze::create_kernel

.. doxygenfunction:: bbfft::ze::compile_to_spirv

.. doxygenfunction:: bbfft::ze::compile_to_native

.. doxygenfunction:: bbfft::ze::create_aot_module

SYCL
====

.. doxygenfunction:: bbfft::sycl::build_native_module(std::string const&, ::sycl::context, ::sycl::device)

.. doxygenfunction:: bbfft::sycl::build_native_module(uint8_t const*, std::size_t, module_format, ::sycl::context, ::sycl::device)

.. doxygenfunction:: bbfft::sycl::make_shared_handle

.. doxygenfunction:: bbfft::sycl::make_kernel_bundle

.. doxygenfunction:: bbfft::sycl::create_kernel

.. doxygenfunction:: bbfft::sycl::create_aot_module

Enumerations
============

.. doxygenenum:: bbfft::module_format

