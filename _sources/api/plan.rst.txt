.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

=====
Plans
=====

A plan encapsulate the online compilation of an FFT and the setup of internal buffers.
The cost of creating a plan is high, therefore plans should be reused.
If you need to create plans multiple times consider :ref:`caching <jit-cache-api>`.

Creating and executing plans
============================

SYCL factory functions
----------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, ::sycl::queue, jit_cache*)

.. doxygenfunction:: bbfft::make_plan(configuration const&, ::sycl::queue, ::sycl::context, ::sycl::device, jit_cache*)

OpenCL factory functions
------------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, cl_command_queue, jit_cache*)

.. doxygenfunction:: bbfft::make_plan(configuration const&, cl_command_queue, cl_context, cl_device_id, jit_cache*)

Level Zero factory function
---------------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, ze_command_list_handle_t, ze_context_handle_t, ze_device_handle_t, jit_cache*)

Plan class
----------

.. doxygenclass:: bbfft::plan
   :members:

Configuration errors
====================

Bad configuration
-----------------

.. doxygenclass:: bbfft::bad_configuration
   :members:

Level Zero
----------

.. doxygendefine:: ZE_CHECK

.. doxygenclass:: bbfft::ze::error
   :members:

OpenCL
------

.. doxygendefine:: CL_CHECK

.. doxygenclass:: bbfft::cl::error
   :members:
