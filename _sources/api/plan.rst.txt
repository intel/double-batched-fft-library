.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

=====
Plans
=====

Creating and executing plans
============================

SYCL factory functions
----------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, ::sycl::queue)

.. doxygenfunction:: bbfft::make_plan(configuration const&, ::sycl::queue, ::sycl::context, ::sycl::device)

OpenCL factory functions
------------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, cl_command_queue)

.. doxygenfunction:: bbfft::make_plan(configuration const&, cl_command_queue, cl_context, cl_device_id)

Level Zero factory function
---------------------------

.. doxygenfunction:: bbfft::make_plan(configuration const&, ze_command_list_handle_t, ze_context_handle_t, ze_device_handle_t)

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
