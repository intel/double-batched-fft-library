.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

==================
Offline generation
==================

OpenCL-C for FFT kernels source code can be generated offline using the following function:

.. doxygenfunction:: bbfft::generate_fft_kernels

Device info
===========

FFTs are specialized for the target device based on information stored in the following struct:

.. doxygenstruct:: bbfft::device_info
   :members:

.. doxygenfunction:: bbfft::operator<<(std::ostream&, device_type)

.. doxygenfunction:: bbfft::operator<<(std::ostream&, device_info const&)

Enumerations
------------

.. doxygenenum:: bbfft::device_type

Query device info in OpenCL
---------------------------

.. doxygenfunction:: bbfft::get_device_info(cl_device_id)

.. doxygenfunction:: bbfft::get_device_id(cl_device_id)

Query device info in Level Zero
-------------------------------

.. doxygenfunction:: bbfft::get_device_info(ze_device_handle_t)

.. doxygenfunction:: bbfft::get_device_id(ze_device_handle_t)

Query device info in SYCL
-------------------------

.. doxygenfunction:: bbfft::get_device_info(::sycl::device)

.. doxygenfunction:: bbfft::get_device_id(::sycl::device)

Algorithms
==========

The :cpp:func:`bbfft::generate_fft_kernels` function automatically selects
the algorithm to generate the FFT kernel.
The functions in this section allow direct access to the generators of each algorithm.

Small batch fft
---------------

The "small batch FFT" is intended for FFT sizes up to about N=64
(the maximum size depends on the size of the register file of the device).

.. doxygenfunction:: bbfft::configure_small_batch_fft

.. doxygenfunction:: bbfft::generate_small_batch_fft

.. doxygenstruct:: bbfft::small_batch_configuration
   :members:

Two factor fft
--------------

The "two factor FFT" is intended for larger FFT up to the size of the shared local memory.

.. doxygenfunction:: bbfft::configure_factor2_slm_fft

.. doxygenfunction:: bbfft::generate_factor2_slm_fft

.. doxygenstruct:: bbfft::factor2_slm_configuration
   :members:

