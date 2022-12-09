.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

==================
Offline generation
==================

The basis for all FFT algorithms are two 1D FFT algorithms.
The "small batch FFT" is intended for FFT sizes up to about N=64

(the maximum size depends on the size of the register file of the device).
The "two factor FFT" is intended for larger FFT up to the size of the shared local memory.

The OpenCL C source code of the two algorithms is generated using the following API calls. 

Small batch fft
===============

.. doxygenfunction:: bbfft::configure_small_batch_fft

.. doxygenfunction:: bbfft::generate_small_batch_fft

.. doxygenstruct:: bbfft::small_batch_configuration
   :members:

Two factor fft
==============

.. doxygenfunction:: bbfft::configure_factor2_slm_fft

.. doxygenfunction:: bbfft::generate_factor2_slm_fft

.. doxygenstruct:: bbfft::factor2_slm_configuration
   :members:

Device info
===========

.. doxygenstruct:: bbfft::device_info
   :members:

