.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

=============
Configuration
=============

Main configuration
==================

.. doxygenstruct:: bbfft::configuration
   :members:

Constants
=========

.. doxygenvariable:: bbfft::max_fft_dim

.. doxygenvariable:: bbfft::max_tensor_dim


Enumerations
============

Precision
---------

.. doxygenenum:: bbfft::precision

In templated code the following helper converts the data type to the precision value:

.. doxygenstruct:: bbfft::to_precision
   :members:

Specializations store the precision in the value member:

.. doxygenstruct:: bbfft::to_precision< float >
   :members:

.. doxygenstruct:: bbfft::to_precision< double >
   :members:

The following helper allows to write shorter code:

.. doxygenvariable:: bbfft::to_precision_v

Direction
---------

.. doxygenenum:: bbfft::direction

Transform type
--------------

.. doxygenenum:: bbfft::transform_type


Stride computation
==================

.. doxygenfunction:: bbfft::default_istride

.. doxygenfunction:: bbfft::default_ostride

User callbacks
==============

.. doxygenenum:: bbfft::kernel_language

.. doxygenstruct:: bbfft::user_module
   :members:
