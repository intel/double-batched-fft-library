.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

========
Examples
========

Here we show the configurations for several standard cases.
FFTW analogues are provided if applicable.

Single complex-data FFT
-----------------------

.. code:: c++

   configuration cfg = {1, {1, N, 1}, direction::forward, transform_type::c2c};

.. tip::

   Analogue

   .. code:: c++

      int n0 = N;
      int sign = FFTW_FORWARD; // -1
      auto plan = fftw_plan_dft_1d(n0, in_ptr, out_ptr, sign, flags);

Single real-data FFT
--------------------

.. code:: c++

   configuration cfg = {1, {1, N, 1}, direction::forward, transform_type::r2c};

.. tip::

   Analogue

   .. code:: c++

      int n0 = N;
      auto plan = fftw_plan_dft_r2c_1d(n0, in_ptr, out_ptr, flags);

Single-batched real-data FFT (in-place)
---------------------------------------

.. code:: c++

   configuration cfg = {1, {1, N, howmany}, direction::forward, transform_type::r2c};
   cfg.set_strides_default(true);

.. tip::

   Analogue

   .. code:: c++

      int rank = 1;
      int n = N;
      int nc = N/2 + 1;
      int inembed = 2 * nc;
      int idist = inembed;
      int onembed = nc;
      int odist = onembed;
      auto plan = fftw_plan_many_dft_r2c(rank, &n, howmany, in_ptr, &inembed, 1, idist,
                                         out_ptr, &onembed, 1, odist, flags);

Double-batched real-data FFT (in-place)
---------------------------------------

.. code:: c++

   configuration cfg = {1, {M, N, howmany}, direction::forward, transform_type::r2c};
   cfg.set_strides_default(true);

.. tip::

   Analogue

   .. code:: c++

      int rank = 1;
      int n = N;
      int nc = N/2 + 1;
      int inembed = 2 * nc;
      int idist = inembed * M;
      int onembed = nc;
      int odist = onembed * M;
      auto plan = fftw_plan_many_dft_r2c(rank, &n, howmany, in_ptr, &inembed, 1, idist,
                                         out_ptr, &onembed, 1, odist, flags);
      // Execution needs loop over M-mode
      for (int m = 0; m < M; ++m) {
         fftw_execute_dft_r2c(plan, in_ptr + m, out_ptr + m);
      }

Single 3D complex-data FFT
--------------------------

.. code:: c++

   configuration cfg = {1, {1, N0, N1, N2, 1}, direction::forward, transform_type::c2c};

.. tip::

   Analogue

   .. code:: c++

      // FFTW expects row-major, therefore reverse the order
      int n0 = N2;
      int n1 = N1;
      int n2 = N0;
      int sign = FFTW_FORWARD; // -1
      auto plan = fftw_plan_dft_3d(n0, n1, n2, in_ptr, out_ptr, flags);
