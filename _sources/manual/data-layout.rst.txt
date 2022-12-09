.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

.. _data-layout:

==========================
Understanding data layouts
==========================

The most important message comes first: Every data layout in the Double-Batched FFT Library
is defined in **column major** order.

Strides
=======

Suppose we are given a :math:`M \times N_1 \times \dots \times N_D \times K` tensor.
The strides for the packed tensor layout (in column major order) is

.. math::

   \text{packed}(M, N_1, \dots, N_D, K) = (1, M, M\cdot N_1, \ldots, M\cdot N_1\cdot\ldots\cdot N_{D})

Let :math:`s = \text{packed}(M, N_1, \dots, N_D, K)`.
Offsets of the entry :math:`(m,n_1,\dots,n_D,k)` are computed with

.. math::

   \text{linear_index} = m s_0 + \sum_{j=1}^D n_j s_j + k s_{D+1}

such that ``x[linear_index]`` gives the correct entry, where ``x`` is the base address of the tensor's data.
Here, ``x`` might be either real or complex, that is, the ``linear_index`` is taken 
w.r.t. to the underlying data type.

.. cpp:namespace:: bbfft

Default strides may be overriden in the :cpp:type:`configuration`.

.. warning::

   :math:`s_0 \neq 1` is unsupported.

Default c2c
===========

Default c2c strides are 

.. math::

   \text{input_strides} = \text{packed}(M, N_1, \dots, N_D, K)

   \text{output_strides} = \text{packed}(M, N_1, \dots, N_D, K)

There is no distinction between in-place and out-of-place transforms.

Default r2c
===========

For r2c, the input tensor is real and the output tensor is complex.
We only need to store half of the modes due to symmetry, therefore define

.. math:: 

   N_1' = \lfloor N_1 / 2 \rfloor + 1

We need to distinguish between the default out-of-place layout and the default in-place layout.

Out-of-place
------------

For the out-of-place transform the input tensor uses the default packed format

.. math::

   \text{input_strides} = \text{packed}(M, N_1, \dots, N_D, K)

For the output tensor we use the default packed format but truncate the first FFT mode:

.. math::

   \text{output_strides} = \text{packed}(M, N_1', \dots, N_D, K)

In-place
--------

The input tensor is overwritten during the FFT, hence it needs enough space to store the output tensor.
Therefore, the first FFT mode needs to be padded. Let

.. math::

   N_1'' = 2N_1'

The strides are 

.. math::

   \text{input_strides} = \text{packed}(M, N_1'', \dots, N_D, K)

   \text{output_strides} = \text{packed}(M, N_1', \dots, N_D, K)

*Example:* Let :math:`N_1=8`. Then :math:`N_1'=5` and we store 5 complex values in the first FFT mode.
As 1 complex value requires the space of 2 real values we pad the input tensor with 2 extra reals
and have :math:`N_1''=10`.

Default c2r
===========

c2r is the converse of r2c, so we simply swap input and output strides.

Tensor indexer
==============

The :cpp:type:`tensor_indexer` is a helpful class to work with input tensors.
E.g. in one dimension for a r2c in-place transform we can use the following code:

.. code:: c++

   std::size_t N_out = N / 2 + 1;
   auto xi = tensor_indexer<std::size_t, 3, layout::col_major>({M, N, K}, {1, M, M * N_out});
   auto x = malloc_device<T>(xi.size(), Q);
   for (std::size_t k = 0; k < K; ++k) {
      for (std::size_t n = 0; n < N; ++n) {
         for (std::size_t m = 0; m < M; ++m) {
            x[xi(m, n, k)] = ...; // Load data for entry (m, n, k)
         }
      }
   }

.. tip::

   The :cpp:type:`tensor_indexer` and :cpp:type:`configuration` strides are compatible.
   For example, given the configuration ``cfg``, one can initialize ``xi`` with

   .. code:: c++

      auto xi = tensor_indexer<std::size_t, 3, layout::col_major>(
                    fit_array<3>(cfg.shape), fit_array<3>(cfg.istride));
