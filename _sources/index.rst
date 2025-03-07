.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

Double-Batched FFT Library
==========================

The Double-Batched FFT Library is a library for computing the Fast Fourier Transform (FFT).
The library targets Graphics Processing Units, supporting
`OpenCL <https://www.khronos.org/opencl/>`_,
`Level Zero <https://spec.oneapi.io/level-zero/latest/>`_,
and `SYCL <https://www.khronos.org/sycl/>`_.

The library supports double-batching.
That is, let :math:`x` be the input tensor of shape :math:`M \times N_1 \times \dots \times N_D \times K`,
with :math:`D=1,2,3`. The D-dimensional FFT should be computed over the N-modes for every M-mode and K-mode.
Think of a two-level nested loop over indices m and k where the FFT is applied on the subtensor
:math:`x(m,:,...,:,k)`.
Double-batching means the capability to batch both indices m and k in a single kernel call.
Note that most FFT library for GPUs support single-batching, e.g. for :math:`M=1`.

.. note::
   For all runtimes, the library requires the
   `cl_intel_required_subgroup_size <https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html>`_ extension.
   For the OpenCL runtime, the `cl_intel_unified_shared_memory <https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html>`_ extension is required.

License
-------

`BSD 3-Clause License <https://www.opensource.org/licenses/BSD-3-Clause>`_

Table of contents
-----------------

.. toctree::
   :includehidden:
   :maxdepth: 2

   manual/index
   api/index



