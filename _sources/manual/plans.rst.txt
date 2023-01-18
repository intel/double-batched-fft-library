.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

========================
Creating and using plans
========================

Complex-valued tensors are the default input and output of a Fast Fourier Transform,
that is, the :math:`N_1\times\dots\times N_D` complex-valued tensor :math:`x` is mapped to the same-sized complex
output tensor :math:`X` by the following formula:

.. math::

    X(k_1,\dots,k_D) = \sum_{j_1=0}^{N_1-1}\ldots\sum_{j_D=0}^{N_D-1}x(j_1,\dots,j_D)\cdot
        \exp\left(\sigma\cdot 2\pi i \sum_{l=1}^{D}k_lj_l/N_l\right)

The sign of the transform :math:`\sigma \in \{-1,1\}` is also called the direction.
We adopt the convention that the forward transform as negative sign and that the backward transform
has positive sign.

In practice, the input tensor :math:`x` is often real-valued, and in that case the following symmetry holds
for the output tensor :math:`X`:

.. math::

    X(k_1,\dots,k_D) = X^*(N_1-k_1,\dots,k_D),

where the star denotes the complex conjugate.

.. note:: 

    Index :math:`k_1` is taken modulo :math:`N_1`, that is, 
    :math:`X^*(N_1,\dots,k_D) = X^*(0,\dots,k_D)`.
    Thus, :math:`X(0,\dots,k_D)=X^*(0,\dots,k_D)` must be real.
    Moreover, similar symmetries holds in modes :math:`2,\dots,D`, too,
    but those are not necessary for the discussion.

Due to the symmetry, only the first :math:`\lfloor N_1/2\rfloor+1` entries need to be stored in the output
tensor. So for the real-valued :math:`N_1\times\dots\times N_D` input tensor one typically
stores the :math:`\lfloor N_1/2\rfloor+1\times\dots\times N_D` complex-valued output tensor.

The above considerations lead to specialized FFT routines for complex-valued FFTs, FFTs with
real-valued input data, or FFTs with conjugate even symmetry.
These are called complex-to-complex (c2c), real-to-complex (r2c), and complex-to-real (c2r).

Bbfft uses just-in-time (JIT) compilation to obtain fast FFTs for every problem size, transform type,
tensor strides, and transform direction.
JIT compilation is rather expensive, thus every configuration is captured in a plan object that is
reused for multiple FFT invocations of the same type.

.. attention::

   Code samples in the following assume you have ``using namespace bbfft;`` in your code.


One dimension
-------------

.. cpp:namespace:: bbfft

Let :math:`x` be the input tensor of shape :math:`M \times N \times K`,
where the FFT is taken over the second mode (N-mode).
Plan generation is unified in the Double-Batched FFT Library via the :cpp:type:`configuration` struct.
Initializer-list syntax may be used to conveniently create configurations: 

.. code:: c++

   configuration cfg = {1,              // One dimensional
                        {M, N, K},      // Tensor shape
                        precision,      // Single (f32) or double (f64)
                        direction,      // Forward (-1) or backward (+1)
                        transform_type, // c2c, r2c, c2r
                        input_strides,  // Strides of input tensor
                        output_strides  // Strides of output tensor
                       };

Input and output strides are explained in more detail in :ref:`data-layout`.
One may omit input_strides and output_strides.
Then, the default layout for an in-place transform is set.
In order to choose the default layout for an out-of-place transform, use

.. code:: c++

   bool inplace = false;
   cfg.set_strides_default(inplace);

SYCL
~~~~

Using SYCL, the plan is created using the :cpp:func:`make_plan` factory function as following:

.. code:: c++

   #include "bbfft/sycl/make_plan.hpp"

   auto Q = sycl::queue{sycl::gpu_selector_v};
   auto plan = make_plan(cfg, Q);

Plans are executed using

.. code:: c++

   auto event = plan.execute(input, output);
   event.wait();

in out-of-place mode or using

.. code:: c++

   auto event = plan.execute(inout);
   event.wait();

in in-place mode. Input, output, or inout are pointers to either device memory, shared memory, or host
memory, using one of the ``sycl::malloc_<device,shared,host>`` functions.
Note that in-place mode is also used if input = output.

The data type of input and output depends on the transform type:

.. code:: c++

   // c2c
   std::complex<real_t>* input, output;
   // r2c
   real_t* input;
   std::complex<real_t>* output;
   // c2r
   std::complex<real_t>* input;
   real_t* output;

where real_t=float for f32 precision and real_t=double for f64 precision.
In SYCL-mode the library uses either the OpenCL or Level Zero back-end.
The back-end can be selected at run-time by setting the SYCL_DEVICE_FILTER appropriately.

OpenCL
~~~~~~

For OpenCL, use

.. code:: c++

   #include "bbfft/cl/make_plan.hpp"

   cl_command_queue Q = clCreateCommandQueueWithProperties(...);
   auto plan = make_plan(cfg, Q);

Execute functions return a ``cl_event`` that shall be used for synchronization.
Input and output buffers must be created with the ``cl<Device,Shared,Host>MemAllocINTEL`` functions
from the
`cl_intel_unified_shared_memory <https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html>`_ extension.

Level Zero
~~~~~~~~~~

.. code:: c++

   #include "bbfft/ze/make_plan.hpp"

   ze_context_handle_t;
   ze_device_handle_t device;
   ze_command_list_handle_t cmd_list;
   zeContextCreate(..., &context);
   zeDeviceGet(.., &device);
   zeCommandListCreateImmediate(..., &cmd_list);
   auto plan = make_plan(cfg, cmd_list, context, device);

Execute functions return a ``ze_event_handle_t`` that shall be used for synchronization.
Input and output buffers must be created with the ``zeMemAlloc<Device,Shared,Host>`` functions.

Two or three dimensions
-----------------------

.. cpp:namespace:: bbfft

Let :math:`x` be the input tensor of shape :math:`M \times N_1 \times \dots \times N_D \times K`,
where the FFT is taken over the N-modes.
Everything works the same way as in the one-dimensional case, except that dimension, shape, and strides
need to be adjusted.
For example,

.. code:: c++

   cfg.dim = 3;
   cfg.shape = {M, N1, N2, N3, K};

.. warning::

   Two- and three-dimensional FFTs are currently under development.
   Do not expect that special data layouts beside the default data layout give the correct results.
