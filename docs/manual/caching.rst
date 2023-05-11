.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

==============
Kernel caching
==============

The FFT kernel are specialized for the :cpp:class:`configuration <bbfft::configuration>`.
Many factors influence the FFT code, including

- Input and and output tensor shape
- Transform mode (complex-to-complex, complex-to-real, real-to-complex, forward, backward)
- Input and output tensor strides
- Target device architecture (sub-group size, size of register file, size of shared local memory)
- User callbacks

Just-in-time (JIT) compilation is employed to specialize kernels at run-time.
JIT compilation happens at plan creation, which is therefore expensive in comparison to plan execution.
In some situations it might be necessary to create the plan multiple times, hence the double-batched FFT library
offers a caching mechanism.

Run-time caching
================

Run-time caching is enabled by passing a pointer to a :cpp:class:`bbfft::jit_cache` to the plan factory
function:

.. code:: c++

   #include "bbfft/jit_cache_all.hpp"
   #include "bbfft/sycl/make_plan.hpp"

   auto cache = jit_cache_all{};
   auto plan = make_plan(cfg, Q, &cache);
   auto plan2 = make_plan(cfg, Q, &cache);

In the above example, the creation of ``plan`` is expensive, whereas the FFT kernel in the creation
of ``plan2`` is looked up in the cache and therefore the creation is fast.

The object of the class :cpp:class:`bbfft::jit_cache_all` simply caches all kernels it encountered.
More advanced caching strategies can be implemented by deriving from the :cpp:class:`bbfft::jit_cache` interface.


Ahead-of-time caching
=====================

In some situations one might only need a small subset of FFT kernels, which are known at compile-time,
but there is little reuse of plans.
In that case creating a JIT cache ahead-of-time (AOT) can be useful.

We need a two-step process in order to enable AOT caching:
First, we compile FFT kernels to a native device binary using the ``bbfft-aot-generate`` tool
and use the GNU linker embed the native device binary in the application.
The second step is to create a :cpp:class:`bbfft::aot_cache` in the application to lookup native device code
during plan creation.

Native device binary generation and linking
-------------------------------------------

We generate the native device binary for selected FFT configurations using
the ``bbfft-aot-generate`` tool that comes with the Double-Batched FFT Library.
For example, in order to create kernels for a complex-to-complex FFT in single precision with
N=16,32 and a batch size of 1000 on Ponte Vecchio use

.. code:: bash

    bbfft-aot-generate -d pvc kernels.bin scfi16*1000 scfi32*1000

See :ref:`descriptor` for the specification of the short FFT descriptor format.
Built-in device info is currently only available for PVC.
For other devices you can pass the output of the ``bbfft-device-info`` tool
to ``bbfft-aot-generate`` with the ``-i`` flag.

The GNU linker is used to embed the native device binary in your application:

.. code:: bash

    ld -r -b binary -o kernels.o kernels.bin

Linking ``kernels.o`` makes the symbols ``_binary_kernels_bin_start`` and ``_binary_kernels_bin_end`` available
that point to the native device binary.
 
CMake users can use the following workflow to automatise the above steps, e.g. for all real-to-complex
power of two FFTs until N=1024:

.. code:: cmake

    find_package(bbfft-aot-generate REQUIRED)

    set(N 2 4 8 16 32 64 128 256 512 1024)
    foreach(n IN LISTS N)
        list(APPEND descriptors "srfi${n}*16384")
    endforeach()

    add_aot_kernels_to_target(TARGET <your-cmake-target> PREFIX kernels DEVICE pvc LIST ${descriptors})

Lookup native device code during plan creation
----------------------------------------------------

If everything worked, we find the object file ``kernels.o`` in the build folder.
The file looks something like the following:

.. code:: bash

    $ nm -a build/examples/aot/kernels.o
    0000000000142770 D _binary_kernels_bin_end
    0000000000142770 A _binary_kernels_bin_size
    0000000000000000 D _binary_kernels_bin_start
    0000000000000000 d .data

Here, we have the symbols ``_binary_kernels_bin_start`` and ``_binary_kernels_bin_end`` that indicate the
start and end of the ahead-of-time compiled binary blob.
With these symbols we create an :cpp:class:`bbfft::aot_module` that we register with the
:cpp:class:`bbfft::aot_cache`:

.. code:: c++

    auto q = sycl::queue{};
    auto cache = bbfft::aot_cache{};
    try {
        extern std::uint8_t _binary_kernels_bin_start, _binary_kernels_bin_end;
        cache.register_module(bbfft::sycl::create_aot_module(
            &_binary_kernels_bin_start, &_binary_kernels_bin_end - &_binary_kernels_bin_start,
            bbfft::module_format::native, q.get_context(), q.get_device()));
    } catch (std::exception const &e) {
        // handle exception
    }

We employ the symbols of the object file to direct ``create_aot_module`` to the native device binary.
We have wrapped ``create_aot_module`` in a ``try ... catch`` block as it might fail, for example if
the application is run on a device requiring a different native device binary.
In that case, the ``aot_module`` is not registered with the ``aot_cache`` and we fall back to
just-in-time compilation.

During plan creation you need to pass the cache to ``make_plan`` such that plan creation benefits from
ahead-of-time compilation:

.. code:: c++

    auto plan = bbfft::make_plan(cfg, q, &cache);
