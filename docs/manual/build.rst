.. Copyright (C) 2022 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

====================
Building and linking
====================

Dependencies
============

Installing the
`Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html>`_
is sufficent to compile the library.

Detailed list of dependencies:

- CMake >= 3.23
- C++ compiler with SYCL support
- OpenCL library
- Level Zero loader library
- ocloc (OpenCL offline compiler from the Intel Compute Runtime)

Build from source using oneAPI
==============================

First, Initialize the oneAPI environment.

.. code:: console

    . /opt/intel/oneapi/setvars.sh

Clone the library's repository to your local filesystem. Enter the directory containing your local copy
of the repository and run

.. code:: console

    cmake -Bbuild -S. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=NO

Static libraries are built when using the above command.
Forshared libraries build use 

.. code:: console

    cmake -Bbuild -S. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=YES

After successful configuration, build the library with

.. code:: console

    cmake --build build

Finally, install with

.. code:: console

    cmake --install build --prefix /path/to/installation

Build options
=============

The build can be customized by passing -D<option>=ON or OFF to cmake.
The following options are supported:

====================== ============
Option                 Description
====================== ============
BUILD_DOCUMENTATION    Generate the documentation
BUILD_BENCHMARK        Build benchmark executables
BUILD_EXAMPLE          Build examples
BUILD_TESTING          Build unit tests
BUILD_SYCL             Build FFT library for SYCL
BUILD_LEVEL_ZERO       Build FFT library for Level Zero (must be ON if BUILD_SYCL=ON)
BUILD_OPENCL           Build FFT library for OpenCL (must be ON if BUILD_SYCL=ON)
ENABLE_WARNINGS        Enable strict warnings
NO_DOUBLE_PRECISION    Disable double precision in benchmarks and tests; useful if GPU
                       has no support for double precision
USE_CUDA               Build CUDA benchmark and examples
USE_MKL                Build MKL benchmark
USE_VKFFT              Build VkFFT benchmark
====================== ============

Linking in a CMake project
==========================

The project builds the three different libraries bbfft-level-zero, bbfft-opencl, and
bbfft-sycl. The first library uses the Level Zero run-time, the second library uses
the OpenCL library, and the last library the SYCL run-time.
Using the BUILD_(SYCL, LEVEL_ZERO, OPENCL) options one can choose which libraries are built.
For example, if you do not use SYCL in your project, you can disable the SYCL build
and as such do not need a C++ compiler with SYCL support.
Note that bbfft-sycl depends on bbfft-level-zero and bbfft-opencl and cannot be
built stand-alone.

For each of the three libraries, CMake targets are exported and installed along with the
libraries and headers. Hence, use the find_package mechanism in your CMake project as following:

.. code:: cmake

    find_package(bbfft-sycl REQUIRED)
    # or
    find_package(bbfft-level-zero REQUIRED)
    # or
    find_package(bbfft-opencl REQUIRED)

You can omit the REQUIRED flag.
For non-standard installation directories you might need to add the installation
location to the CMAKE_PREFIX_PATH.
For specifically requesting the static or shared library version use

.. code:: cmake

    find_package(bbfft-sycl REQUIRED static)
    # or
    find_package(bbfft-sycl REQUIRED shared)

To link the library and to set include directories you only need

.. code:: cmake

    target_link_libraries(your-target PRIVATE bbfft::bbfft-sycl)
