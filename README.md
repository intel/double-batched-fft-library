<h1 align="center">
  Double-Batched FFT Library
</h1>

<h4 align="center">
  A library for computing the Discrete Fourier Transform; targeting Graphics Processing Units; supporting OpenCL, Level Zero, and SYCL; with double-batching.
</h4>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a>
</p>

## Introduction

The Double-Batched FFT Library is a library for computing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).
GPU support is enabled via SYCL, OpenCL, or Level Zero.

A distinctive feature is the support of double-batching. That is, given the M x N_1 x ... x N_d x K input tensor, where the
Fourier transform shall be taken over the N-modes, the library support batching over the M-mode and K-mode in a single kernel.
Single-batching, offered by many FFT libraries, is included as the special case M=1.

## Features

* Forward and backward FFTs with complex input data and complex output data (c2c)
* FFTs with real input data and complex output data (r2c)
* FFTs with complex input data and real output data (c2r)
* 1d, 2d, and 3d FFTs
* Single and double precision
* Single-batching and double-batching
* User callbacks written in OpenCL-C for loads and stores (only in 1d)
* Optimized for small FFTs with N <= 512

## Installation

Install the oneAPI Base Toolkit. Build and install the project with

```bash
cmake -Bbuild -S. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/installation
```

## Documentation

### Online

The documentation is available at https://intel.github.io/double-batched-fft-library/

### Local build

Install Doxygen and Python dependencies:

```bash
apt install doxygen
pip install -r docs/requirements.txt
```

Build and install with CMake (add -DBUILD_DOCUMENTATION=ON). To read the docs:

```bash
cd /path/to/installation/share/doc/bbfft
python -m http.server 8000
```

Open http://127.0.0.1:8000/ in a browser.

## License

[BSD 3-Clause License](LICENSE.md)
