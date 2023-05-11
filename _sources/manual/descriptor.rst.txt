.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

.. _descriptor:

=================
Descriptor format
=================

The ``bbfft-aot-generate`` and ``bbfft-offline-generate`` tools use a short FFT descriptor format.
The general form of a FFT descriptor is (in `ABNF syntax <https://www.ietf.org/rfc/rfc5234.txt>`_):

.. code:: abnf

    fft_descriptor  =  precision domain direction placement shape [istride] [ostride]
    precision       =  "s" / "d"
    domain          =  "c" / "r"
    direction       =  "f" / "b"
    placement       =  "i" / "o"
    shape           =  [number "."] number *2("x" number) ["*" number]
    number          =  1*DIGIT
    istride         =  "i" stride
    ostride         =  "o" stride
    stride          =  number 2*4("," number)

The precision, domain, direction, and placement options are:

.. list-table::
   :widths: 1 10
   :header-rows: 1

   * - Option
     - Meaning
   * - **s**
     - Single precision
   * - **d**
     - Double precision
   * - **c**
     - Complex input and complex output
   * - **r**
     - Real input and complex output in forward direction;
       complex input and real output in backward direction
   * - **f**
     - Forward direction
   * - **b**
     - Backward direction
   * - **i**
     - In-place
   * - **o**
     - Out-of-place

The shape rule defines the shape of the :math:`M \times N_1 \times \dots \times N_D \times K` tensor
(see :ref:`data-layout`).
Here, the number before the dot is the left-batch size :math:`M`.
If omitted, then :math:`M=1`.

The numbers before and after "x" correspond to :math:`N_1,N_2,N_3` and the number of "x"
is equal to the FFT dimension plus 1.
E.g., "5" corresponds to :math:`N_1=5` and FFT dimension 1, and "8x16x32" corresponds to
:math:`N_1=8,N_2=16,N_3=32` and FFT dimension 3.

The last number, after "*", is the right-batch size :math:`K`. If omitted, then :math:`K=1`.

The istride and ostride rules can be used to override the default strides of the input and output tensor.
The sequence of numbers corresponds to :math:`s_0,\dots,s_{D+1}` (see :ref:`data-layout`).
The length of the sequence of numbers must be equal to the FFT dimension plus two.

.. note:: 

   Custom strides for in-place transforms need to be repeated, i.e. both istride and ostride
   need to be given.

Examples
========

.. list-table::
   :widths: 1 10
   :header-rows: 1

   * - Gibberish
     - English
   * - srfi5
     - Single precision real-to-complex 5-point FFT with in-place data layout.
   * - dcbi4*5
     - Double precision complex-to-complex 4-point FFT in backward direction with
       in-place data layout and right-batch size of 5.
   * - dcbi4.5
     - Double precision complex-to-complex 5-point FFT in backward direction with
       in-place data layout and left-batch size of 4.
   * - drfo5x6x7
     - Double precision real-to-complex 3-D FFT on :math:`5x6x7` tensor with out-of-place data layout.
   * - srbo4.5x6*7
     - Double precision complex-to-real 2-D FFT on :math:`5x6` tensor with left-batch size 4 and
       right-batch size 7 with out-of-place data layout.
   * - scfo16*32i1,1,20
     - Single precision complex-to-complex 16-point FFT with right-batch size 32 with out-place data-layout
       and input stride override (20 complex numbers between batch elements in input tensor).
