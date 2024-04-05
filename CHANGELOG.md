# Changelog

## [0.5.1] - 2024-04-05
* clir: Fix vloadn
* clir: Add support for atomics
* clir: Add type qualifiers
* clir: Add while loop builder
* clir: Bump version to 0.5.1

## [0.5.0] - 2024-02-26

* Added compiler options to online compiler interface
* Introduce new API for Level Zero to support regular command lists and to better handle events
* Refactored FFT generator
* Implemented real N-point FFT with N/2-point complex FFT (for even N); improved small batch real FFT and factor2 slm FFT performance
*  "uint3 mnk" was removed from the callback signature. Codes that used the (undocumented and experimental) callback feature need to update their callbacks and compute mnk from offset.
* Extended factor2 slm algorithm to support more than 2 factors.
* Added new in-register FFT algorithm that needs less multiplications.

## [0.4.0] - 2023-05-11

* Introduce JIT caching feature
* Support for ahead-of-time compilation of FFT kernels
* Specialization for CPU devices added (for OpenCL CPU run-time)

## [0.3.6] - 2023-02-10

* Moved clir to top-level; clir is now a self-contained project
* Support out-of-place nd-fft; throw bad\_configuration if stride is unsupported (nd-fft only works for default tensor layouts)

## [0.3.5] - 2022-01-25

* Support gcc 8.5

## [0.3.4] - 2022-01-17

* Fix Ubuntu build

## [0.3.3] - 2022-01-16

* Improve r2c performance in some cases
* Fix identity tests

## [0.3.2] - 2022-01-11

* Add $ORIGIN to RPATH of installed libraries.

## [0.3.1] - 2022-12-08

Initial release
