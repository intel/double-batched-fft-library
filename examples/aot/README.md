# Ahead-of-time compilation example

In this example we show how to compile FFT kernels ahead-of-time (AOT).
AOT compilation is useful if the plan creation overhead becomes significant and 
the set of required FFT plans is known at compile-time.

Refer to the [documentation](../../docs/manual/caching.rst) for a general explanation of the implementation of AOT caching.
