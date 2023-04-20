# Ahead-of-time compilation example

In this example we show how to compile FFT kernels ahead-of-time (AOT).
AOT compilation is useful if the plan creation overhead becomes significant and 
the set of required FFT plans is known at compile-time.

We need a two-step process:
First, we compile FFT kernels to a native device binary using the `bbfft-aot-generate` tool
and use the GNU linker embed the native device binary in the application.
The second step is to create `aot_cache` in the application to lookup native device code
during plan creation.

## 1. Native device binary generation and linking

We generate the native device binary for selected FFT configurations using
the `bbfft-aot-generate` tool that comes with the Double-Batched FFT Library.
For example, in order to create kernels for a complex-to-complex FFT in single precision with
N=16,32 and a batch size of 1000 on Ponte Vecchio use

```bash
bbfft-aot-generate -d pvc kernels.bin scfi16*1000 scfi32*1000
```

Then the GNU linker is used to embed the native device binary in your application:
```bash
ld -r -b binary -o kernels.o kernels.bin
```
Linking `kernels.o` makes the symbols `_binary_kernels_bin_start` and `_binary_kernels_bin_end` available
that point to the native device binary.
 
CMake users can use the following workflow to automatise the above steps, e.g. for all real-to-complex
power of two FFTs until N=1024:

```cmake
find_package(bbfft-aot-generate REQUIRED)

set(N 2 4 8 16 32 64 128 256 512 1024)
foreach(n IN LISTS N)
    list(APPEND descriptor_list "srfi${n}*16384")
endforeach()

add_aot_kernels_to_target(TARGET <your-cmake-target> PREFIX kernels DEVICE pvc LIST ${descriptor_list})
```

## 2. Lookup native device code during plan creation

If everything worked, we find the object file `kernels.o` in the build folder.
The file looks something like the following:

```bash
$ nm -a build/examples/aot/kernels.o
0000000000142770 D _binary_kernels_bin_end
0000000000142770 A _binary_kernels_bin_size
0000000000000000 D _binary_kernels_bin_start
0000000000000000 d .data
```

Here, we have the symbols `_binary_kernels_bin_start` and `_binary_kernels_bin_end` that indicate the
start and end of the ahead-of-time compiled binary blob.
With these symbols we create an `aot_module` that we register with the `aot_cache`
```c++
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
```
Here, we employ the symbols of the object file to direct `create_aot_module` to the binary blob.
We have wrapped `create_aot_module` in a `try ... catch` block as it might fail, for example if
the application is run on a device requiring a different native device binary.
In that case, the `aot_module` is not registered with the `aot_cache` and we fall back to
just-in-time compilation.

During plan creation you need to pass the cache as argument such that plan creation benefits from
ahead-of-time compilation:
```c++
auto plan = bbfft::make_plan(cfg, q, &cache);
```
