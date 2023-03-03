# Ahead-of-time compilation example

In this example we show how to compile FFT kernels ahead-of-time (AOT).
AOT compilation is useful if the plan creation overhead becomes significant and 
the set of required FFT plans is known at compile-time.

We need a three-step compilation process: First, we compile a generator that uses
the double batched FFT library to generate the FFT kernels (that is, OpenCL source code) and
compiles the source code to the native device binary.
In the second step, the linker is used to create an object from the native device binary.
The third step is to compile the main example where we link the object that included the native binary
that we generated in the second step.

## 1. Generator (generate.cpp)

The source code for the FFT kernels is generated using `generate_fft_kernels` from `bbfft/generator.hpp`:

```c++
    std::ostringstream oss;
    device_info info = {1024, {16, 32}, 2, 128 * 1024};
    auto kernel_names = generate_fft_kernels(oss, configurations(), info);
```

The source code for the configurations given by `configuration()` is written to a stringstream.
The code is specialized for a device with properties given by `device_info`.
(Refer to the API documentation for the meaning of the numbers in `device_info`.)

`generate_fft_kernels` returns a list of kernel names.
We generate a cpp file that contains that list in the global variable
```c++
std::unordered_set<std::string> aot_compiled_kernels;
```

Finally, the source code is compiled for "pvc" and the binary is saved in `kernel_file`.
```c++
    auto bin = ze::compile_to_native(oss.str(), "pvc");
    kernel_file.write(reinterpret_cast<char *>(bin.data()), bin.size());
```

## 2. Object file creation (CMakeLists.txt)

We create an object file from the native binary by adding a custom command in our `CMakeLists.txt`:

```cmake
add_custom_command(
    ...
    COMMAND aot-generate ${BIN_FILE} ${NAMES_FILE}    
    COMMAND ${CMAKE_LINKER} -r -b binary -o ${OBJ_FILE} ${BIN_FILE}
    ...
)
```

The first command calls our generator (generate.cpp).
In the second command, we link the binary blob created by the generator into an object file using the binary input format (`-b binary`).

## 3. Main example

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
We create the class `aot_cache` that derives from `jit_cache`.
The constructor of `aot_cache` loads the binary blob and stores it in either a `cl_program` or a
`ze_module_handle_t`, depending on the back-end selected in the SYCL queue.

```c++
aot_cache::aot_cache(::sycl::queue q) {
    extern const uint8_t _binary_kernels_bin_start, _binary_kernels_bin_end;

    auto handle = bbfft::sycl::build_native_module(
        &_binary_kernels_bin_start, &_binary_kernels_bin_end - &_binary_kernels_bin_start,
        q.get_context(), q.get_device());
    module_ = bbfft::sycl::make_shared_handle(handle, q.get_backend());
}
```

Here, we employ the symbols of the object file to direct `build_native_module` to the binary blob.
The last ingredient we need is a get function that returns the native 1module in the case that the FFT kernel
was compiled ahead-of-time.

```c++
auto aot_cache::get(jit_cache_key const &key) const -> shared_handle<module_handle_t> {
    if (auto it = aot_compiled_kernels.find(key.kernel_name); it != aot_compiled_kernels.end()) {
        return module_;
    }
    return {};
}
```
