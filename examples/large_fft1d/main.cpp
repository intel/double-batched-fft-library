// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "args.hpp"
#include "fft1d.hpp"
#include "fft1d_custom.hpp"
#include "test_signal.hpp"

#include "bbfft/cl/error.hpp"
#include "bbfft/configuration.hpp"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace bbfft;

void test_fft(configuration const &cfg, double min_time, cl_command_queue queue, cl_context context,
              cl_device_id device);

int main(int argc, char **argv) {
    auto parser = arg_parser{};
    args a;
    try {
        a = parser.parse_args(argc, argv);
    } catch (std::runtime_error const &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
    if (a.help) {
        parser.show_help(std::cout);
        return 0;
    }

    auto platforms = std::vector<cl_platform_id>{};
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_int err;

    try {
        cl_uint platform_count = 0;
        CL_CHECK(clGetPlatformIDs(platform_count, NULL, &platform_count));

        platforms.resize(platform_count);
        CL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), &platform_count));

        cl_uint device_count = 0;
        for (auto const &platform : platforms) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, NULL, &device_count);
            if (err == CL_SUCCESS && device_count > 0) {
                device_count = 1;
                CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, &device, NULL));
                char name[512] = {0};
                clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name) - 1, name, NULL);
                std::cout << "Platform: " << name << std::endl;
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
                std::cout << "Device: " << name << std::endl;
                break;
            }
        }
        if (device_count == 0) {
            CL_CHECK(CL_DEVICE_NOT_FOUND);
        }

        context = clCreateContext(NULL, device_count, &device, NULL, NULL, &err);
        CL_CHECK(err);

        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        CL_CHECK(err);

        std::cout << "descriptor,time,repetitions,bytes,bandwidth" << std::endl;
        for (auto const &cfg : a.cfgs) {
            test_fft(cfg, a.min_time, queue, context, device);
        }
    } catch (std::exception const &e) {
        std::cerr << "==> Error: " << e.what() << std::endl;
    }

    if (queue) {
        clReleaseCommandQueue(queue);
    }
    if (context) {
        clReleaseContext(context);
    }

    return 0;
}

void test_fft(configuration const &cfg, double min_time, cl_command_queue queue, cl_context context,
              cl_device_id device) {
    const bool inplace = cfg.istride == bbfft::default_istride(cfg.dim, cfg.shape, cfg.type, true);

    const auto element_size = static_cast<int>(cfg.fp);
    const auto i_element_size = cfg.type != transform_type::r2c ? 2 * element_size : element_size;
    const auto o_element_size = 2 * element_size;
    const std::size_t isize = i_element_size * cfg.istride[2] * cfg.shape[2];
    const std::size_t osize = o_element_size * cfg.ostride[2] * cfg.shape[2];
    const std::size_t bytes = isize + osize;
    auto in = std::vector<double>(isize / sizeof(double));
    auto out = std::vector<double>(osize / sizeof(double));
    const long first_mode = cfg.shape[1] / 8;
    cl_mem in_device = nullptr, in_copy = nullptr, out_device_buffer = nullptr;
    cl_int err;

    auto const print_result = [&](auto time, auto n, auto bw) {
        std::cout << cfg << "," << time << "," << n << "," << bytes << "," << bw << std::endl;
    };

    try {
        // auto fft = fft1d(cfg, queue, context, device);
        auto fft = fft1d_custom(cfg, queue, context, device);

        test_signal_1d(in.data(), cfg, first_mode);

        in_device = clCreateBuffer(context, CL_MEM_READ_WRITE, isize, nullptr, &err);
        CL_CHECK(err);
        cl_mem out_device = in_device;
        if (!inplace) {
            out_device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, osize, nullptr, &err);
            CL_CHECK(err);
            out_device = out_device_buffer;
        } else {
            in_copy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, isize,
                                     nullptr, &err);
            CL_CHECK(err);
        }

        CL_CHECK(clEnqueueWriteBuffer(queue, in_device, CL_TRUE, 0, isize, in.data(), 0, nullptr,
                                      nullptr));
        if (inplace) {
            CL_CHECK(
                clEnqueueCopyBuffer(queue, in_device, in_copy, 0, 0, isize, 0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));
        }

        // auto plan = make_plan(cfg, queue);
        // auto e = plan.execute(in_device, out_device);
        auto e = fft.execute(in_device, out_device, {});
        CL_CHECK(clWaitForEvents(1, &e));
        clReleaseEvent(e);

        CL_CHECK(clEnqueueReadBuffer(queue, out_device, CL_TRUE, 0, osize, out.data(), 0, nullptr,
                                     nullptr));

        if (!check_signal_1d(out.data(), cfg, first_mode, &std::cerr)) {
            std::cerr << "Check signal failed" << std::endl;
        }
        double min_exec_time = std::numeric_limits<double>::max();
        double elapsed_time = 0.0;
        std::size_t nrepeat = 0;
        while (elapsed_time < min_time) {
            if (inplace) {
                CL_CHECK(clEnqueueCopyBuffer(queue, in_copy, in_device, 0, 0, isize, 0, nullptr,
                                             nullptr));
                CL_CHECK(clFinish(queue));
            }

            auto start = std::chrono::high_resolution_clock::now();

            auto e = fft.execute(in_device, out_device, {});
            CL_CHECK(clWaitForEvents(1, &e));
            clReleaseEvent(e);

            auto end = std::chrono::high_resolution_clock::now();
            double exec_time_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            double exec_time = exec_time_ns * 1e-9;
            min_exec_time = std::min(min_exec_time, exec_time);
            elapsed_time += exec_time;
            ++nrepeat;
        }
        print_result(min_exec_time, nrepeat, bytes / min_exec_time * 1e-9);
    } catch (std::exception const &e) {
        std::cerr << "==> Error: " << e.what() << std::endl;
    }

    if (in_device) {
        clReleaseMemObject(in_device);
    }
    if (in_copy) {
        clReleaseMemObject(in_copy);
    }
    if (out_device_buffer) {
        clReleaseMemObject(out_device_buffer);
    }
}
