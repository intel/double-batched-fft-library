// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CL_ARGUMENT_HANDLER_20221129_HPP
#define CL_ARGUMENT_HANDLER_20221129_HPP

#include "bbfft/cl/error.hpp"
#include "bbfft/mem.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstddef>
#include <stdexcept>

namespace bbfft::cl {

using clSetKernelArgMemPointerINTEL_t = cl_int (*)(cl_kernel kernel, cl_uint arg_index,
                                                   const void *arg_value);

class argument_handler {
  public:
    inline argument_handler() : clSetKernelArgMemPointerINTEL_{nullptr} {}
    inline argument_handler(cl_platform_id plat) {
        clSetKernelArgMemPointerINTEL_ =
            (clSetKernelArgMemPointerINTEL_t)clGetExtensionFunctionAddressForPlatform(
                plat, "clSetKernelArgMemPointerINTEL");
    }

    inline void set_arg(cl_kernel kernel, unsigned index, std::size_t size,
                        const void *value) const {
        CL_CHECK(clSetKernelArg(kernel, index, size, value));
    }

    inline void set_mem_arg(cl_kernel kernel, unsigned index, const void *value,
                            mem_type type) const {
        switch (type) {
        case mem_type::buffer:
            set_arg(kernel, index, sizeof(value), &value);
            return;
        case mem_type::usm_pointer:
            if (clSetKernelArgMemPointerINTEL_ == nullptr) {
                throw cl::error("OpenCL unified shared memory extension unavailable",
                                CL_INVALID_COMMAND_QUEUE);
            }
            CL_CHECK(clSetKernelArgMemPointerINTEL_(kernel, index, value));
            return;
        case mem_type::svm_pointer:
            CL_CHECK(clSetKernelArgSVMPointer(kernel, index, value));
            return;
        }
        throw std::logic_error("Unsupported mem type");
    }

  private:
    clSetKernelArgMemPointerINTEL_t clSetKernelArgMemPointerINTEL_;
};

} // namespace bbfft::cl

#endif // CL_ARGUMENT_HANDLER_20221129_HPP
