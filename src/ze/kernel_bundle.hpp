// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ZE_KERNEL_BUNDLE_20221208_HPP
#define ZE_KERNEL_BUNDLE_20221208_HPP

#include "kernel.hpp"
#include "shared_handle.hpp"

#include <level_zero/ze_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace bbfft::ze {

class kernel_bundle {
  public:
    kernel_bundle();
    kernel_bundle(std::string source, ze_context_handle_t context, ze_device_handle_t device);
    kernel_bundle(uint8_t const *binary, std::size_t binary_size, ze_context_handle_t context,
                  ze_device_handle_t device);
    kernel create_kernel(std::string name);

    inline ze_module_handle_t get_native() const { return module_.get(); }
    std::vector<uint8_t> get_binary() const;

  private:
    inline static void delete_module(ze_module_handle_t k) { zeModuleDestroy(k); }

    shared_handle<ze_module_handle_t> module_;
};

} // namespace bbfft::ze

#endif // ZE_KERNEL_BUNDLE_20221208_HPP
