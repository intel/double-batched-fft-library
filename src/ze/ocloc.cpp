// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ocloc.hpp"

#include <dlfcn.h>
#include <memory>
#include <stdexcept>

namespace bbfft::ze {

class ocloc_library_handle {
  public:
    ocloc_library_handle();
    ~ocloc_library_handle();

    ocloc_library_handle(ocloc_library_handle const &) = delete;
    ocloc_library_handle(ocloc_library_handle &&) = delete;
    ocloc_library_handle &operator=(ocloc_library_handle const &) = delete;
    ocloc_library_handle &operator=(ocloc_library_handle &&) = delete;

    inline auto get_oclocInvoke() -> oclocInvoke_t {
        return reinterpret_cast<oclocInvoke_t>(oclocInvoke_);
    }
    inline auto get_oclocFreeOutput() -> oclocFreeOutput_t {
        return reinterpret_cast<oclocFreeOutput_t>(oclocFreeOutput_);
    }

  private:
    void *handle_ = nullptr;
    void *oclocInvoke_ = nullptr;
    void *oclocFreeOutput_ = nullptr;
};

ocloc_library_handle::ocloc_library_handle() {
    handle_ = dlopen("libocloc.so", RTLD_NOW);
    if (!handle_) {
        throw std::runtime_error(dlerror());
    }

    auto const load_symbol = [&](void *&symbol_handle, const char *symbol) {
        dlerror(); // Clear existing error

        symbol_handle = dlsym(handle_, symbol);
        char *err = dlerror();
        if (err) {
            throw std::runtime_error(err);
        }
    };

    load_symbol(oclocInvoke_, "oclocInvoke");
    load_symbol(oclocFreeOutput_, "oclocFreeOutput");
}

ocloc_library_handle::~ocloc_library_handle() {
    if (handle_) {
        dlclose(handle_);
    }
}

static auto ocloc_library = std::unique_ptr<ocloc_library_handle>{nullptr};

auto get_oclocInvoke() -> oclocInvoke_t {
    if (!ocloc_library) {
        ocloc_library = std::make_unique<ocloc_library_handle>();
    }
    return ocloc_library->get_oclocInvoke();
}

auto get_oclocFreeOutput() -> oclocFreeOutput_t {
    if (!ocloc_library) {
        ocloc_library = std::make_unique<ocloc_library_handle>();
    }
    return ocloc_library->get_oclocFreeOutput();
}

} // namespace bbfft::ze
