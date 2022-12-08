// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SHARED_HANDLE_20220602_HPP
#define SHARED_HANDLE_20220602_HPP

#include <memory>

namespace bbfft::ze {

template <typename T, typename Enable = void> class shared_handle;

template <typename T>
class shared_handle<
    T, typename std::enable_if_t<sizeof(T) <= sizeof(void *) && alignof(T) <= sizeof(void *)>> {
  public:
    shared_handle() : handle_(nullptr, Deleter(nullptr)) {}
    shared_handle(T t, void (*delete_handle)(T t))
        : handle_(reinterpret_cast<void *>(t), Deleter(delete_handle)) {}
    T get() const { return reinterpret_cast<T>(handle_.get()); }

  private:
    struct Deleter {
        Deleter(void (*delete_handle)(T t)) : delete_handle(delete_handle) {}
        void operator()(void *ptr) {
            if (delete_handle) {
                delete_handle(reinterpret_cast<T>(ptr));
            }
        }
        void (*delete_handle)(T t);
    };
    std::shared_ptr<void> handle_;
};

} // namespace bbfft::ze

#endif // SHARED_HANDLE_20220602_HPP
