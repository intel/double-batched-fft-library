// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EVENT_POOL_20220610_HPP
#define EVENT_POOL_20220610_HPP

#include <level_zero/ze_api.h>

#include <cstdint>

namespace bbfft::ze {

class event_pool {
  public:
    static constexpr uint32_t event_pool_size = 256;

    event_pool(ze_context_handle_t context);
    ~event_pool();

    ze_event_handle_t create_event();

  private:
    ze_event_pool_handle_t event_pool_;
    uint32_t event_index_;
};

} // namespace bbfft::ze

#endif // EVENT_POOL_20220610_HPP
