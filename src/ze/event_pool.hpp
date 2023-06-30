// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EVENT_POOL_20220610_HPP
#define EVENT_POOL_20220610_HPP

#include <level_zero/ze_api.h>

#include <cstdint>
#include <vector>

namespace bbfft::ze {

class event_pool {
  public:
    event_pool(ze_context_handle_t context, uint32_t pool_size);
    ~event_pool();

    event_pool(event_pool const &) = delete;
    event_pool(event_pool &) = delete;
    event_pool &operator=(event_pool const &) = delete;
    event_pool &operator=(event_pool &&) = delete;

    ze_event_handle_t get_event();
    void resize(uint32_t pool_size);

  private:
    void create_pool(uint32_t pool_size);
    void destroy_pool();

    ze_context_handle_t context_;
    ze_event_pool_handle_t event_pool_;
    std::vector<ze_event_handle_t> events_;
    uint32_t event_index_;
};

} // namespace bbfft::ze

#endif // EVENT_POOL_20220610_HPP
