// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "event_pool.hpp"
#include "bbfft/ze/error.hpp"

namespace bbfft::ze {

event_pool::event_pool(ze_context_handle_t context) {
    // FIXME: How large does the pool need to be?
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE, event_pool_size};
    ZE_CHECK(zeEventPoolCreate(context, &event_pool_desc, 0, nullptr, &event_pool_));
    event_index_ = 0;
}

event_pool::~event_pool() { zeEventPoolDestroy(event_pool_); }

ze_event_handle_t event_pool::create_event() {
    ze_event_handle_t event;
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index_, 0, 0};
    event_index_ = (event_index_ + 1) % event_pool_size;
    ZE_CHECK(zeEventCreate(event_pool_, &event_desc, &event));
    return event;
}

} // namespace bbfft::ze
