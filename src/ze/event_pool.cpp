// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "event_pool.hpp"
#include "bbfft/ze/error.hpp"

namespace bbfft::ze {

event_pool::event_pool(ze_context_handle_t context, uint32_t pool_size) : context_(context) {
    create_pool(pool_size);
}

event_pool::~event_pool() { destroy_pool(); }

ze_event_handle_t event_pool::get_event() {
    event_index_ = (event_index_ + 1) % events_.size();
    return events_[event_index_];
}

void event_pool::resize(uint32_t pool_size) {
    destroy_pool();
    create_pool(pool_size);
}

void event_pool::create_pool(uint32_t pool_size) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, 0,
                                            pool_size};
    ZE_CHECK(zeEventPoolCreate(context_, &event_pool_desc, 0, nullptr, &event_pool_));
    events_.resize(pool_size);
    for (event_index_ = 0; event_index_ < pool_size; ++event_index_) {
        ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index_,
                                      ZE_EVENT_SCOPE_FLAG_DEVICE, 0};
        ZE_CHECK(zeEventCreate(event_pool_, &event_desc, &events_[event_index_]));
    }
}

void event_pool::destroy_pool() {
    for (auto &e : events_) {
        zeEventDestroy(e);
    }
    zeEventPoolDestroy(event_pool_);
}

} // namespace bbfft::ze
