// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PLAN_IMPL_20221205_HPP
#define PLAN_IMPL_20221205_HPP

#include <vector>

/**
 * @brief bbfft namespace
 */
namespace bbfft {
namespace detail {
template <typename EventT> class plan_impl {
  public:
    using event_t = EventT;
    virtual ~plan_impl() {}
    virtual auto execute(void const *in, void *out) -> event_t {
        return execute(in, out, std::vector<event_t>{});
    }
    virtual auto execute(void const *in, void *out, event_t dep_event) -> event_t {
        return execute(in, out, std::vector<event_t>{dep_event});
    }
    virtual auto execute(void const *in, void *out, std::vector<event_t> const &dep_events)
        -> event_t = 0;
};
} // namespace detail
} // namespace bbfft

#endif // PLAN_IMPL_20221205_HPP
