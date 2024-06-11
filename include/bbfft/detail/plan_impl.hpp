// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PLAN_IMPL_20221205_HPP
#define PLAN_IMPL_20221205_HPP

#include "bbfft/mem.hpp"

#include <cstdint>
#include <utility>
#include <vector>

/**
 * @brief bbfft namespace
 */
namespace bbfft {
namespace detail {
/**
 * @brief Interface for plan implementations
 *
 * @tparam EventT Event type of underlying run-time
 */
template <typename EventT> class plan_impl {
  public:
    using event_t = EventT; ///< event type
    /**
     * @brief Dtor
     */
    virtual ~plan_impl() {}

    /**
     * @brief Execute plan
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     *
     * @return Completion event
     */
    virtual auto execute(mem const &in, mem const &out) -> event_t {
        return execute(in, out, std::vector<event_t>{});
    }
    /**
     * @brief Execute plan
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param dep_event Event to wait on before launching
     *
     * @return Completion event
     */
    virtual auto execute(mem const &in, mem const &out, event_t dep_event) -> event_t {
        return execute(in, out, std::vector<event_t>{std::move(dep_event)});
    }
    /**
     * @brief Execute plan
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param dep_events Events to wait on before launching
     *
     * @return Completion event
     */
    virtual auto execute(mem const &in, mem const &out, std::vector<event_t> const &dep_events)
        -> event_t = 0;
};

/**
 * @brief Interface for plan implementations with unmanaged events
 *
 * @tparam EventT Event type of underlying run-time
 */
template <typename EventT> class plan_unmanaged_event_impl {
  public:
    using event_t = EventT; ///< event type
    /**
     * @brief Dtor
     */
    virtual ~plan_unmanaged_event_impl() {}

    /**
     * @brief Execute plan
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param signal_event Event signaled on FFT completion [Optional]
     * @param num_wait_events Number of events to wait on before launch; must be zero if wait_events
     * == nullptr [Optional]
     * @param wait_events Pointer to events to wait on before launch; must point to at least
     * num_wait_events [Optional]
     */
    virtual void execute(mem const &in, mem const &out, event_t signal_event,
                         std::uint32_t num_wait_events, event_t *wait_events) = 0;
};
} // namespace detail
} // namespace bbfft

#endif // PLAN_IMPL_20221205_HPP
