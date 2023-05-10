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
    virtual auto execute(void const *in, void *out) -> event_t {
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
    virtual auto execute(void const *in, void *out, event_t dep_event) -> event_t {
        return execute(in, out, std::vector<event_t>{dep_event});
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
    virtual auto execute(void const *in, void *out, std::vector<event_t> const &dep_events)
        -> event_t = 0;
};
} // namespace detail
} // namespace bbfft

#endif // PLAN_IMPL_20221205_HPP
