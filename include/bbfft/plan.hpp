// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PLAN_20220412_HPP
#define PLAN_20220412_HPP

#include "bbfft/detail/plan_impl.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

/**
 * @brief bbfft namespace
 */
namespace bbfft {

/**
 * @brief Basic functionality of a plan
 *
 * @tparam Impl plan implementation type
 */
template <class Impl> class base_plan {
  public:
    /**
     * @brief The "empty" plan. Cannot be used.
     */
    base_plan() : impl_(nullptr) {}
    /**
     * @brief Check whether a plan is non-empty.
     *
     * @return True if plan is non-empty.
     */
    inline explicit operator bool() const noexcept { return bool(impl_); }

    /**
     * @brief Construct plan from Impl.
     *
     * Use ::make_plan instead of this constructor.
     *
     * @param impl Implementation
     */
    base_plan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

  protected:
    std::shared_ptr<Impl> impl_;
};

/**
 * @brief The plan class contains the kernels to run a specific FFT configuration.
 *
 * FFT kernels are specialized for the required problem size. Plans store the best
 * kernels for the requested problem size or problem configuration.
 *
 * Plan objects are not created directly but via ::make_plan.
 *
 * @tparam EventT event type of the compute runtime
 */
template <typename EventT> class plan : public base_plan<detail::plan_impl<EventT>> {
  public:
    using base_plan<detail::plan_impl<EventT>>::base_plan;

    /**
     * @brief Event type returned by execute functions
     */
    using event_t = EventT;

    /**
     * @brief Execute plan (out-of-place)
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     *
     * @return Completion event
     */
    auto execute(void const *in, void *out) -> event_t { return this->impl_->execute(in, out); }
    /**
     * @brief Execute plan (out-of-place)
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param dep_event Event to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void const *in, void *out, event_t dep_event) -> event_t {
        return this->impl_->execute(in, out, std::move(dep_event));
    }
    /**
     * @brief Execute plan (out-of-place)
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param dep_events Events to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void const *in, void *out, std::vector<event_t> const &dep_events) -> event_t {
        return this->impl_->execute(in, out, dep_events);
    }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     *
     * @return Completion event
     */
    auto execute(void *inout) -> event_t { return this->impl_->execute(inout, inout); }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     * @param dep_event Event to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void *inout, event_t dep_event) -> event_t {
        return this->impl_->execute(inout, inout, dep_event);
    }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     * @param dep_events Events to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void *inout, std::vector<event_t> const &dep_events) -> event_t {
        return this->impl_->execute(inout, inout, dep_events);
    }
};

/**
 * @brief The plan class contains the kernels to run a specific FFT configuration.
 *
 * In some APIs, e.g. Level Zero, events are not returned by a kernel launch but
 * created by the user. In order to enable the API's full capability, such as event
 * scope settings and event recycling, a different low-level interface is offered
 * by plan_unmanaged_event.
 *
 * @tparam EventT event type of the compute runtime
 */
template <typename EventT>
class plan_unmanaged_event : public base_plan<detail::plan_unmanaged_event_impl<EventT>> {
  public:
    using base_plan<detail::plan_unmanaged_event_impl<EventT>>::base_plan;

    /**
     * @brief Event type returned by execute functions
     */
    using event_t = EventT;

    /**
     * @brief Execute plan (out-of-place)
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     * @param signal_event Event signaled on FFT completion [Optional]
     * @param num_wait_events Number of events to wait on before launch; must be zero if wait_events
     * == nullptr [Optional]
     * @param wait_events Pointer to events to wait on before launch; must point to at least
     * num_wait_events [Optional]
     */
    void execute(void const *in, void *out, event_t signal_event = nullptr,
                 std::uint32_t num_wait_events = 0, event_t *wait_events = nullptr) {
        this->impl_->execute(in, out, signal_event, num_wait_events, wait_events);
    }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     * @param out Pointer to output tensor
     * @param signal_event Event signaled on FFT completion [Optional]
     * @param num_wait_events Number of events to wait on before launch; must be zero if wait_events
     * == nullptr [Optional]
     * @param wait_events Pointer to events to wait on before launch; must point to at least
     * num_wait_events [Optional]
     */
    void execute(void *inout, event_t signal_event = nullptr, std::uint32_t num_wait_events = 0,
                 event_t *wait_events = nullptr) {
        this->impl_->execute(inout, inout, signal_event, num_wait_events, wait_events);
    }
};

} // namespace bbfft

#endif // PLAN_20220412_HPP
