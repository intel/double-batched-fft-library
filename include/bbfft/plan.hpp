// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PLAN_20220412_HPP
#define PLAN_20220412_HPP

#include "bbfft/detail/plan_impl.hpp"

#include <memory>
#include <utility>
#include <vector>

/**
 * @brief bbfft namespace
 */
namespace bbfft {

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
template <typename EventT> class plan {
  public:
    /**
     * @brief Event type returned by execute functions
     */
    using event_t = EventT;

    /**
     * @brief The "empty" plan. Cannot be used.
     */
    plan() : impl_(nullptr) {}
    /**
     * @brief Check whether a plan is non-empty.
     *
     * @return True if plan is non-empty.
     */
    explicit operator bool() const noexcept { return bool(impl_); }

    /**
     * @brief Construct plan from plan_impl.
     *
     * Use ::make_plan instead of this constructor.
     *
     * @param impl Implementation
     */
    plan(std::shared_ptr<detail::plan_impl<EventT>> impl) : impl_(std::move(impl)) {}

    /**
     * @brief Execute plan (out-of-place)
     *
     * @param in Pointer to input tensor
     * @param out Pointer to output tensor
     *
     * @return Completion event
     */
    auto execute(void const *in, void *out) -> event_t { return impl_->execute(in, out); }
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
        return impl_->execute(in, out, dep_event);
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
        return impl_->execute(in, out, dep_events);
    }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     *
     * @return Completion event
     */
    auto execute(void *inout) -> event_t { return impl_->execute(inout, inout); }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     * @param dep_event Event to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void const *inout, event_t dep_event) -> event_t {
        return impl_->execute(inout, inout, dep_event);
    }
    /**
     * @brief Execute plan (in-place)
     *
     * @param inout Pointer to input and output tensor
     * @param dep_events Events to wait on before launching
     *
     * @return Completion event
     */
    auto execute(void const *inout, std::vector<event_t> const &dep_events) -> event_t {
        return impl_->execute(inout, inout, dep_events);
    }

  private:
    std::shared_ptr<detail::plan_impl<EventT>> impl_;
};

} // namespace bbfft

#endif // PLAN_20220412_HPP
