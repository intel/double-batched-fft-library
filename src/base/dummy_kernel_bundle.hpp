// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DUMMY_KERNEL_BUNDLE_20230202_HPP
#define DUMMY_KERNEL_BUNDLE_20230202_HPP

#include <string>
#include <vector>

namespace bbfft {

class dummy_kernel_bundle {
  public:
    using kernel_type = int;
    inline kernel_type create_kernel(std::string) { return 0; }
    inline std::vector<uint8_t> get_binary() const { return {0}; }
};

} // namespace bbfft

#endif // DUMMY_KERNEL_BUNDLE_20230202_HPP
