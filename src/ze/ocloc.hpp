// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef OCLOC_20250226_HPP
#define OCLOC_20250226_HPP

#include <cstdint>

namespace bbfft::ze {

using oclocInvoke_t = int (*)(unsigned int numArgs, const char *argv[],
                              const std::uint32_t numSources, const std::uint8_t **dataSources,
                              const std::uint64_t *lenSources, const char **nameSources,
                              const std::uint32_t numInputHeaders,
                              const std::uint8_t **dataInputHeaders,
                              const std::uint64_t *lenInputHeaders, const char **nameInputHeaders,
                              std::uint32_t *numOutputs, std::uint8_t ***dataOutputs,
                              std::uint64_t **lenOutputs, char ***nameOutputs);

using oclocFreeOutput_t = int (*)(std::uint32_t *numOutputs, std::uint8_t ***dataOutputs,
                                  std::uint64_t **lenOutputs, char ***nameOutputs);

auto get_oclocInvoke() -> oclocInvoke_t;
auto get_oclocFreeOutput() -> oclocFreeOutput_t;

} // namespace bbfft::ze

#endif // OCLOC_20250226_HPP
