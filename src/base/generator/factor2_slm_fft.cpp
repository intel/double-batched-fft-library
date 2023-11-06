// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"
#include "generator/f2fft_gen.hpp"
#include "math.hpp"
#include "prime_factorization.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace bbfft {

factor2_slm_configuration configure_factor2_slm_fft(configuration const &cfg,
                                                    device_info const &info) {
    bool const is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    std::size_t N = cfg.shape[1];
    std::size_t N_fft = N;
    if (is_real && N % 2 == 0) {
        N_fft /= 2;
    }
    std::size_t N_slm = N_fft;
    if (is_real) {
        N_slm = N_fft + 1;
    }

    std::size_t sgs = info.min_subgroup_size();
    std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);
    unsigned max_N_in_registers_without_spilling =
        (info.register_space_max() / 2) / (2 * sizeof_real) / sgs;

    auto factorization = std::vector<unsigned>{};
    for (unsigned index = 2; index <= 4; ++index) {
        factorization = factor(N_fft, index);
        auto fmax = *std::max_element(factorization.begin(), factorization.end());
        if (fmax < max_N_in_registers_without_spilling || is_prime(fmax)) {
            break;
        }
    }
    auto Nf_max = *std::max_element(factorization.begin(), factorization.end());
    auto N_parallel_max = product(factorization.begin(), factorization.end(), 1) / Nf_max;
    auto M = cfg.shape[0];

    std::size_t work_group_size_limit = info.max_subgroup_size();

    std::size_t Nb = min_power_of_2_greater_equal(N_parallel_max);
    std::size_t max_compute_Mb = info.max_work_group_size / Nb;
    std::size_t max_slm_Mb = info.local_memory_size / (2 * N_slm * sizeof_real);
    std::size_t max_Mb = std::min(max_compute_Mb, max_slm_Mb);
    if (max_Mb >= sgs) {
        max_Mb = max_Mb / sgs * sgs;
    }
    max_Mb = std::min(max_Mb, work_group_size_limit);
    std::size_t Mb = std::min(max_Mb, min_power_of_2_greater_equal(M));
    std::size_t max_compute_Kb = std::max(std::size_t(1), info.max_work_group_size / (Mb * Nb));
    std::size_t max_slm_Kb = info.local_memory_size / (2 * Mb * N_slm * sizeof_real);
    std::size_t max_Kb = std::min(max_compute_Kb, max_slm_Kb);
    std::size_t min_Kb = (sgs - 1) / (Mb * Nb) + 1;
    std::size_t Kb = std::min(cfg.shape[2], std::min(min_Kb, max_Kb));

    bool inplace_unsupported = is_real && Mb < M;

    auto istride = std::array<std::size_t, 3>{cfg.istride[0], cfg.istride[1], cfg.istride[2]};
    auto ostride = std::array<std::size_t, 3>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]};

    std::stringstream ss;
    return {
        static_cast<int>(cfg.dir),                                    // direction
        M,                                                            // M
        Mb,                                                           // Mb
        N,                                                            // N
        std::vector<int>(factorization.begin(), factorization.end()), // factorization
        Nb,                                                           // Nb
        Kb,                                                           // Kb
        sgs,                                                          // sgs
        cfg.fp,                                                       // precision
        cfg.type,                                                     // transform_type
        istride,                                                      // istride
        ostride,                                                      // ostride
        inplace_unsupported,                                          // inplace_unsupported
        cfg.callbacks.load_function,                                  // load_function
        cfg.callbacks.store_function                                  // store_function
    };
}

std::string factor2_slm_configuration::identifier() const {
    std::ostringstream oss;
    oss << "f2fft_" << (direction < 0 ? 'm' : 'p') << std::abs(direction) << "_M" << M << "_Mb"
        << Mb << "_N" << N << "_factorization";
    auto it = factorization.begin();
    if (it != factorization.end()) {
        oss << *it++;
        for (; it != factorization.end(); ++it) {
            oss << "x" << *it;
        }
    }
    oss << "_Nb" << Nb << "_Kb" << Kb << "_sgs" << sgs << "_f" << static_cast<int>(fp) * 8 << '_'
        << to_string(type) << "_is";
    for (auto const &is : istride) {
        oss << is << "_";
    }
    oss << "os";
    for (auto const &os : ostride) {
        oss << os << "_";
    }
    oss << "in" << inplace_unsupported;
    if (load_function) {
        oss << "_" << load_function;
    }
    if (store_function) {
        oss << "_" << store_function;
    }
    return oss.str();
}

void generate_factor2_slm_fft(std::ostream &os, factor2_slm_configuration const &cfg,
                              std::string_view name) {
    auto gen = std::unique_ptr<f2fft_gen>{};
    switch (cfg.type) {
    case transform_type::c2c:
        gen = std::make_unique<f2fft_gen_c2c>(cfg.N);
        break;
    case transform_type::r2c:
        if (cfg.N % 2 == 1) {
            gen = std::make_unique<f2fft_gen_r2c_double>(cfg.N);
        } else {
            gen = std::make_unique<f2fft_gen_r2c_half>(cfg.N);
        }
        break;
    case transform_type::c2r:
        if (cfg.N % 2 == 1) {
            gen = std::make_unique<f2fft_gen_c2r_double>(cfg.N);
        } else {
            gen = std::make_unique<f2fft_gen_c2r_half>(cfg.N);
        }
        break;
    default:
        break;
    }
    if (!gen) {
        throw std::logic_error("Internal logic error: Did you mess with the cfg.type field?");
    }
    gen->generate(os, cfg, name);
}

} // namespace bbfft
