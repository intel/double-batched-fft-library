// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "bbfft/configuration.hpp"
#include "generator/sbfft_gen.hpp"
#include "math.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace clir;

namespace bbfft {

small_batch_configuration configure_small_batch_fft(configuration const &cfg,
                                                    device_info const &info) {
    auto M = cfg.shape[0];
    std::size_t N = cfg.shape[1];
    std::size_t N_slm = N;
    std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);

    bool is_real = cfg.type == transform_type::r2c || cfg.type == transform_type::c2r;
    if (is_real) {
        N_slm = N / 2 + 1;
    }

    std::size_t sgs = info.min_subgroup_size();
    if (M == 1) {
        auto register_space = info.register_space_max();
        for (auto sgs_i : info.subgroup_sizes) {
            auto required_register_space = 2 * sizeof_real * N * sgs_i;
            if (sgs < sgs_i && required_register_space < register_space / 2) {
                sgs = sgs_i;
            }
        }
    }

    std::size_t Mb = 1;
    std::size_t max_work_group_size = std::min(std::size_t(128), info.max_work_group_size);
    std::size_t work_group_size_limit = info.max_subgroup_size();
    Mb = std::min(min_power_of_2_greater_equal(M), work_group_size_limit);

    std::size_t max_compute_Kb = max_work_group_size / Mb;
    std::size_t max_slm_Kb = info.local_memory_size / (Mb * N_slm * 2 * sizeof_real);
    std::size_t max_Kb = std::min(max_compute_Kb, max_slm_Kb);
    std::size_t Kb = std::min(cfg.shape[2], max_power_of_2_less_equal(max_Kb));

    bool inplace_unsupported = is_real && Mb < M;

    auto istride = std::array<std::size_t, 3>{cfg.istride[0], cfg.istride[1], cfg.istride[2]};
    auto ostride = std::array<std::size_t, 3>{cfg.ostride[0], cfg.ostride[1], cfg.ostride[2]};

    return {
        static_cast<int>(cfg.dir),   // direction
        M,                           // M
        Mb,                          // Mb
        N,                           // N
        Kb,                          // Kb
        sgs,                         // sgs
        cfg.fp,                      // precision
        cfg.type,                    // transform type
        istride,                     // istride
        ostride,                     // ostride
        inplace_unsupported,         // inplace_unsupported
        cfg.callbacks.load_function, // load_function
        cfg.callbacks.store_function // store_function
    };
}

std::string small_batch_configuration::identifier() const {
    std::ostringstream oss;
    oss << "sbfft_" << (direction < 0 ? 'm' : 'p') << std::abs(direction) << "_M" << M << "_Mb"
        << Mb << "_N" << N << "_Kb" << Kb << "_sgs" << sgs << "_f" << static_cast<int>(fp) * 8
        << '_' << to_string(type) << "_is";
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

void generate_small_batch_fft(std::ostream &os, small_batch_configuration const &cfg,
                              std::string_view name) {
    auto gen = std::unique_ptr<sbfft_gen>{};
    switch (cfg.type) {
    case transform_type::c2c:
        gen = std::make_unique<sbfft_gen_c2c>(cfg.N);
        break;
    case transform_type::r2c:
        if (cfg.N % 2 == 1) {
            gen = std::make_unique<sbfft_gen_r2c_double>(cfg.N);
        } else {
            gen = std::make_unique<sbfft_gen_r2c_half>(cfg.N);
        }
        break;
    case transform_type::c2r:
        if (cfg.N % 2 == 1) {
            gen = std::make_unique<sbfft_gen_c2r_double>(cfg.N);
        } else {
            gen = std::make_unique<sbfft_gen_c2r_half>(cfg.N);
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
