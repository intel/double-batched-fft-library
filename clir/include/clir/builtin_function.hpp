// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BUILTIN_FUNCTION_20220405_HPP
#define BUILTIN_FUNCTION_20220405_HPP

#include "clir/export.hpp"
#include "clir/internal/macro_helper.hpp"

#include <iosfwd>
#include <vector>

// Syntax: (Function name, min #args, max #args)
#define CLIR_STANDARD_BUILTIN_FUNCTION(X)                                                          \
    X(get_work_dim, 0, 0)                                                                          \
    X(get_global_size, 1, 1)                                                                       \
    X(get_global_id, 1, 1)                                                                         \
    X(get_local_size, 1, 1)                                                                        \
    X(get_enqueued_local_size, 1, 1)                                                               \
    X(get_local_id, 1, 1)                                                                          \
    X(get_num_groups, 1, 1)                                                                        \
    X(get_group_id, 1, 1)                                                                          \
    X(get_global_offset, 1, 1)                                                                     \
    X(get_global_linear_id, 0, 0)                                                                  \
    X(get_local_linear_id, 0, 0)                                                                   \
    X(get_sub_group_size, 0, 0)                                                                    \
    X(get_max_sub_group_size, 0, 0)                                                                \
    X(get_num_sub_groups, 0, 0)                                                                    \
    X(get_enqueued_num_sub_groups, 0, 0)                                                           \
    X(get_sub_group_id, 0, 0)                                                                      \
    X(get_sub_group_local_id, 0, 0)                                                                \
    X(acos, 1, 1)                                                                                  \
    X(acosh, 1, 1)                                                                                 \
    X(acospi, 1, 1)                                                                                \
    X(asin, 1, 1)                                                                                  \
    X(asinh, 1, 1)                                                                                 \
    X(asinpi, 1, 1)                                                                                \
    X(atan, 1, 1)                                                                                  \
    X(atan2, 2, 2)                                                                                 \
    X(atanh, 1, 1)                                                                                 \
    X(atanpi, 1, 1)                                                                                \
    X(atan2pi, 2, 2)                                                                               \
    X(cbrt, 1, 1)                                                                                  \
    X(ceil, 1, 1)                                                                                  \
    X(copysign, 2, 2)                                                                              \
    X(cos, 1, 1)                                                                                   \
    X(cosh, 1, 1)                                                                                  \
    X(cospi, 1, 1)                                                                                 \
    X(erfc, 1, 1)                                                                                  \
    X(erf, 1, 1)                                                                                   \
    X(exp, 1, 1)                                                                                   \
    X(exp2, 1, 1)                                                                                  \
    X(exp10, 1, 1)                                                                                 \
    X(expm1, 1, 1)                                                                                 \
    X(fabs, 1, 1)                                                                                  \
    X(fdim, 2, 2)                                                                                  \
    X(floor, 1, 1)                                                                                 \
    X(fma, 3, 3)                                                                                   \
    X(fmax, 2, 2)                                                                                  \
    X(fmin, 2, 2)                                                                                  \
    X(fmod, 2, 2)                                                                                  \
    X(trunc, 1, 1)                                                                                 \
    X(fract, 2, 2)                                                                                 \
    X(frexp, 2, 2)                                                                                 \
    X(hypot, 2, 2)                                                                                 \
    X(ilogb, 1, 1)                                                                                 \
    X(ldexp, 2, 2)                                                                                 \
    X(lgamma, 1, 1)                                                                                \
    X(lgamma_r, 2, 2)                                                                              \
    X(log, 1, 1)                                                                                   \
    X(log2, 1, 1)                                                                                  \
    X(log10, 1, 1)                                                                                 \
    X(log1p, 1, 1)                                                                                 \
    X(logb, 1, 1)                                                                                  \
    X(mad, 3, 3)                                                                                   \
    X(maxmag, 2, 2)                                                                                \
    X(minmag, 2, 2)                                                                                \
    X(modf, 2, 2)                                                                                  \
    X(nan, 1, 1)                                                                                   \
    X(nextafter, 2, 2)                                                                             \
    X(pow, 2, 2)                                                                                   \
    X(pown, 2, 2)                                                                                  \
    X(powr, 2, 2)                                                                                  \
    X(remainder, 2, 2)                                                                             \
    X(remquo, 3, 3)                                                                                \
    X(rint, 1, 1)                                                                                  \
    X(rootn, 2, 2)                                                                                 \
    X(round, 1, 1)                                                                                 \
    X(rsqrt, 1, 1)                                                                                 \
    X(sin, 1, 1)                                                                                   \
    X(sincos, 2, 2)                                                                                \
    X(sinh, 1, 1)                                                                                  \
    X(sinpi, 1, 1)                                                                                 \
    X(sqrt, 1, 1)                                                                                  \
    X(tan, 1, 1)                                                                                   \
    X(tanh, 1, 1)                                                                                  \
    X(tanpi, 1, 1)                                                                                 \
    X(tgamma, 1, 1)                                                                                \
    X(half_cos, 1, 1)                                                                              \
    X(half_divide, 2, 2)                                                                           \
    X(half_exp, 1, 1)                                                                              \
    X(half_exp2, 1, 1)                                                                             \
    X(half_exp10, 1, 1)                                                                            \
    X(half_log, 1, 1)                                                                              \
    X(half_log2, 1, 1)                                                                             \
    X(half_log10, 1, 1)                                                                            \
    X(half_powr, 2, 2)                                                                             \
    X(half_recip, 1, 1)                                                                            \
    X(half_rsqrt, 1, 1)                                                                            \
    X(half_sin, 1, 1)                                                                              \
    X(half_sqrt, 1, 1)                                                                             \
    X(half_tan, 1, 1)                                                                              \
    X(native_cos, 1, 1)                                                                            \
    X(native_divide, 2, 2)                                                                         \
    X(native_exp, 1, 1)                                                                            \
    X(native_exp2, 1, 1)                                                                           \
    X(native_exp10, 1, 1)                                                                          \
    X(native_log, 1, 1)                                                                            \
    X(native_log2, 1, 1)                                                                           \
    X(native_log10, 1, 1)                                                                          \
    X(native_powr, 2, 2)                                                                           \
    X(native_recip, 1, 1)                                                                          \
    X(native_rsqrt, 1, 1)                                                                          \
    X(native_sin, 1, 1)                                                                            \
    X(native_sqrt, 1, 1)                                                                           \
    X(native_tan, 1, 1)                                                                            \
    X(abs, 1, 1)                                                                                   \
    X(abs_diff, 2, 2)                                                                              \
    X(add_sat, 2, 2)                                                                               \
    X(hadd, 2, 2)                                                                                  \
    X(rhadd, 2, 2)                                                                                 \
    X(clamp, 3, 3)                                                                                 \
    X(min, 2, 2)                                                                                   \
    X(clz, 1, 1)                                                                                   \
    X(ctz, 1, 1)                                                                                   \
    X(mad_hi, 3, 3)                                                                                \
    X(mul_hi, 2, 2)                                                                                \
    X(mad_sat, 3, 3)                                                                               \
    X(max, 2, 2)                                                                                   \
    X(rotate, 2, 2)                                                                                \
    X(sub_sat, 2, 2)                                                                               \
    X(upsample, 2, 2)                                                                              \
    X(popcount, 1, 1)                                                                              \
    X(mad24, 3, 3)                                                                                 \
    X(mul24, 2, 2)                                                                                 \
    X(degrees, 1, 1)                                                                               \
    X(mix, 3, 3)                                                                                   \
    X(radians, 1, 1)                                                                               \
    X(step, 2, 2)                                                                                  \
    X(smoothstep, 3, 3)                                                                            \
    X(sign, 1, 1)                                                                                  \
    X(cross, 2, 2)                                                                                 \
    X(dot, 2, 2)                                                                                   \
    X(distance, 2, 2)                                                                              \
    X(length, 1, 1)                                                                                \
    X(normalize, 1, 1)                                                                             \
    X(fast_distance, 2, 2)                                                                         \
    X(fast_length, 1, 1)                                                                           \
    X(fast_normalize, 1, 1)                                                                        \
    X(isequal, 2, 2)                                                                               \
    X(isnotequal, 2, 2)                                                                            \
    X(isgreater, 2, 2)                                                                             \
    X(isgreaterequal, 2, 2)                                                                        \
    X(isless, 2, 2)                                                                                \
    X(islessequal, 2, 2)                                                                           \
    X(islessgreater, 2, 2)                                                                         \
    X(isfinite, 1, 1)                                                                              \
    X(isinf, 1, 1)                                                                                 \
    X(isnan, 1, 1)                                                                                 \
    X(isnormal, 1, 1)                                                                              \
    X(isordered, 2, 2)                                                                             \
    X(isunordered, 2, 2)                                                                           \
    X(signbit, 1, 1)                                                                               \
    X(any, 1, 1)                                                                                   \
    X(all, 1, 1)                                                                                   \
    X(bitselect, 3, 3)                                                                             \
    X(select, 3, 3)                                                                                \
    X(vloadn, 2, 2)                                                                                \
    X(vstoren, 3, 3)                                                                               \
    X(vload_half, 2, 2)                                                                            \
    X(vload_halfn, 2, 2)                                                                           \
    X(vstore_half, 3, 3)                                                                           \
    X(vstore_half_rte, 3, 3)                                                                       \
    X(vstore_half_rtz, 3, 3)                                                                       \
    X(vstore_half_rtp, 3, 3)                                                                       \
    X(vstore_half_rtn, 3, 3)                                                                       \
    X(vstore_halfn, 3, 3)                                                                          \
    X(vstore_halfn_rte, 3, 3)                                                                      \
    X(vstore_halfn_rtz, 3, 3)                                                                      \
    X(vstore_halfn_rtp, 3, 3)                                                                      \
    X(vstore_halfn_rtn, 3, 3)                                                                      \
    X(vloada_halfn, 2, 2)                                                                          \
    X(vstorea_halfn, 3, 3)                                                                         \
    X(vstorea_halfn_rte, 3, 3)                                                                     \
    X(vstorea_halfn_rtz, 3, 3)                                                                     \
    X(vstorea_halfn_rtp, 3, 3)                                                                     \
    X(vstorea_halfn_rtn, 3, 3)                                                                     \
    X(barrier, 1, 1)                                                                               \
    X(work_group_barrier, 1, 2)                                                                    \
    X(sub_group_barrier, 1, 2)                                                                     \
    X(to_global, 1, 1)                                                                             \
    X(to_local, 1, 1)                                                                              \
    X(to_private, 1, 1)                                                                            \
    X(get_fence, 1, 1)                                                                             \
    X(async_work_group_copy, 4, 4)                                                                 \
    X(async_work_group_strided_copy, 5, 5)                                                         \
    X(wait_group_events, 2, 2)                                                                     \
    X(prefetch, 2, 2)                                                                              \
    X(vec_step, 1, 1)                                                                              \
    X(shuffle, 2, 2)                                                                               \
    X(shuffle2, 3, 3)                                                                              \
    X(sub_group_all, 1, 1)                                                                         \
    X(sub_group_any, 1, 1)                                                                         \
    X(sub_group_broadcast, 2, 2)                                                                   \
    X(sub_group_reduce_add, 1, 1)                                                                  \
    X(sub_group_reduce_min, 1, 1)                                                                  \
    X(sub_group_reduce_max, 1, 1)                                                                  \
    X(sub_group_scan_exclusive_add, 1, 1)                                                          \
    X(sub_group_scan_exclusive_min, 1, 1)                                                          \
    X(sub_group_scan_exclusive_max, 1, 1)                                                          \
    X(sub_group_scan_inclusive_add, 1, 1)                                                          \
    X(sub_group_scan_inclusive_min, 1, 1)                                                          \
    X(sub_group_scan_inclusive_max, 1, 1)                                                          \
    X(as_char, 1, 1)                                                                               \
    X(as_char2, 1, 1)                                                                              \
    X(as_char3, 1, 1)                                                                              \
    X(as_char4, 1, 1)                                                                              \
    X(as_char8, 1, 1)                                                                              \
    X(as_char16, 1, 1)                                                                             \
    X(as_uchar, 1, 1)                                                                              \
    X(as_uchar2, 1, 1)                                                                             \
    X(as_uchar3, 1, 1)                                                                             \
    X(as_uchar4, 1, 1)                                                                             \
    X(as_uchar8, 1, 1)                                                                             \
    X(as_uchar16, 1, 1)                                                                            \
    X(as_short, 1, 1)                                                                              \
    X(as_short2, 1, 1)                                                                             \
    X(as_short3, 1, 1)                                                                             \
    X(as_short4, 1, 1)                                                                             \
    X(as_short8, 1, 1)                                                                             \
    X(as_short16, 1, 1)                                                                            \
    X(as_ushort, 1, 1)                                                                             \
    X(as_ushort2, 1, 1)                                                                            \
    X(as_ushort3, 1, 1)                                                                            \
    X(as_ushort4, 1, 1)                                                                            \
    X(as_ushort8, 1, 1)                                                                            \
    X(as_ushort16, 1, 1)                                                                           \
    X(as_int, 1, 1)                                                                                \
    X(as_int2, 1, 1)                                                                               \
    X(as_int3, 1, 1)                                                                               \
    X(as_int4, 1, 1)                                                                               \
    X(as_int8, 1, 1)                                                                               \
    X(as_int16, 1, 1)                                                                              \
    X(as_uint, 1, 1)                                                                               \
    X(as_uint2, 1, 1)                                                                              \
    X(as_uint3, 1, 1)                                                                              \
    X(as_uint4, 1, 1)                                                                              \
    X(as_uint8, 1, 1)                                                                              \
    X(as_uint16, 1, 1)                                                                             \
    X(as_long, 1, 1)                                                                               \
    X(as_long2, 1, 1)                                                                              \
    X(as_long3, 1, 1)                                                                              \
    X(as_long4, 1, 1)                                                                              \
    X(as_long8, 1, 1)                                                                              \
    X(as_long16, 1, 1)                                                                             \
    X(as_ulong, 1, 1)                                                                              \
    X(as_ulong2, 1, 1)                                                                             \
    X(as_ulong3, 1, 1)                                                                             \
    X(as_ulong4, 1, 1)                                                                             \
    X(as_ulong8, 1, 1)                                                                             \
    X(as_ulong16, 1, 1)                                                                            \
    X(as_float, 1, 1)                                                                              \
    X(as_float2, 1, 1)                                                                             \
    X(as_float3, 1, 1)                                                                             \
    X(as_float4, 1, 1)                                                                             \
    X(as_float8, 1, 1)                                                                             \
    X(as_float16, 1, 1)                                                                            \
    X(as_double, 1, 1)                                                                             \
    X(as_double2, 1, 1)                                                                            \
    X(as_double3, 1, 1)                                                                            \
    X(as_double4, 1, 1)                                                                            \
    X(as_double8, 1, 1)                                                                            \
    X(as_double16, 1, 1)                                                                           \
    X(printf, 0, inf)

#define CLIR_EXTENSION_INTEL_SUBGROUPS(X)                                                          \
    X(intel_sub_group_shuffle, 2, 2)                                                               \
    X(intel_sub_group_shuffle_down, 3, 3)                                                          \
    X(intel_sub_group_shuffle_up, 3, 3)                                                            \
    X(intel_sub_group_shuffle_xor, 2, 2)                                                           \
    X(intel_sub_group_block_read, 1, 2)                                                            \
    X(intel_sub_group_block_read2, 1, 2)                                                           \
    X(intel_sub_group_block_read4, 1, 2)                                                           \
    X(intel_sub_group_block_read8, 1, 2)                                                           \
    X(intel_sub_group_block_write, 2, 3)                                                           \
    X(intel_sub_group_block_write2, 2, 3)                                                          \
    X(intel_sub_group_block_write4, 2, 3)                                                          \
    X(intel_sub_group_block_write8, 2, 3)

#define CLIR_EXTENSION_INTEL_SUBGROUPS_LONG(X)                                                     \
    X(intel_sub_group_block_read_ul, 1, 2)                                                         \
    X(intel_sub_group_block_read_ul2, 1, 2)                                                        \
    X(intel_sub_group_block_read_ul4, 1, 2)                                                        \
    X(intel_sub_group_block_read_ul8, 1, 2)                                                        \
    X(intel_sub_group_block_write_ul, 2, 3)                                                        \
    X(intel_sub_group_block_write_ul2, 2, 3)                                                       \
    X(intel_sub_group_block_write_ul4, 2, 3)                                                       \
    X(intel_sub_group_block_write_ul8, 2, 3)                                                       \
    X(intel_sub_group_block_read_ui, 1, 2)                                                         \
    X(intel_sub_group_block_read_ui2, 1, 2)                                                        \
    X(intel_sub_group_block_read_ui4, 1, 2)                                                        \
    X(intel_sub_group_block_read_ui8, 1, 2)                                                        \
    X(intel_sub_group_block_write_ui, 2, 3)                                                        \
    X(intel_sub_group_block_write_ui2, 2, 3)                                                       \
    X(intel_sub_group_block_write_ui4, 2, 3)                                                       \
    X(intel_sub_group_block_write_ui8, 2, 3)

#define CLIR_EXTENSION_INTEL_SUBGROUPS_SHORT(X)                                                    \
    X(intel_sub_group_block_read_us, 1, 2)                                                         \
    X(intel_sub_group_block_read_us2, 1, 2)                                                        \
    X(intel_sub_group_block_read_us4, 1, 2)                                                        \
    X(intel_sub_group_block_read_us8, 1, 2)                                                        \
    X(intel_sub_group_block_read_us16, 1, 2)                                                       \
    X(intel_sub_group_block_write_us, 2, 3)                                                        \
    X(intel_sub_group_block_write_us2, 2, 3)                                                       \
    X(intel_sub_group_block_write_us4, 2, 3)                                                       \
    X(intel_sub_group_block_write_us8, 2, 3)                                                       \
    X(intel_sub_group_block_write_us16, 2, 3)

#define CLIR_DECLARE_BUILTIN_FUNCTION_0_0(NAME) CLIR_EXPORT expr NAME();
#define CLIR_DECLARE_BUILTIN_FUNCTION_1_1(NAME) CLIR_EXPORT expr NAME(expr e1);
#define CLIR_DECLARE_BUILTIN_FUNCTION_2_2(NAME) CLIR_EXPORT expr NAME(expr e1, expr e2);
#define CLIR_DECLARE_BUILTIN_FUNCTION_3_3(NAME) CLIR_EXPORT expr NAME(expr e1, expr e2, expr e3);
#define CLIR_DECLARE_BUILTIN_FUNCTION_4_4(NAME)                                                    \
    CLIR_EXPORT expr NAME(expr e1, expr e2, expr e3, expr e4);
#define CLIR_DECLARE_BUILTIN_FUNCTION_5_5(NAME)                                                    \
    CLIR_EXPORT expr NAME(expr e1, expr e2, expr e3, expr e4, expr e5);
#define CLIR_DECLARE_BUILTIN_FUNCTION_0_inf(NAME) CLIR_EXPORT expr NAME(std::vector<expr> args);
#define CLIR_DECLARE_BUILTIN_FUNCTION_1_2(NAME)                                                    \
    CLIR_DECLARE_BUILTIN_FUNCTION_1_1(NAME) CLIR_DECLARE_BUILTIN_FUNCTION_2_2(NAME)
#define CLIR_DECLARE_BUILTIN_FUNCTION_2_3(NAME)                                                    \
    CLIR_DECLARE_BUILTIN_FUNCTION_2_2(NAME) CLIR_DECLARE_BUILTIN_FUNCTION_3_3(NAME)
#define CLIR_DECLARE_BUILTIN_FUNCTION(NAME, A, B) CLIR_DECLARE_BUILTIN_FUNCTION_##A##_##B(NAME)

#define CLIR_BUILTIN_FUNCTION(X)                                                                   \
    CLIR_STANDARD_BUILTIN_FUNCTION(X)                                                              \
    CLIR_EXTENSION_INTEL_SUBGROUPS(X)                                                              \
    CLIR_EXTENSION_INTEL_SUBGROUPS_LONG(X)                                                         \
    CLIR_EXTENSION_INTEL_SUBGROUPS_SHORT(X)

namespace clir {

class CLIR_EXPORT expr;

enum class builtin_function : int { CLIR_BUILTIN_FUNCTION(CLIR_NAME_LIST_3) };
enum class extension { // values must be contiguous and start from 0
    builtin,
    cl_intel_subgroups,
    cl_intel_subgroups_long,
    cl_intel_subgroups_short,
    unknown // must be last
};

CLIR_EXPORT extension get_extension(builtin_function fn);

CLIR_EXPORT char const *to_string(builtin_function fn);
CLIR_EXPORT char const *to_string(extension ext);

CLIR_BUILTIN_FUNCTION(CLIR_DECLARE_BUILTIN_FUNCTION)

} // namespace clir

namespace std {
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, clir::builtin_function fn);
CLIR_EXPORT std::ostream &operator<<(std::ostream &os, clir::extension ext);
} // namespace std

#endif // BUILTIN_FUNCTION_20220405_HPP
