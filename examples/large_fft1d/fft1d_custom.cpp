// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "fft1d_custom.hpp"
#include "bbfft/cl/error.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

static char const kernels[] = R"OpenCL(
kernel
__attribute__((reqd_work_group_size(8,32,1)))
__attribute__((intel_reqd_sub_group_size(8)))
void stage0(global float2* in, global float2* out, constant float2* twiddle, ulong K) {
    local float2 X1[4200];
    size_t kk = get_global_id(2);
    size_t mm = get_global_id(0);
    size_t n_local = get_local_id(1);
    local float2* sub = X1 + (get_local_id(0) + get_local_id(2) * (8u * 525u));
    for (short j1j2 = n_local; j1j2 < 21; j1j2 += 32u) {
        float2 x[25];
        if (mm < 256000u && kk < K) {
            global float2* sub1 = in + (mm + j1j2 * 256000u + kk * 134400000u);
            __attribute__((opencl_unroll_hint(25)))
            for (short j1 = 0; j1 < 25; ++j1) {
                x[j1] = sub1[j1 * (21 * 256000u)];
            }
        }
        constant float2* tw_j1 = twiddle + j1j2 * 25;
        float2 y[5];
        y[0] = x[0] + (float2) (x[5].x, x[5].y) + (float2) (x[10].x, x[10].y) + (float2) (x[15].x, x[15].y) + (float2) (x[20].x, x[20].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p1 = -0x1.9e3779b97f4a8p-1f * (x[15] + x[10]);
        float2 p2 = 0x1.2cf2304755a5ep-1f * (float2) (x[10].y - x[15].y, x[15].x - x[10].x);
        float2 p11 = 0x1.3c6ef372fe95p-2f * (x[20] + x[5]);
        float2 p21 = 0x1.e6f0e134454ffp-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[1] = x[0] + p1 + p2 + p11 + p21;
        y[1] = (float2) (y[1].x, y[1].y);
        float2 p12 = 0x1.3c6ef372fe95p-2f * (x[10] + x[15]);
        float2 p22 = 0x1.e6f0e134454ffp-1f * (float2) (x[15].y - x[10].y, x[10].x - x[15].x);
        float2 p13 = -0x1.9e3779b97f4a8p-1f * (x[20] + x[5]);
        float2 p23 = 0x1.2cf2304755a5ep-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[2] = x[0] + p12 + p22 + p13 + p23;
        y[2] = (float2) (y[2].x, y[2].y);
        y[3] = x[0] + p12 - p22 + p13 - p23;
        y[3] = (float2) (y[3].x, y[3].y);
        y[4] = x[0] + p1 - p2 + p11 - p21;
        y[4] = (float2) (y[4].x, y[4].y);
        x[0] = y[0];
        x[5] = y[1];
        x[10] = y[2];
        x[15] = y[3];
        x[20] = y[4];
        y[0] = x[1] + (float2) (x[6].x, x[6].y) + (float2) (x[11].x, x[11].y) + (float2) (x[16].x, x[16].y) + (float2) (x[21].x, x[21].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p14 = -0x1.9e3779b97f4a8p-1f * (x[16] + x[11]);
        float2 p24 = 0x1.2cf2304755a5ep-1f * (float2) (x[11].y - x[16].y, x[16].x - x[11].x);
        float2 p15 = 0x1.3c6ef372fe95p-2f * (x[21] + x[6]);
        float2 p25 = 0x1.e6f0e134454ffp-1f * (float2) (x[6].y - x[21].y, x[21].x - x[6].x);
        y[1] = x[1] + p14 + p24 + p15 + p25;
        y[1] = (float2) (y[1].x * 0x1.efea21d101eep-1f - y[1].y * -0x1.fd511fa1c0796p-3f, y[1].x * -0x1.fd511fa1c0796p-3f + y[1].y * 0x1.efea21d101eep-1f);
        float2 p16 = 0x1.3c6ef372fe95p-2f * (x[11] + x[16]);
        float2 p26 = 0x1.e6f0e134454ffp-1f * (float2) (x[16].y - x[11].y, x[11].x - x[16].x);
        float2 p17 = -0x1.9e3779b97f4a8p-1f * (x[21] + x[6]);
        float2 p27 = 0x1.2cf2304755a5ep-1f * (float2) (x[6].y - x[21].y, x[21].x - x[6].x);
        y[2] = x[1] + p16 + p26 + p17 + p27;
        y[2] = (float2) (y[2].x * 0x1.c0ab44e81c059p-1f - y[2].y * -0x1.ed50d5cbfa951p-2f, y[2].x * -0x1.ed50d5cbfa951p-2f + y[2].y * 0x1.c0ab44e81c059p-1f);
        y[3] = x[1] + p16 - p26 + p17 - p27;
        y[3] = (float2) (y[3].x * 0x1.753b603d2b816p-1f - y[3].y * -0x1.5e7cf55112014p-1f, y[3].x * -0x1.5e7cf55112014p-1f + y[3].y * 0x1.753b603d2b816p-1f);
        y[4] = x[1] + p14 - p24 + p15 - p25;
        y[4] = (float2) (y[4].x * 0x1.1257e3c182b51p-1f - y[4].y * -0x1.b04bbff642e86p-1f, y[4].x * -0x1.b04bbff642e86p-1f + y[4].y * 0x1.1257e3c182b51p-1f);
        x[1] = y[0];
        x[6] = y[1];
        x[11] = y[2];
        x[16] = y[3];
        x[21] = y[4];
        y[0] = x[2] + (float2) (x[7].x, x[7].y) + (float2) (x[12].x, x[12].y) + (float2) (x[17].x, x[17].y) + (float2) (x[22].x, x[22].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p18 = -0x1.9e3779b97f4a8p-1f * (x[17] + x[12]);
        float2 p28 = 0x1.2cf2304755a5ep-1f * (float2) (x[12].y - x[17].y, x[17].x - x[12].x);
        float2 p19 = 0x1.3c6ef372fe95p-2f * (x[22] + x[7]);
        float2 p29 = 0x1.e6f0e134454ffp-1f * (float2) (x[7].y - x[22].y, x[22].x - x[7].x);
        y[1] = x[2] + p18 + p28 + p19 + p29;
        y[1] = (float2) (y[1].x * 0x1.c0ab44e81c059p-1f - y[1].y * -0x1.ed50d5cbfa951p-2f, y[1].x * -0x1.ed50d5cbfa951p-2f + y[1].y * 0x1.c0ab44e81c059p-1f);
        float2 p110 = 0x1.3c6ef372fe95p-2f * (x[12] + x[17]);
        float2 p210 = 0x1.e6f0e134454ffp-1f * (float2) (x[17].y - x[12].y, x[12].x - x[17].x);
        float2 p111 = -0x1.9e3779b97f4a8p-1f * (x[22] + x[7]);
        float2 p211 = 0x1.2cf2304755a5ep-1f * (float2) (x[7].y - x[22].y, x[22].x - x[7].x);
        y[2] = x[2] + p110 + p210 + p111 + p211;
        y[2] = (float2) (y[2].x * 0x1.1257e3c182b51p-1f - y[2].y * -0x1.b04bbff642e86p-1f, y[2].x * -0x1.b04bbff642e86p-1f + y[2].y * 0x1.1257e3c182b51p-1f);
        y[3] = x[2] + p110 - p210 + p111 - p211;
        y[3] = (float2) (y[3].x * 0x1.0130a1be09379p-4f - y[3].y * -0x1.fefd5bfe443fep-1f, y[3].x * -0x1.fefd5bfe443fep-1f + y[3].y * 0x1.0130a1be09379p-4f);
        y[4] = x[2] + p18 - p28 + p19 - p29;
        y[4] = (float2) (y[4].x * -0x1.b3ff7c925819cp-2f - y[4].y * -0x1.cf457dcdc158cp-1f, y[4].x * -0x1.cf457dcdc158cp-1f + y[4].y * -0x1.b3ff7c925819cp-2f);
        x[2] = y[0];
        x[7] = y[1];
        x[12] = y[2];
        x[17] = y[3];
        x[22] = y[4];
        y[0] = x[3] + (float2) (x[8].x, x[8].y) + (float2) (x[13].x, x[13].y) + (float2) (x[18].x, x[18].y) + (float2) (x[23].x, x[23].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p112 = -0x1.9e3779b97f4a8p-1f * (x[18] + x[13]);
        float2 p212 = 0x1.2cf2304755a5ep-1f * (float2) (x[13].y - x[18].y, x[18].x - x[13].x);
        float2 p113 = 0x1.3c6ef372fe95p-2f * (x[23] + x[8]);
        float2 p213 = 0x1.e6f0e134454ffp-1f * (float2) (x[8].y - x[23].y, x[23].x - x[8].x);
        y[1] = x[3] + p112 + p212 + p113 + p213;
        y[1] = (float2) (y[1].x * 0x1.753b603d2b816p-1f - y[1].y * -0x1.5e7cf55112014p-1f, y[1].x * -0x1.5e7cf55112014p-1f + y[1].y * 0x1.753b603d2b816p-1f);
        float2 p114 = 0x1.3c6ef372fe95p-2f * (x[13] + x[18]);
        float2 p214 = 0x1.e6f0e134454ffp-1f * (float2) (x[18].y - x[13].y, x[13].x - x[18].x);
        float2 p115 = -0x1.9e3779b97f4a8p-1f * (x[23] + x[8]);
        float2 p215 = 0x1.2cf2304755a5ep-1f * (float2) (x[8].y - x[23].y, x[23].x - x[8].x);
        y[2] = x[3] + p114 + p214 + p115 + p215;
        y[2] = (float2) (y[2].x * 0x1.0130a1be09379p-4f - y[2].y * -0x1.fefd5bfe443fep-1f, y[2].x * -0x1.fefd5bfe443fep-1f + y[2].y * 0x1.0130a1be09379p-4f);
        y[3] = x[3] + p114 - p214 + p115 - p215;
        y[3] = (float2) (y[3].x * -0x1.465c6feb501bcp-1f - y[3].y * -0x1.8a80b635b6beap-1f, y[3].x * -0x1.8a80b635b6beap-1f + y[3].y * -0x1.465c6feb501bcp-1f);
        y[4] = x[3] + p112 - p212 + p113 - p213;
        y[4] = (float2) (y[4].x * -0x1.fbf675480d903p-1f - y[4].y * -0x1.00aeb5da15bep-3f, y[4].x * -0x1.00aeb5da15bep-3f + y[4].y * -0x1.fbf675480d903p-1f);
        x[3] = y[0];
        x[8] = y[1];
        x[13] = y[2];
        x[18] = y[3];
        x[23] = y[4];
        y[0] = x[4] + (float2) (x[9].x, x[9].y) + (float2) (x[14].x, x[14].y) + (float2) (x[19].x, x[19].y) + (float2) (x[24].x, x[24].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p116 = -0x1.9e3779b97f4a8p-1f * (x[19] + x[14]);
        float2 p216 = 0x1.2cf2304755a5ep-1f * (float2) (x[14].y - x[19].y, x[19].x - x[14].x);
        float2 p117 = 0x1.3c6ef372fe95p-2f * (x[24] + x[9]);
        float2 p217 = 0x1.e6f0e134454ffp-1f * (float2) (x[9].y - x[24].y, x[24].x - x[9].x);
        y[1] = x[4] + p116 + p216 + p117 + p217;
        y[1] = (float2) (y[1].x * 0x1.1257e3c182b51p-1f - y[1].y * -0x1.b04bbff642e86p-1f, y[1].x * -0x1.b04bbff642e86p-1f + y[1].y * 0x1.1257e3c182b51p-1f);
        float2 p118 = 0x1.3c6ef372fe95p-2f * (x[14] + x[19]);
        float2 p218 = 0x1.e6f0e134454ffp-1f * (float2) (x[19].y - x[14].y, x[14].x - x[19].x);
        float2 p119 = -0x1.9e3779b97f4a8p-1f * (x[24] + x[9]);
        float2 p219 = 0x1.2cf2304755a5ep-1f * (float2) (x[9].y - x[24].y, x[24].x - x[9].x);
        y[2] = x[4] + p118 + p218 + p119 + p219;
        y[2] = (float2) (y[2].x * -0x1.b3ff7c925819cp-2f - y[2].y * -0x1.cf457dcdc158cp-1f, y[2].x * -0x1.cf457dcdc158cp-1f + y[2].y * -0x1.b3ff7c925819cp-2f);
        y[3] = x[4] + p118 - p218 + p119 - p219;
        y[3] = (float2) (y[3].x * -0x1.fbf675480d903p-1f - y[3].y * -0x1.00aeb5da15bep-3f, y[3].x * -0x1.00aeb5da15bep-3f + y[3].y * -0x1.fbf675480d903p-1f);
        y[4] = x[4] + p116 - p216 + p117 - p217;
        y[4] = (float2) (y[4].x * -0x1.465c6feb501bcp-1f - y[4].y * 0x1.8a80b635b6beap-1f, y[4].x * 0x1.8a80b635b6beap-1f + y[4].y * -0x1.465c6feb501bcp-1f);
        x[4] = y[0];
        x[9] = y[1];
        x[14] = y[2];
        x[19] = y[3];
        x[24] = y[4];
        float2 y1[5];
        y1[0] = x[0] + (float2) (x[1].x, x[1].y) + (float2) (x[2].x, x[2].y) + (float2) (x[3].x, x[3].y) + (float2) (x[4].x, x[4].y);
        float2 tw_tmp = (float2) (tw_j1[0].x, tw_j1[0].y);
        y1[0] = (float2) (y1[0].x * tw_tmp.x - y1[0].y * tw_tmp.y, y1[0].x * tw_tmp.y + y1[0].y * tw_tmp.x);
        float2 p120 = -0x1.9e3779b97f4a8p-1f * (x[3] + x[2]);
        float2 p220 = 0x1.2cf2304755a5ep-1f * (float2) (x[2].y - x[3].y, x[3].x - x[2].x);
        float2 p121 = 0x1.3c6ef372fe95p-2f * (x[4] + x[1]);
        float2 p221 = 0x1.e6f0e134454ffp-1f * (float2) (x[1].y - x[4].y, x[4].x - x[1].x);
        y1[1] = x[0] + p120 + p220 + p121 + p221;
        float2 tw_tmp1 = (float2) (tw_j1[5].x, tw_j1[5].y);
        y1[1] = (float2) (y1[1].x * tw_tmp1.x - y1[1].y * tw_tmp1.y, y1[1].x * tw_tmp1.y + y1[1].y * tw_tmp1.x);
        float2 p122 = 0x1.3c6ef372fe95p-2f * (x[2] + x[3]);
        float2 p222 = 0x1.e6f0e134454ffp-1f * (float2) (x[3].y - x[2].y, x[2].x - x[3].x);
        float2 p123 = -0x1.9e3779b97f4a8p-1f * (x[4] + x[1]);
        float2 p223 = 0x1.2cf2304755a5ep-1f * (float2) (x[1].y - x[4].y, x[4].x - x[1].x);
        y1[2] = x[0] + p122 + p222 + p123 + p223;
        float2 tw_tmp2 = (float2) (tw_j1[10].x, tw_j1[10].y);
        y1[2] = (float2) (y1[2].x * tw_tmp2.x - y1[2].y * tw_tmp2.y, y1[2].x * tw_tmp2.y + y1[2].y * tw_tmp2.x);
        y1[3] = x[0] + p122 - p222 + p123 - p223;
        float2 tw_tmp3 = (float2) (tw_j1[15].x, tw_j1[15].y);
        y1[3] = (float2) (y1[3].x * tw_tmp3.x - y1[3].y * tw_tmp3.y, y1[3].x * tw_tmp3.y + y1[3].y * tw_tmp3.x);
        y1[4] = x[0] + p120 - p220 + p121 - p221;
        float2 tw_tmp4 = (float2) (tw_j1[20].x, tw_j1[20].y);
        y1[4] = (float2) (y1[4].x * tw_tmp4.x - y1[4].y * tw_tmp4.y, y1[4].x * tw_tmp4.y + y1[4].y * tw_tmp4.x);
        x[0] = y1[0];
        x[1] = y1[1];
        x[2] = y1[2];
        x[3] = y1[3];
        x[4] = y1[4];
        y1[0] = x[5] + (float2) (x[6].x, x[6].y) + (float2) (x[7].x, x[7].y) + (float2) (x[8].x, x[8].y) + (float2) (x[9].x, x[9].y);
        float2 tw_tmp5 = (float2) (tw_j1[1].x, tw_j1[1].y);
        y1[0] = (float2) (y1[0].x * tw_tmp5.x - y1[0].y * tw_tmp5.y, y1[0].x * tw_tmp5.y + y1[0].y * tw_tmp5.x);
        float2 p124 = -0x1.9e3779b97f4a8p-1f * (x[8] + x[7]);
        float2 p224 = 0x1.2cf2304755a5ep-1f * (float2) (x[7].y - x[8].y, x[8].x - x[7].x);
        float2 p125 = 0x1.3c6ef372fe95p-2f * (x[9] + x[6]);
        float2 p225 = 0x1.e6f0e134454ffp-1f * (float2) (x[6].y - x[9].y, x[9].x - x[6].x);
        y1[1] = x[5] + p124 + p224 + p125 + p225;
        float2 tw_tmp6 = (float2) (tw_j1[6].x, tw_j1[6].y);
        y1[1] = (float2) (y1[1].x * tw_tmp6.x - y1[1].y * tw_tmp6.y, y1[1].x * tw_tmp6.y + y1[1].y * tw_tmp6.x);
        float2 p126 = 0x1.3c6ef372fe95p-2f * (x[7] + x[8]);
        float2 p226 = 0x1.e6f0e134454ffp-1f * (float2) (x[8].y - x[7].y, x[7].x - x[8].x);
        float2 p127 = -0x1.9e3779b97f4a8p-1f * (x[9] + x[6]);
        float2 p227 = 0x1.2cf2304755a5ep-1f * (float2) (x[6].y - x[9].y, x[9].x - x[6].x);
        y1[2] = x[5] + p126 + p226 + p127 + p227;
        float2 tw_tmp7 = (float2) (tw_j1[11].x, tw_j1[11].y);
        y1[2] = (float2) (y1[2].x * tw_tmp7.x - y1[2].y * tw_tmp7.y, y1[2].x * tw_tmp7.y + y1[2].y * tw_tmp7.x);
        y1[3] = x[5] + p126 - p226 + p127 - p227;
        float2 tw_tmp8 = (float2) (tw_j1[16].x, tw_j1[16].y);
        y1[3] = (float2) (y1[3].x * tw_tmp8.x - y1[3].y * tw_tmp8.y, y1[3].x * tw_tmp8.y + y1[3].y * tw_tmp8.x);
        y1[4] = x[5] + p124 - p224 + p125 - p225;
        float2 tw_tmp9 = (float2) (tw_j1[21].x, tw_j1[21].y);
        y1[4] = (float2) (y1[4].x * tw_tmp9.x - y1[4].y * tw_tmp9.y, y1[4].x * tw_tmp9.y + y1[4].y * tw_tmp9.x);
        x[5] = y1[0];
        x[6] = y1[1];
        x[7] = y1[2];
        x[8] = y1[3];
        x[9] = y1[4];
        y1[0] = x[10] + (float2) (x[11].x, x[11].y) + (float2) (x[12].x, x[12].y) + (float2) (x[13].x, x[13].y) + (float2) (x[14].x, x[14].y);
        float2 tw_tmp10 = (float2) (tw_j1[2].x, tw_j1[2].y);
        y1[0] = (float2) (y1[0].x * tw_tmp10.x - y1[0].y * tw_tmp10.y, y1[0].x * tw_tmp10.y + y1[0].y * tw_tmp10.x);
        float2 p128 = -0x1.9e3779b97f4a8p-1f * (x[13] + x[12]);
        float2 p228 = 0x1.2cf2304755a5ep-1f * (float2) (x[12].y - x[13].y, x[13].x - x[12].x);
        float2 p129 = 0x1.3c6ef372fe95p-2f * (x[14] + x[11]);
        float2 p229 = 0x1.e6f0e134454ffp-1f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        y1[1] = x[10] + p128 + p228 + p129 + p229;
        float2 tw_tmp11 = (float2) (tw_j1[7].x, tw_j1[7].y);
        y1[1] = (float2) (y1[1].x * tw_tmp11.x - y1[1].y * tw_tmp11.y, y1[1].x * tw_tmp11.y + y1[1].y * tw_tmp11.x);
        float2 p130 = 0x1.3c6ef372fe95p-2f * (x[12] + x[13]);
        float2 p230 = 0x1.e6f0e134454ffp-1f * (float2) (x[13].y - x[12].y, x[12].x - x[13].x);
        float2 p131 = -0x1.9e3779b97f4a8p-1f * (x[14] + x[11]);
        float2 p231 = 0x1.2cf2304755a5ep-1f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        y1[2] = x[10] + p130 + p230 + p131 + p231;
        float2 tw_tmp12 = (float2) (tw_j1[12].x, tw_j1[12].y);
        y1[2] = (float2) (y1[2].x * tw_tmp12.x - y1[2].y * tw_tmp12.y, y1[2].x * tw_tmp12.y + y1[2].y * tw_tmp12.x);
        y1[3] = x[10] + p130 - p230 + p131 - p231;
        float2 tw_tmp13 = (float2) (tw_j1[17].x, tw_j1[17].y);
        y1[3] = (float2) (y1[3].x * tw_tmp13.x - y1[3].y * tw_tmp13.y, y1[3].x * tw_tmp13.y + y1[3].y * tw_tmp13.x);
        y1[4] = x[10] + p128 - p228 + p129 - p229;
        float2 tw_tmp14 = (float2) (tw_j1[22].x, tw_j1[22].y);
        y1[4] = (float2) (y1[4].x * tw_tmp14.x - y1[4].y * tw_tmp14.y, y1[4].x * tw_tmp14.y + y1[4].y * tw_tmp14.x);
        x[10] = y1[0];
        x[11] = y1[1];
        x[12] = y1[2];
        x[13] = y1[3];
        x[14] = y1[4];
        y1[0] = x[15] + (float2) (x[16].x, x[16].y) + (float2) (x[17].x, x[17].y) + (float2) (x[18].x, x[18].y) + (float2) (x[19].x, x[19].y);
        float2 tw_tmp15 = (float2) (tw_j1[3].x, tw_j1[3].y);
        y1[0] = (float2) (y1[0].x * tw_tmp15.x - y1[0].y * tw_tmp15.y, y1[0].x * tw_tmp15.y + y1[0].y * tw_tmp15.x);
        float2 p132 = -0x1.9e3779b97f4a8p-1f * (x[18] + x[17]);
        float2 p232 = 0x1.2cf2304755a5ep-1f * (float2) (x[17].y - x[18].y, x[18].x - x[17].x);
        float2 p133 = 0x1.3c6ef372fe95p-2f * (x[19] + x[16]);
        float2 p233 = 0x1.e6f0e134454ffp-1f * (float2) (x[16].y - x[19].y, x[19].x - x[16].x);
        y1[1] = x[15] + p132 + p232 + p133 + p233;
        float2 tw_tmp16 = (float2) (tw_j1[8].x, tw_j1[8].y);
        y1[1] = (float2) (y1[1].x * tw_tmp16.x - y1[1].y * tw_tmp16.y, y1[1].x * tw_tmp16.y + y1[1].y * tw_tmp16.x);
        float2 p134 = 0x1.3c6ef372fe95p-2f * (x[17] + x[18]);
        float2 p234 = 0x1.e6f0e134454ffp-1f * (float2) (x[18].y - x[17].y, x[17].x - x[18].x);
        float2 p135 = -0x1.9e3779b97f4a8p-1f * (x[19] + x[16]);
        float2 p235 = 0x1.2cf2304755a5ep-1f * (float2) (x[16].y - x[19].y, x[19].x - x[16].x);
        y1[2] = x[15] + p134 + p234 + p135 + p235;
        float2 tw_tmp17 = (float2) (tw_j1[13].x, tw_j1[13].y);
        y1[2] = (float2) (y1[2].x * tw_tmp17.x - y1[2].y * tw_tmp17.y, y1[2].x * tw_tmp17.y + y1[2].y * tw_tmp17.x);
        y1[3] = x[15] + p134 - p234 + p135 - p235;
        float2 tw_tmp18 = (float2) (tw_j1[18].x, tw_j1[18].y);
        y1[3] = (float2) (y1[3].x * tw_tmp18.x - y1[3].y * tw_tmp18.y, y1[3].x * tw_tmp18.y + y1[3].y * tw_tmp18.x);
        y1[4] = x[15] + p132 - p232 + p133 - p233;
        float2 tw_tmp19 = (float2) (tw_j1[23].x, tw_j1[23].y);
        y1[4] = (float2) (y1[4].x * tw_tmp19.x - y1[4].y * tw_tmp19.y, y1[4].x * tw_tmp19.y + y1[4].y * tw_tmp19.x);
        x[15] = y1[0];
        x[16] = y1[1];
        x[17] = y1[2];
        x[18] = y1[3];
        x[19] = y1[4];
        y1[0] = x[20] + (float2) (x[21].x, x[21].y) + (float2) (x[22].x, x[22].y) + (float2) (x[23].x, x[23].y) + (float2) (x[24].x, x[24].y);
        float2 tw_tmp20 = (float2) (tw_j1[4].x, tw_j1[4].y);
        y1[0] = (float2) (y1[0].x * tw_tmp20.x - y1[0].y * tw_tmp20.y, y1[0].x * tw_tmp20.y + y1[0].y * tw_tmp20.x);
        float2 p136 = -0x1.9e3779b97f4a8p-1f * (x[23] + x[22]);
        float2 p236 = 0x1.2cf2304755a5ep-1f * (float2) (x[22].y - x[23].y, x[23].x - x[22].x);
        float2 p137 = 0x1.3c6ef372fe95p-2f * (x[24] + x[21]);
        float2 p237 = 0x1.e6f0e134454ffp-1f * (float2) (x[21].y - x[24].y, x[24].x - x[21].x);
        y1[1] = x[20] + p136 + p236 + p137 + p237;
        float2 tw_tmp21 = (float2) (tw_j1[9].x, tw_j1[9].y);
        y1[1] = (float2) (y1[1].x * tw_tmp21.x - y1[1].y * tw_tmp21.y, y1[1].x * tw_tmp21.y + y1[1].y * tw_tmp21.x);
        float2 p138 = 0x1.3c6ef372fe95p-2f * (x[22] + x[23]);
        float2 p238 = 0x1.e6f0e134454ffp-1f * (float2) (x[23].y - x[22].y, x[22].x - x[23].x);
        float2 p139 = -0x1.9e3779b97f4a8p-1f * (x[24] + x[21]);
        float2 p239 = 0x1.2cf2304755a5ep-1f * (float2) (x[21].y - x[24].y, x[24].x - x[21].x);
        y1[2] = x[20] + p138 + p238 + p139 + p239;
        float2 tw_tmp22 = (float2) (tw_j1[14].x, tw_j1[14].y);
        y1[2] = (float2) (y1[2].x * tw_tmp22.x - y1[2].y * tw_tmp22.y, y1[2].x * tw_tmp22.y + y1[2].y * tw_tmp22.x);
        y1[3] = x[20] + p138 - p238 + p139 - p239;
        float2 tw_tmp23 = (float2) (tw_j1[19].x, tw_j1[19].y);
        y1[3] = (float2) (y1[3].x * tw_tmp23.x - y1[3].y * tw_tmp23.y, y1[3].x * tw_tmp23.y + y1[3].y * tw_tmp23.x);
        y1[4] = x[20] + p136 - p236 + p137 - p237;
        float2 tw_tmp24 = (float2) (tw_j1[24].x, tw_j1[24].y);
        y1[4] = (float2) (y1[4].x * tw_tmp24.x - y1[4].y * tw_tmp24.y, y1[4].x * tw_tmp24.y + y1[4].y * tw_tmp24.x);
        x[20] = y1[0];
        x[21] = y1[1];
        x[22] = y1[2];
        x[23] = y1[3];
        x[24] = y1[4];
        local float2* sub2 = sub + j1j2 * 8u;
        sub2[0u] = x[0u];
        sub2[21 * 8u] = x[5u];
        sub2[2u * (21 * 8u)] = x[10u];
        sub2[3u * (21 * 8u)] = x[15u];
        sub2[4u * (21 * 8u)] = x[20u];
        sub2[5u * (21 * 8u)] = x[1];
        sub2[6u * (21 * 8u)] = x[6u];
        sub2[7u * (21 * 8u)] = x[11u];
        sub2[8u * (21 * 8u)] = x[16u];
        sub2[9u * (21 * 8u)] = x[21u];
        sub2[10u * (21 * 8u)] = x[2u];
        sub2[11u * (21 * 8u)] = x[7u];
        sub2[12u * (21 * 8u)] = x[12u];
        sub2[13u * (21 * 8u)] = x[17u];
        sub2[14u * (21 * 8u)] = x[22u];
        sub2[15u * (21 * 8u)] = x[3u];
        sub2[16u * (21 * 8u)] = x[8u];
        sub2[17u * (21 * 8u)] = x[13u];
        sub2[18u * (21 * 8u)] = x[18u];
        sub2[19u * (21 * 8u)] = x[23u];
        sub2[20u * (21 * 8u)] = x[4u];
        sub2[21u * (21 * 8u)] = x[9u];
        sub2[22u * (21 * 8u)] = x[14u];
        sub2[23u * (21 * 8u)] = x[19u];
        sub2[24u * (21 * 8u)] = x[24u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (short j1j2 = n_local; j1j2 < 25; j1j2 += 32u) {
        float2 x[21];
        local float2* sub3 = sub + j1j2 * (21 * 8u);
        __attribute__((opencl_unroll_hint(21)))
        for (short j1 = 0; j1 < 21; ++j1) {
            x[j1] = sub3[j1 * 8u];
        }
        float2 y[7];
        y[0] = x[0] + (float2) (x[3].x, x[3].y) + (float2) (x[6].x, x[6].y) + (float2) (x[9].x, x[9].y) + (float2) (x[12].x, x[12].y) + (float2) (x[15].x, x[15].y) + (float2) (x[18].x, x[18].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p1 = -0x1.cd4bca9cb5c71p-1f * (x[12] + x[9]);
        float2 p2 = 0x1.bc4c04d71abc1p-2f * (float2) (x[9].y - x[12].y, x[12].x - x[9].x);
        float2 p11 = -0x1.c7b90e3024582p-3f * (x[15] + x[6]);
        float2 p21 = 0x1.f329c0558e969p-1f * (float2) (x[6].y - x[15].y, x[15].x - x[6].x);
        float2 p12 = 0x1.3f3a0e28bedd1p-1f * (x[18] + x[3]);
        float2 p22 = 0x1.904c37505de4bp-1f * (float2) (x[3].y - x[18].y, x[18].x - x[3].x);
        y[1] = x[0] + p1 + p2 + p11 + p21 + p12 + p22;
        y[1] = (float2) (y[1].x, y[1].y);
        float2 p13 = 0x1.3f3a0e28bedd1p-1f * (x[9] + x[12]);
        float2 p23 = 0x1.904c37505de4bp-1f * (float2) (x[12].y - x[9].y, x[9].x - x[12].x);
        float2 p14 = -0x1.cd4bca9cb5c71p-1f * (x[6] + x[15]);
        float2 p24 = 0x1.bc4c04d71abc1p-2f * (float2) (x[15].y - x[6].y, x[6].x - x[15].x);
        float2 p15 = -0x1.c7b90e3024582p-3f * (x[18] + x[3]);
        float2 p25 = 0x1.f329c0558e969p-1f * (float2) (x[3].y - x[18].y, x[18].x - x[3].x);
        y[2] = x[0] + p13 + p23 + p14 + p24 + p15 + p25;
        y[2] = (float2) (y[2].x, y[2].y);
        float2 p16 = -0x1.c7b90e3024582p-3f * (x[12] + x[9]);
        float2 p26 = 0x1.f329c0558e969p-1f * (float2) (x[9].y - x[12].y, x[12].x - x[9].x);
        float2 p17 = 0x1.3f3a0e28bedd1p-1f * (x[6] + x[15]);
        float2 p27 = 0x1.904c37505de4bp-1f * (float2) (x[15].y - x[6].y, x[6].x - x[15].x);
        float2 p18 = -0x1.cd4bca9cb5c71p-1f * (x[18] + x[3]);
        float2 p28 = 0x1.bc4c04d71abc1p-2f * (float2) (x[3].y - x[18].y, x[18].x - x[3].x);
        y[3] = x[0] + p16 + p26 + p17 + p27 + p18 + p28;
        y[3] = (float2) (y[3].x, y[3].y);
        y[4] = x[0] + p16 - p26 + p17 - p27 + p18 - p28;
        y[4] = (float2) (y[4].x, y[4].y);
        y[5] = x[0] + p13 - p23 + p14 - p24 + p15 - p25;
        y[5] = (float2) (y[5].x, y[5].y);
        y[6] = x[0] + p1 - p2 + p11 - p21 + p12 - p22;
        y[6] = (float2) (y[6].x, y[6].y);
        x[0] = y[0];
        x[3] = y[1];
        x[6] = y[2];
        x[9] = y[3];
        x[12] = y[4];
        x[15] = y[5];
        x[18] = y[6];
        y[0] = x[1] + (float2) (x[4].x, x[4].y) + (float2) (x[7].x, x[7].y) + (float2) (x[10].x, x[10].y) + (float2) (x[13].x, x[13].y) + (float2) (x[16].x, x[16].y) + (float2) (x[19].x, x[19].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p19 = -0x1.cd4bca9cb5c71p-1f * (x[13] + x[10]);
        float2 p29 = 0x1.bc4c04d71abc1p-2f * (float2) (x[10].y - x[13].y, x[13].x - x[10].x);
        float2 p110 = -0x1.c7b90e3024582p-3f * (x[16] + x[7]);
        float2 p210 = 0x1.f329c0558e969p-1f * (float2) (x[7].y - x[16].y, x[16].x - x[7].x);
        float2 p111 = 0x1.3f3a0e28bedd1p-1f * (x[19] + x[4]);
        float2 p211 = 0x1.904c37505de4bp-1f * (float2) (x[4].y - x[19].y, x[19].x - x[4].x);
        y[1] = x[1] + p19 + p29 + p110 + p210 + p111 + p211;
        y[1] = (float2) (y[1].x * 0x1.e940d6bb98cc5p-1f - y[1].y * -0x1.2dd44ce9afba7p-2f, y[1].x * -0x1.2dd44ce9afba7p-2f + y[1].y * 0x1.e940d6bb98cc5p-1f);
        float2 p112 = 0x1.3f3a0e28bedd1p-1f * (x[10] + x[13]);
        float2 p212 = 0x1.904c37505de4bp-1f * (float2) (x[13].y - x[10].y, x[10].x - x[13].x);
        float2 p113 = -0x1.cd4bca9cb5c71p-1f * (x[7] + x[16]);
        float2 p213 = 0x1.bc4c04d71abc1p-2f * (float2) (x[16].y - x[7].y, x[7].x - x[16].x);
        float2 p114 = -0x1.c7b90e3024582p-3f * (x[19] + x[4]);
        float2 p214 = 0x1.f329c0558e969p-1f * (float2) (x[4].y - x[19].y, x[19].x - x[4].x);
        y[2] = x[1] + p112 + p212 + p113 + p213 + p114 + p214;
        y[2] = (float2) (y[2].x * 0x1.a708c4c4bfa74p-1f - y[2].y * -0x1.206b7c9520cedp-1f, y[2].x * -0x1.206b7c9520cedp-1f + y[2].y * 0x1.a708c4c4bfa74p-1f);
        float2 p115 = -0x1.c7b90e3024582p-3f * (x[13] + x[10]);
        float2 p215 = 0x1.f329c0558e969p-1f * (float2) (x[10].y - x[13].y, x[13].x - x[10].x);
        float2 p116 = 0x1.3f3a0e28bedd1p-1f * (x[7] + x[16]);
        float2 p216 = 0x1.904c37505de4bp-1f * (float2) (x[16].y - x[7].y, x[7].x - x[16].x);
        float2 p117 = -0x1.cd4bca9cb5c71p-1f * (x[19] + x[4]);
        float2 p217 = 0x1.bc4c04d71abc1p-2f * (float2) (x[4].y - x[19].y, x[19].x - x[4].x);
        y[3] = x[1] + p115 + p215 + p116 + p216 + p117 + p217;
        y[3] = (float2) (y[3].x * 0x1.3f3a0e28bedd1p-1f - y[3].y * -0x1.904c37505de4bp-1f, y[3].x * -0x1.904c37505de4bp-1f + y[3].y * 0x1.3f3a0e28bedd1p-1f);
        y[4] = x[1] + p115 - p215 + p116 - p216 + p117 - p217;
        y[4] = (float2) (y[4].x * 0x1.761bf51e29c9p-2f - y[4].y * -0x1.dc9b7be64378ep-1f, y[4].x * -0x1.dc9b7be64378ep-1f + y[4].y * 0x1.761bf51e29c9p-2f);
        y[5] = x[1] + p112 - p212 + p113 - p213 + p114 - p214;
        y[5] = (float2) (y[5].x * 0x1.32182ebfb0fe9p-4f - y[5].y * -0x1.fe917f00ae2cdp-1f, y[5].x * -0x1.fe917f00ae2cdp-1f + y[5].y * 0x1.32182ebfb0fe9p-4f);
        y[6] = x[1] + p19 - p29 + p110 - p210 + p111 - p211;
        y[6] = (float2) (y[6].x * -0x1.c7b90e3024582p-3f - y[6].y * -0x1.f329c0558e969p-1f, y[6].x * -0x1.f329c0558e969p-1f + y[6].y * -0x1.c7b90e3024582p-3f);
        x[1] = y[0];
        x[4] = y[1];
        x[7] = y[2];
        x[10] = y[3];
        x[13] = y[4];
        x[16] = y[5];
        x[19] = y[6];
        y[0] = x[2] + (float2) (x[5].x, x[5].y) + (float2) (x[8].x, x[8].y) + (float2) (x[11].x, x[11].y) + (float2) (x[14].x, x[14].y) + (float2) (x[17].x, x[17].y) + (float2) (x[20].x, x[20].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p118 = -0x1.cd4bca9cb5c71p-1f * (x[14] + x[11]);
        float2 p218 = 0x1.bc4c04d71abc1p-2f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        float2 p119 = -0x1.c7b90e3024582p-3f * (x[17] + x[8]);
        float2 p219 = 0x1.f329c0558e969p-1f * (float2) (x[8].y - x[17].y, x[17].x - x[8].x);
        float2 p120 = 0x1.3f3a0e28bedd1p-1f * (x[20] + x[5]);
        float2 p220 = 0x1.904c37505de4bp-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[1] = x[2] + p118 + p218 + p119 + p219 + p120 + p220;
        y[1] = (float2) (y[1].x * 0x1.a708c4c4bfa74p-1f - y[1].y * -0x1.206b7c9520cedp-1f, y[1].x * -0x1.206b7c9520cedp-1f + y[1].y * 0x1.a708c4c4bfa74p-1f);
        float2 p121 = 0x1.3f3a0e28bedd1p-1f * (x[11] + x[14]);
        float2 p221 = 0x1.904c37505de4bp-1f * (float2) (x[14].y - x[11].y, x[11].x - x[14].x);
        float2 p122 = -0x1.cd4bca9cb5c71p-1f * (x[8] + x[17]);
        float2 p222 = 0x1.bc4c04d71abc1p-2f * (float2) (x[17].y - x[8].y, x[8].x - x[17].x);
        float2 p123 = -0x1.c7b90e3024582p-3f * (x[20] + x[5]);
        float2 p223 = 0x1.f329c0558e969p-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[2] = x[2] + p121 + p221 + p122 + p222 + p123 + p223;
        y[2] = (float2) (y[2].x * 0x1.761bf51e29c9p-2f - y[2].y * -0x1.dc9b7be64378ep-1f, y[2].x * -0x1.dc9b7be64378ep-1f + y[2].y * 0x1.761bf51e29c9p-2f);
        float2 p124 = -0x1.c7b90e3024582p-3f * (x[14] + x[11]);
        float2 p224 = 0x1.f329c0558e969p-1f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        float2 p125 = 0x1.3f3a0e28bedd1p-1f * (x[8] + x[17]);
        float2 p225 = 0x1.904c37505de4bp-1f * (float2) (x[17].y - x[8].y, x[8].x - x[17].x);
        float2 p126 = -0x1.cd4bca9cb5c71p-1f * (x[20] + x[5]);
        float2 p226 = 0x1.bc4c04d71abc1p-2f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[3] = x[2] + p124 + p224 + p125 + p225 + p126 + p226;
        y[3] = (float2) (y[3].x * -0x1.c7b90e3024582p-3f - y[3].y * -0x1.f329c0558e969p-1f, y[3].x * -0x1.f329c0558e969p-1f + y[3].y * -0x1.c7b90e3024582p-3f);
        y[4] = x[2] + p124 - p224 + p125 - p225 + p126 - p226;
        y[4] = (float2) (y[4].x * -0x1.7752932f8fb65p-1f - y[4].y * -0x1.5c3f99e0b6b95p-1f, y[4].x * -0x1.5c3f99e0b6b95p-1f + y[4].y * -0x1.7752932f8fb65p-1f);
        y[5] = x[2] + p121 - p221 + p122 - p222 + p123 - p223;
        y[5] = (float2) (y[5].x * -0x1.fa4808b7d3c19p-1f - y[5].y * -0x1.313d12579650cp-3f, y[5].x * -0x1.313d12579650cp-3f + y[5].y * -0x1.fa4808b7d3c19p-1f);
        y[6] = x[2] + p118 - p218 + p119 - p219 + p120 - p220;
        y[6] = (float2) (y[6].x * -0x1.cd4bca9cb5c71p-1f - y[6].y * 0x1.bc4c04d71abc1p-2f, y[6].x * 0x1.bc4c04d71abc1p-2f + y[6].y * -0x1.cd4bca9cb5c71p-1f);
        x[2] = y[0];
        x[5] = y[1];
        x[8] = y[2];
        x[11] = y[3];
        x[14] = y[4];
        x[17] = y[5];
        x[20] = y[6];
        float2 y1[3];
        y1[0] = x[0] + (float2) (x[1].x, x[1].y) + (float2) (x[2].x, x[2].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p127 = -0x1p-1f * (x[2] + x[1]);
        float2 p227 = 0x1.bb67ae8584caap-1f * (float2) (x[1].y - x[2].y, x[2].x - x[1].x);
        y1[1] = x[0] + p127 + p227;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[0] + p127 - p227;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[0] = y1[0];
        x[1] = y1[1];
        x[2] = y1[2];
        y1[0] = x[3] + (float2) (x[4].x, x[4].y) + (float2) (x[5].x, x[5].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p128 = -0x1p-1f * (x[5] + x[4]);
        float2 p228 = 0x1.bb67ae8584caap-1f * (float2) (x[4].y - x[5].y, x[5].x - x[4].x);
        y1[1] = x[3] + p128 + p228;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[3] + p128 - p228;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[3] = y1[0];
        x[4] = y1[1];
        x[5] = y1[2];
        y1[0] = x[6] + (float2) (x[7].x, x[7].y) + (float2) (x[8].x, x[8].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p129 = -0x1p-1f * (x[8] + x[7]);
        float2 p229 = 0x1.bb67ae8584caap-1f * (float2) (x[7].y - x[8].y, x[8].x - x[7].x);
        y1[1] = x[6] + p129 + p229;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[6] + p129 - p229;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[6] = y1[0];
        x[7] = y1[1];
        x[8] = y1[2];
        y1[0] = x[9] + (float2) (x[10].x, x[10].y) + (float2) (x[11].x, x[11].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p130 = -0x1p-1f * (x[11] + x[10]);
        float2 p230 = 0x1.bb67ae8584caap-1f * (float2) (x[10].y - x[11].y, x[11].x - x[10].x);
        y1[1] = x[9] + p130 + p230;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[9] + p130 - p230;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[9] = y1[0];
        x[10] = y1[1];
        x[11] = y1[2];
        y1[0] = x[12] + (float2) (x[13].x, x[13].y) + (float2) (x[14].x, x[14].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p131 = -0x1p-1f * (x[14] + x[13]);
        float2 p231 = 0x1.bb67ae8584caap-1f * (float2) (x[13].y - x[14].y, x[14].x - x[13].x);
        y1[1] = x[12] + p131 + p231;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[12] + p131 - p231;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[12] = y1[0];
        x[13] = y1[1];
        x[14] = y1[2];
        y1[0] = x[15] + (float2) (x[16].x, x[16].y) + (float2) (x[17].x, x[17].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p132 = -0x1p-1f * (x[17] + x[16]);
        float2 p232 = 0x1.bb67ae8584caap-1f * (float2) (x[16].y - x[17].y, x[17].x - x[16].x);
        y1[1] = x[15] + p132 + p232;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[15] + p132 - p232;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[15] = y1[0];
        x[16] = y1[1];
        x[17] = y1[2];
        y1[0] = x[18] + (float2) (x[19].x, x[19].y) + (float2) (x[20].x, x[20].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        float2 p133 = -0x1p-1f * (x[20] + x[19]);
        float2 p233 = 0x1.bb67ae8584caap-1f * (float2) (x[19].y - x[20].y, x[20].x - x[19].x);
        y1[1] = x[18] + p133 + p233;
        y1[1] = (float2) (y1[1].x, y1[1].y);
        y1[2] = x[18] + p133 - p233;
        y1[2] = (float2) (y1[2].x, y1[2].y);
        x[18] = y1[0];
        x[19] = y1[1];
        x[20] = y1[2];
        local float2* sub4 = sub + j1j2 * (21 * 8u);
        sub4[0u] = x[0u];
        sub4[8u] = x[3u];
        sub4[2u * 8u] = x[6u];
        sub4[3u * 8u] = x[9u];
        sub4[4u * 8u] = x[12u];
        sub4[5u * 8u] = x[15u];
        sub4[6u * 8u] = x[18u];
        sub4[7u * 8u] = x[1];
        sub4[8u * 8u] = x[4u];
        sub4[9u * 8u] = x[7u];
        sub4[10u * 8u] = x[10u];
        sub4[11u * 8u] = x[13u];
        sub4[12u * 8u] = x[16u];
        sub4[13u * 8u] = x[19u];
        sub4[14u * 8u] = x[2u];
        sub4[15u * 8u] = x[5u];
        sub4[16u * 8u] = x[8u];
        sub4[17u * 8u] = x[11u];
        sub4[18u * 8u] = x[14u];
        sub4[19u * 8u] = x[17u];
        sub4[20u * 8u] = x[20u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /*for (short j2 = n_local; j2 < 25; j2 += 32u) {
        local float2* sub5 = sub + j2 * (21 * 8u);
        if (mm < 256000u && kk < K) {
            global float2* sub6 = out + (mm + j2 * 256000u + kk * 134400000u);
            __attribute__((opencl_unroll_hint(2)))
            for (short j1 = 0; j1 < 21; ++j1) {
                sub6[j1 * (25 * 256000u)] = sub5[j1 * 8u];
            }
        }
    }*/
    global float2* sub6 = out + kk * 134400000u;
    for (short n2 = get_local_linear_id(); n2 < 525; n2 += 256) {
        short f1 = n2 % 25;
        short f2 = n2 / 25 % 21;
        local float2* sub5 = X1 + (f2 + f1 * 21) * 8u;
        for (short i = 0; i < 8; ++i) {
            size_t n01 = i + get_group_id(0) * 8;
            if (n01 < 256000u) {
                size_t n1 = n01 / 500 % 512;
                float arg = -((float) 6.28318530717958647693) / (500 * 512) * n2 * n1;
                float2 tw = (float2) (native_cos(arg), native_sin(arg));
                float2 val = sub5[i];
                sub6[n2 + n01 * 525] = (float2) (val.x * tw.x - val.y * tw.y, val.x * tw.y + val.y * tw.x) / 525;
            }
        }
    }
}

kernel
__attribute__((reqd_work_group_size(8,64,1)))
__attribute__((intel_reqd_sub_group_size(8)))
void stage1(global float2* in, global float2* out, constant float2* twiddle, ulong K0) {
    local float2 X1[4096];
    size_t k0 = get_global_id(2);
    size_t k2 = get_global_id(0);
    size_t n_local = get_local_id(1);
    local float2* sub = X1 + get_local_id(0);
    {
        float2 x[8];
        if (k2 < 525 && k0 < K0) {
            global float2* sub1 = in + (k2 + k0 * 525 + n_local * 262500u);
            __attribute__((opencl_unroll_hint(8)))
            for (short j1 = 0; j1 < 8; ++j1) {
                x[j1] = sub1[j1 * (64 * 262500u)];
            }
        }
        constant float2* tw_j1 = twiddle + n_local * 8;
        float2 y[2];
        y[0] = x[0] + (float2) (x[4].x, x[4].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[0] + (float2) (x[4].x * -0x1p+0f, x[4].y * -0x1p+0f);
        y[1] = (float2) (y[1].x, y[1].y);
        x[0] = y[0];
        x[4] = y[1];
        y[0] = x[1] + (float2) (x[5].x, x[5].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[1] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * 0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * 0x1.6a09e667f3bcdp-1f);
        x[1] = y[0];
        x[5] = y[1];
        y[0] = x[2] + (float2) (x[6].x, x[6].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[2] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y[1] = (float2) (-(y[1].y * -0x1p+0f), y[1].x * -0x1p+0f);
        x[2] = y[0];
        x[6] = y[1];
        y[0] = x[3] + (float2) (x[7].x, x[7].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[3] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * -0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * -0x1.6a09e667f3bcdp-1f);
        x[3] = y[0];
        x[7] = y[1];
        float2 y1[2];
        y1[0] = x[0] + (float2) (x[2].x, x[2].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[0] + (float2) (x[2].x * -0x1p+0f, x[2].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[0] = y1[0];
        x[2] = y1[1];
        y1[0] = x[4] + (float2) (x[6].x, x[6].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[4] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[4] = y1[0];
        x[6] = y1[1];
        y1[0] = x[1] + (float2) (x[3].x, x[3].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[1] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[1] = y1[0];
        x[3] = y1[1];
        y1[0] = x[5] + (float2) (x[7].x, x[7].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[5] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[5] = y1[0];
        x[7] = y1[1];
        float2 y2[2];
        y2[0] = x[0] + (float2) (x[1].x, x[1].y);
        float2 tw_tmp = (float2) (tw_j1[0].x, tw_j1[0].y);
        y2[0] = (float2) (y2[0].x * tw_tmp.x - y2[0].y * tw_tmp.y, y2[0].x * tw_tmp.y + y2[0].y * tw_tmp.x);
        y2[1] = x[0] + (float2) (x[1].x * -0x1p+0f, x[1].y * -0x1p+0f);
        float2 tw_tmp1 = (float2) (tw_j1[4].x, tw_j1[4].y);
        y2[1] = (float2) (y2[1].x * tw_tmp1.x - y2[1].y * tw_tmp1.y, y2[1].x * tw_tmp1.y + y2[1].y * tw_tmp1.x);
        x[0] = y2[0];
        x[1] = y2[1];
        y2[0] = x[2] + (float2) (x[3].x, x[3].y);
        float2 tw_tmp2 = (float2) (tw_j1[2].x, tw_j1[2].y);
        y2[0] = (float2) (y2[0].x * tw_tmp2.x - y2[0].y * tw_tmp2.y, y2[0].x * tw_tmp2.y + y2[0].y * tw_tmp2.x);
        y2[1] = x[2] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        float2 tw_tmp3 = (float2) (tw_j1[6].x, tw_j1[6].y);
        y2[1] = (float2) (y2[1].x * tw_tmp3.x - y2[1].y * tw_tmp3.y, y2[1].x * tw_tmp3.y + y2[1].y * tw_tmp3.x);
        x[2] = y2[0];
        x[3] = y2[1];
        y2[0] = x[4] + (float2) (x[5].x, x[5].y);
        float2 tw_tmp4 = (float2) (tw_j1[1].x, tw_j1[1].y);
        y2[0] = (float2) (y2[0].x * tw_tmp4.x - y2[0].y * tw_tmp4.y, y2[0].x * tw_tmp4.y + y2[0].y * tw_tmp4.x);
        y2[1] = x[4] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        float2 tw_tmp5 = (float2) (tw_j1[5].x, tw_j1[5].y);
        y2[1] = (float2) (y2[1].x * tw_tmp5.x - y2[1].y * tw_tmp5.y, y2[1].x * tw_tmp5.y + y2[1].y * tw_tmp5.x);
        x[4] = y2[0];
        x[5] = y2[1];
        y2[0] = x[6] + (float2) (x[7].x, x[7].y);
        float2 tw_tmp6 = (float2) (tw_j1[3].x, tw_j1[3].y);
        y2[0] = (float2) (y2[0].x * tw_tmp6.x - y2[0].y * tw_tmp6.y, y2[0].x * tw_tmp6.y + y2[0].y * tw_tmp6.x);
        y2[1] = x[6] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        float2 tw_tmp7 = (float2) (tw_j1[7].x, tw_j1[7].y);
        y2[1] = (float2) (y2[1].x * tw_tmp7.x - y2[1].y * tw_tmp7.y, y2[1].x * tw_tmp7.y + y2[1].y * tw_tmp7.x);
        x[6] = y2[0];
        x[7] = y2[1];
        local float2* sub2 = sub + n_local * 8u;
        sub2[0u] = x[0u];
        sub2[64 * 8u] = x[4u];
        sub2[2u * (64 * 8u)] = x[2u];
        sub2[3u * (64 * 8u)] = x[6u];
        sub2[4u * (64 * 8u)] = x[1];
        sub2[5u * (64 * 8u)] = x[5u];
        sub2[6u * (64 * 8u)] = x[3u];
        sub2[7u * (64 * 8u)] = x[7u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        short j1 = n_local % 8;
        short j2 = n_local / 8;
        float2 x[8];
        local float2* sub3 = sub + (j1 * 8u + j2 * (8 * (8 * 8u)));
        __attribute__((opencl_unroll_hint(8)))
        for (short j11 = 0; j11 < 8; ++j11) {
            x[j11] = sub3[j11 * (8 * 8u)];
        }
        constant float2* tw_j1 = twiddle + 512 + j1 * 8;
        float2 y[2];
        y[0] = x[0] + (float2) (x[4].x, x[4].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[0] + (float2) (x[4].x * -0x1p+0f, x[4].y * -0x1p+0f);
        y[1] = (float2) (y[1].x, y[1].y);
        x[0] = y[0];
        x[4] = y[1];
        y[0] = x[1] + (float2) (x[5].x, x[5].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[1] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * 0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * 0x1.6a09e667f3bcdp-1f);
        x[1] = y[0];
        x[5] = y[1];
        y[0] = x[2] + (float2) (x[6].x, x[6].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[2] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y[1] = (float2) (-(y[1].y * -0x1p+0f), y[1].x * -0x1p+0f);
        x[2] = y[0];
        x[6] = y[1];
        y[0] = x[3] + (float2) (x[7].x, x[7].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[3] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * -0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * -0x1.6a09e667f3bcdp-1f);
        x[3] = y[0];
        x[7] = y[1];
        float2 y1[2];
        y1[0] = x[0] + (float2) (x[2].x, x[2].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[0] + (float2) (x[2].x * -0x1p+0f, x[2].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[0] = y1[0];
        x[2] = y1[1];
        y1[0] = x[4] + (float2) (x[6].x, x[6].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[4] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[4] = y1[0];
        x[6] = y1[1];
        y1[0] = x[1] + (float2) (x[3].x, x[3].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[1] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[1] = y1[0];
        x[3] = y1[1];
        y1[0] = x[5] + (float2) (x[7].x, x[7].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[5] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[5] = y1[0];
        x[7] = y1[1];
        float2 y2[2];
        y2[0] = x[0] + (float2) (x[1].x, x[1].y);
        float2 tw_tmp = (float2) (tw_j1[0].x, tw_j1[0].y);
        y2[0] = (float2) (y2[0].x * tw_tmp.x - y2[0].y * tw_tmp.y, y2[0].x * tw_tmp.y + y2[0].y * tw_tmp.x);
        y2[1] = x[0] + (float2) (x[1].x * -0x1p+0f, x[1].y * -0x1p+0f);
        float2 tw_tmp1 = (float2) (tw_j1[4].x, tw_j1[4].y);
        y2[1] = (float2) (y2[1].x * tw_tmp1.x - y2[1].y * tw_tmp1.y, y2[1].x * tw_tmp1.y + y2[1].y * tw_tmp1.x);
        x[0] = y2[0];
        x[1] = y2[1];
        y2[0] = x[2] + (float2) (x[3].x, x[3].y);
        float2 tw_tmp2 = (float2) (tw_j1[2].x, tw_j1[2].y);
        y2[0] = (float2) (y2[0].x * tw_tmp2.x - y2[0].y * tw_tmp2.y, y2[0].x * tw_tmp2.y + y2[0].y * tw_tmp2.x);
        y2[1] = x[2] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        float2 tw_tmp3 = (float2) (tw_j1[6].x, tw_j1[6].y);
        y2[1] = (float2) (y2[1].x * tw_tmp3.x - y2[1].y * tw_tmp3.y, y2[1].x * tw_tmp3.y + y2[1].y * tw_tmp3.x);
        x[2] = y2[0];
        x[3] = y2[1];
        y2[0] = x[4] + (float2) (x[5].x, x[5].y);
        float2 tw_tmp4 = (float2) (tw_j1[1].x, tw_j1[1].y);
        y2[0] = (float2) (y2[0].x * tw_tmp4.x - y2[0].y * tw_tmp4.y, y2[0].x * tw_tmp4.y + y2[0].y * tw_tmp4.x);
        y2[1] = x[4] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        float2 tw_tmp5 = (float2) (tw_j1[5].x, tw_j1[5].y);
        y2[1] = (float2) (y2[1].x * tw_tmp5.x - y2[1].y * tw_tmp5.y, y2[1].x * tw_tmp5.y + y2[1].y * tw_tmp5.x);
        x[4] = y2[0];
        x[5] = y2[1];
        y2[0] = x[6] + (float2) (x[7].x, x[7].y);
        float2 tw_tmp6 = (float2) (tw_j1[3].x, tw_j1[3].y);
        y2[0] = (float2) (y2[0].x * tw_tmp6.x - y2[0].y * tw_tmp6.y, y2[0].x * tw_tmp6.y + y2[0].y * tw_tmp6.x);
        y2[1] = x[6] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        float2 tw_tmp7 = (float2) (tw_j1[7].x, tw_j1[7].y);
        y2[1] = (float2) (y2[1].x * tw_tmp7.x - y2[1].y * tw_tmp7.y, y2[1].x * tw_tmp7.y + y2[1].y * tw_tmp7.x);
        x[6] = y2[0];
        x[7] = y2[1];
        local float2* sub4 = sub + (j1 * 8u + j2 * (8 * (8 * 8u)));
        sub4[0u] = x[0u];
        sub4[8 * 8u] = x[4u];
        sub4[2u * (8 * 8u)] = x[2u];
        sub4[3u * (8 * 8u)] = x[6u];
        sub4[4u * (8 * 8u)] = x[1];
        sub4[5u * (8 * 8u)] = x[5u];
        sub4[6u * (8 * 8u)] = x[3u];
        sub4[7u * (8 * 8u)] = x[7u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        float2 x[8];
        local float2* sub5 = sub + n_local * (8 * 8u);
        __attribute__((opencl_unroll_hint(8)))
        for (short j1 = 0; j1 < 8; ++j1) {
            x[j1] = sub5[j1 * 8u];
        }
        float2 y[2];
        y[0] = x[0] + (float2) (x[4].x, x[4].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[0] + (float2) (x[4].x * -0x1p+0f, x[4].y * -0x1p+0f);
        y[1] = (float2) (y[1].x, y[1].y);
        x[0] = y[0];
        x[4] = y[1];
        y[0] = x[1] + (float2) (x[5].x, x[5].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[1] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * 0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * 0x1.6a09e667f3bcdp-1f);
        x[1] = y[0];
        x[5] = y[1];
        y[0] = x[2] + (float2) (x[6].x, x[6].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[2] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y[1] = (float2) (-(y[1].y * -0x1p+0f), y[1].x * -0x1p+0f);
        x[2] = y[0];
        x[6] = y[1];
        y[0] = x[3] + (float2) (x[7].x, x[7].y);
        y[0] = (float2) (y[0].x, y[0].y);
        y[1] = x[3] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y[1] = (float2) (y[1].x * -0x1.6a09e667f3bcdp-1f - y[1].y * -0x1.6a09e667f3bcdp-1f, y[1].x * -0x1.6a09e667f3bcdp-1f + y[1].y * -0x1.6a09e667f3bcdp-1f);
        x[3] = y[0];
        x[7] = y[1];
        float2 y1[2];
        y1[0] = x[0] + (float2) (x[2].x, x[2].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[0] + (float2) (x[2].x * -0x1p+0f, x[2].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[0] = y1[0];
        x[2] = y1[1];
        y1[0] = x[4] + (float2) (x[6].x, x[6].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[4] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[4] = y1[0];
        x[6] = y1[1];
        y1[0] = x[1] + (float2) (x[3].x, x[3].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[1] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[1] = y1[0];
        x[3] = y1[1];
        y1[0] = x[5] + (float2) (x[7].x, x[7].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[5] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[5] = y1[0];
        x[7] = y1[1];
        float2 y2[2];
        y2[0] = x[0] + (float2) (x[1].x, x[1].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[0] + (float2) (x[1].x * -0x1p+0f, x[1].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[0] = y2[0];
        x[1] = y2[1];
        y2[0] = x[2] + (float2) (x[3].x, x[3].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[2] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[2] = y2[0];
        x[3] = y2[1];
        y2[0] = x[4] + (float2) (x[5].x, x[5].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[4] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[4] = y2[0];
        x[5] = y2[1];
        y2[0] = x[6] + (float2) (x[7].x, x[7].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[6] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[6] = y2[0];
        x[7] = y2[1];
        local float2* sub6 = sub + n_local * (8 * 8u);
        sub6[0u] = x[0u];
        sub6[8u] = x[4u];
        sub6[2u * 8u] = x[2u];
        sub6[3u * 8u] = x[6u];
        sub6[4u * 8u] = x[1];
        sub6[5u * 8u] = x[5u];
        sub6[6u * 8u] = x[3u];
        sub6[7u * 8u] = x[7u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /*{
        local float2* sub7 = sub + (n_local % 8 * 8 + n_local / 8 % 8) * (8 * 8u);
        if (mm < 262500u && kk < K) {
            global float2* sub8 = out + (mm + n_local * 262500u + kk * 134400000u);
            __attribute__((opencl_unroll_hint(2)))
            for (short j1 = 0; j1 < 8; ++j1) {
                sub8[j1 * (64 * 262500u)] = sub7[j1 * 8u];
            }
        }
    }*/
    if (k2 < 525 && k0 < K0) {
        for (short k1 = get_local_id(1); k1 < 512; k1 += 64) {
            short f3 = k1 % 8;
            short f2 = k1 / 8 % 8;
            short f1 = k1 / 64 % 8;
            float2 val = sub[f1 * 8 + f2 * (8 * 8) + f3 * (8 * 8 * 8)];
            float arg = -((float) 6.28318530717958647693) / (500 * 512 * 525) * k0 * (k2 + k1 * 525);
            float2 tw = (float2) (native_cos(arg), native_sin(arg));
            out[k2 + (k1 + k0 * 512) * 525] = (float2) (val.x * tw.x - val.y * tw.y, val.x * tw.y + val.y * tw.x) / 512;
        }
    }
}

kernel
__attribute__((reqd_work_group_size(16,32,1)))
__attribute__((intel_reqd_sub_group_size(8)))
void stage2(global float2* in, global float2* out, constant float2* twiddle, ulong K) {
    local float2 X1[8000];
    size_t kk = get_global_id(2);
    size_t mm = get_global_id(0);
    size_t n_local = get_local_id(1);
    local float2* sub = X1 + (get_local_id(0) + get_local_id(2) * (16u * 500u));
    for (short j1j2 = n_local; j1j2 < 20; j1j2 += 32u) {
        float2 x[25];
        if (mm < 268800u && kk < K) {
            global float2* sub1 = in + (mm + j1j2 * 268800u + kk * 134400000u);
            __attribute__((opencl_unroll_hint(25)))
            for (short j1 = 0; j1 < 25; ++j1) {
                x[j1] = sub1[j1 * (20 * 268800u)];
            }
        }
        constant float2* tw_j1 = twiddle + j1j2 * 25;
        float2 y[5];
        y[0] = x[0] + (float2) (x[5].x, x[5].y) + (float2) (x[10].x, x[10].y) + (float2) (x[15].x, x[15].y) + (float2) (x[20].x, x[20].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p1 = -0x1.9e3779b97f4a8p-1f * (x[15] + x[10]);
        float2 p2 = 0x1.2cf2304755a5ep-1f * (float2) (x[10].y - x[15].y, x[15].x - x[10].x);
        float2 p11 = 0x1.3c6ef372fe95p-2f * (x[20] + x[5]);
        float2 p21 = 0x1.e6f0e134454ffp-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[1] = x[0] + p1 + p2 + p11 + p21;
        y[1] = (float2) (y[1].x, y[1].y);
        float2 p12 = 0x1.3c6ef372fe95p-2f * (x[10] + x[15]);
        float2 p22 = 0x1.e6f0e134454ffp-1f * (float2) (x[15].y - x[10].y, x[10].x - x[15].x);
        float2 p13 = -0x1.9e3779b97f4a8p-1f * (x[20] + x[5]);
        float2 p23 = 0x1.2cf2304755a5ep-1f * (float2) (x[5].y - x[20].y, x[20].x - x[5].x);
        y[2] = x[0] + p12 + p22 + p13 + p23;
        y[2] = (float2) (y[2].x, y[2].y);
        y[3] = x[0] + p12 - p22 + p13 - p23;
        y[3] = (float2) (y[3].x, y[3].y);
        y[4] = x[0] + p1 - p2 + p11 - p21;
        y[4] = (float2) (y[4].x, y[4].y);
        x[0] = y[0];
        x[5] = y[1];
        x[10] = y[2];
        x[15] = y[3];
        x[20] = y[4];
        y[0] = x[1] + (float2) (x[6].x, x[6].y) + (float2) (x[11].x, x[11].y) + (float2) (x[16].x, x[16].y) + (float2) (x[21].x, x[21].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p14 = -0x1.9e3779b97f4a8p-1f * (x[16] + x[11]);
        float2 p24 = 0x1.2cf2304755a5ep-1f * (float2) (x[11].y - x[16].y, x[16].x - x[11].x);
        float2 p15 = 0x1.3c6ef372fe95p-2f * (x[21] + x[6]);
        float2 p25 = 0x1.e6f0e134454ffp-1f * (float2) (x[6].y - x[21].y, x[21].x - x[6].x);
        y[1] = x[1] + p14 + p24 + p15 + p25;
        y[1] = (float2) (y[1].x * 0x1.efea21d101eep-1f - y[1].y * -0x1.fd511fa1c0796p-3f, y[1].x * -0x1.fd511fa1c0796p-3f + y[1].y * 0x1.efea21d101eep-1f);
        float2 p16 = 0x1.3c6ef372fe95p-2f * (x[11] + x[16]);
        float2 p26 = 0x1.e6f0e134454ffp-1f * (float2) (x[16].y - x[11].y, x[11].x - x[16].x);
        float2 p17 = -0x1.9e3779b97f4a8p-1f * (x[21] + x[6]);
        float2 p27 = 0x1.2cf2304755a5ep-1f * (float2) (x[6].y - x[21].y, x[21].x - x[6].x);
        y[2] = x[1] + p16 + p26 + p17 + p27;
        y[2] = (float2) (y[2].x * 0x1.c0ab44e81c059p-1f - y[2].y * -0x1.ed50d5cbfa951p-2f, y[2].x * -0x1.ed50d5cbfa951p-2f + y[2].y * 0x1.c0ab44e81c059p-1f);
        y[3] = x[1] + p16 - p26 + p17 - p27;
        y[3] = (float2) (y[3].x * 0x1.753b603d2b816p-1f - y[3].y * -0x1.5e7cf55112014p-1f, y[3].x * -0x1.5e7cf55112014p-1f + y[3].y * 0x1.753b603d2b816p-1f);
        y[4] = x[1] + p14 - p24 + p15 - p25;
        y[4] = (float2) (y[4].x * 0x1.1257e3c182b51p-1f - y[4].y * -0x1.b04bbff642e86p-1f, y[4].x * -0x1.b04bbff642e86p-1f + y[4].y * 0x1.1257e3c182b51p-1f);
        x[1] = y[0];
        x[6] = y[1];
        x[11] = y[2];
        x[16] = y[3];
        x[21] = y[4];
        y[0] = x[2] + (float2) (x[7].x, x[7].y) + (float2) (x[12].x, x[12].y) + (float2) (x[17].x, x[17].y) + (float2) (x[22].x, x[22].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p18 = -0x1.9e3779b97f4a8p-1f * (x[17] + x[12]);
        float2 p28 = 0x1.2cf2304755a5ep-1f * (float2) (x[12].y - x[17].y, x[17].x - x[12].x);
        float2 p19 = 0x1.3c6ef372fe95p-2f * (x[22] + x[7]);
        float2 p29 = 0x1.e6f0e134454ffp-1f * (float2) (x[7].y - x[22].y, x[22].x - x[7].x);
        y[1] = x[2] + p18 + p28 + p19 + p29;
        y[1] = (float2) (y[1].x * 0x1.c0ab44e81c059p-1f - y[1].y * -0x1.ed50d5cbfa951p-2f, y[1].x * -0x1.ed50d5cbfa951p-2f + y[1].y * 0x1.c0ab44e81c059p-1f);
        float2 p110 = 0x1.3c6ef372fe95p-2f * (x[12] + x[17]);
        float2 p210 = 0x1.e6f0e134454ffp-1f * (float2) (x[17].y - x[12].y, x[12].x - x[17].x);
        float2 p111 = -0x1.9e3779b97f4a8p-1f * (x[22] + x[7]);
        float2 p211 = 0x1.2cf2304755a5ep-1f * (float2) (x[7].y - x[22].y, x[22].x - x[7].x);
        y[2] = x[2] + p110 + p210 + p111 + p211;
        y[2] = (float2) (y[2].x * 0x1.1257e3c182b51p-1f - y[2].y * -0x1.b04bbff642e86p-1f, y[2].x * -0x1.b04bbff642e86p-1f + y[2].y * 0x1.1257e3c182b51p-1f);
        y[3] = x[2] + p110 - p210 + p111 - p211;
        y[3] = (float2) (y[3].x * 0x1.0130a1be09379p-4f - y[3].y * -0x1.fefd5bfe443fep-1f, y[3].x * -0x1.fefd5bfe443fep-1f + y[3].y * 0x1.0130a1be09379p-4f);
        y[4] = x[2] + p18 - p28 + p19 - p29;
        y[4] = (float2) (y[4].x * -0x1.b3ff7c925819cp-2f - y[4].y * -0x1.cf457dcdc158cp-1f, y[4].x * -0x1.cf457dcdc158cp-1f + y[4].y * -0x1.b3ff7c925819cp-2f);
        x[2] = y[0];
        x[7] = y[1];
        x[12] = y[2];
        x[17] = y[3];
        x[22] = y[4];
        y[0] = x[3] + (float2) (x[8].x, x[8].y) + (float2) (x[13].x, x[13].y) + (float2) (x[18].x, x[18].y) + (float2) (x[23].x, x[23].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p112 = -0x1.9e3779b97f4a8p-1f * (x[18] + x[13]);
        float2 p212 = 0x1.2cf2304755a5ep-1f * (float2) (x[13].y - x[18].y, x[18].x - x[13].x);
        float2 p113 = 0x1.3c6ef372fe95p-2f * (x[23] + x[8]);
        float2 p213 = 0x1.e6f0e134454ffp-1f * (float2) (x[8].y - x[23].y, x[23].x - x[8].x);
        y[1] = x[3] + p112 + p212 + p113 + p213;
        y[1] = (float2) (y[1].x * 0x1.753b603d2b816p-1f - y[1].y * -0x1.5e7cf55112014p-1f, y[1].x * -0x1.5e7cf55112014p-1f + y[1].y * 0x1.753b603d2b816p-1f);
        float2 p114 = 0x1.3c6ef372fe95p-2f * (x[13] + x[18]);
        float2 p214 = 0x1.e6f0e134454ffp-1f * (float2) (x[18].y - x[13].y, x[13].x - x[18].x);
        float2 p115 = -0x1.9e3779b97f4a8p-1f * (x[23] + x[8]);
        float2 p215 = 0x1.2cf2304755a5ep-1f * (float2) (x[8].y - x[23].y, x[23].x - x[8].x);
        y[2] = x[3] + p114 + p214 + p115 + p215;
        y[2] = (float2) (y[2].x * 0x1.0130a1be09379p-4f - y[2].y * -0x1.fefd5bfe443fep-1f, y[2].x * -0x1.fefd5bfe443fep-1f + y[2].y * 0x1.0130a1be09379p-4f);
        y[3] = x[3] + p114 - p214 + p115 - p215;
        y[3] = (float2) (y[3].x * -0x1.465c6feb501bcp-1f - y[3].y * -0x1.8a80b635b6beap-1f, y[3].x * -0x1.8a80b635b6beap-1f + y[3].y * -0x1.465c6feb501bcp-1f);
        y[4] = x[3] + p112 - p212 + p113 - p213;
        y[4] = (float2) (y[4].x * -0x1.fbf675480d903p-1f - y[4].y * -0x1.00aeb5da15bep-3f, y[4].x * -0x1.00aeb5da15bep-3f + y[4].y * -0x1.fbf675480d903p-1f);
        x[3] = y[0];
        x[8] = y[1];
        x[13] = y[2];
        x[18] = y[3];
        x[23] = y[4];
        y[0] = x[4] + (float2) (x[9].x, x[9].y) + (float2) (x[14].x, x[14].y) + (float2) (x[19].x, x[19].y) + (float2) (x[24].x, x[24].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p116 = -0x1.9e3779b97f4a8p-1f * (x[19] + x[14]);
        float2 p216 = 0x1.2cf2304755a5ep-1f * (float2) (x[14].y - x[19].y, x[19].x - x[14].x);
        float2 p117 = 0x1.3c6ef372fe95p-2f * (x[24] + x[9]);
        float2 p217 = 0x1.e6f0e134454ffp-1f * (float2) (x[9].y - x[24].y, x[24].x - x[9].x);
        y[1] = x[4] + p116 + p216 + p117 + p217;
        y[1] = (float2) (y[1].x * 0x1.1257e3c182b51p-1f - y[1].y * -0x1.b04bbff642e86p-1f, y[1].x * -0x1.b04bbff642e86p-1f + y[1].y * 0x1.1257e3c182b51p-1f);
        float2 p118 = 0x1.3c6ef372fe95p-2f * (x[14] + x[19]);
        float2 p218 = 0x1.e6f0e134454ffp-1f * (float2) (x[19].y - x[14].y, x[14].x - x[19].x);
        float2 p119 = -0x1.9e3779b97f4a8p-1f * (x[24] + x[9]);
        float2 p219 = 0x1.2cf2304755a5ep-1f * (float2) (x[9].y - x[24].y, x[24].x - x[9].x);
        y[2] = x[4] + p118 + p218 + p119 + p219;
        y[2] = (float2) (y[2].x * -0x1.b3ff7c925819cp-2f - y[2].y * -0x1.cf457dcdc158cp-1f, y[2].x * -0x1.cf457dcdc158cp-1f + y[2].y * -0x1.b3ff7c925819cp-2f);
        y[3] = x[4] + p118 - p218 + p119 - p219;
        y[3] = (float2) (y[3].x * -0x1.fbf675480d903p-1f - y[3].y * -0x1.00aeb5da15bep-3f, y[3].x * -0x1.00aeb5da15bep-3f + y[3].y * -0x1.fbf675480d903p-1f);
        y[4] = x[4] + p116 - p216 + p117 - p217;
        y[4] = (float2) (y[4].x * -0x1.465c6feb501bcp-1f - y[4].y * 0x1.8a80b635b6beap-1f, y[4].x * 0x1.8a80b635b6beap-1f + y[4].y * -0x1.465c6feb501bcp-1f);
        x[4] = y[0];
        x[9] = y[1];
        x[14] = y[2];
        x[19] = y[3];
        x[24] = y[4];
        float2 y1[5];
        y1[0] = x[0] + (float2) (x[1].x, x[1].y) + (float2) (x[2].x, x[2].y) + (float2) (x[3].x, x[3].y) + (float2) (x[4].x, x[4].y);
        float2 tw_tmp = (float2) (tw_j1[0].x, tw_j1[0].y);
        y1[0] = (float2) (y1[0].x * tw_tmp.x - y1[0].y * tw_tmp.y, y1[0].x * tw_tmp.y + y1[0].y * tw_tmp.x);
        float2 p120 = -0x1.9e3779b97f4a8p-1f * (x[3] + x[2]);
        float2 p220 = 0x1.2cf2304755a5ep-1f * (float2) (x[2].y - x[3].y, x[3].x - x[2].x);
        float2 p121 = 0x1.3c6ef372fe95p-2f * (x[4] + x[1]);
        float2 p221 = 0x1.e6f0e134454ffp-1f * (float2) (x[1].y - x[4].y, x[4].x - x[1].x);
        y1[1] = x[0] + p120 + p220 + p121 + p221;
        float2 tw_tmp1 = (float2) (tw_j1[5].x, tw_j1[5].y);
        y1[1] = (float2) (y1[1].x * tw_tmp1.x - y1[1].y * tw_tmp1.y, y1[1].x * tw_tmp1.y + y1[1].y * tw_tmp1.x);
        float2 p122 = 0x1.3c6ef372fe95p-2f * (x[2] + x[3]);
        float2 p222 = 0x1.e6f0e134454ffp-1f * (float2) (x[3].y - x[2].y, x[2].x - x[3].x);
        float2 p123 = -0x1.9e3779b97f4a8p-1f * (x[4] + x[1]);
        float2 p223 = 0x1.2cf2304755a5ep-1f * (float2) (x[1].y - x[4].y, x[4].x - x[1].x);
        y1[2] = x[0] + p122 + p222 + p123 + p223;
        float2 tw_tmp2 = (float2) (tw_j1[10].x, tw_j1[10].y);
        y1[2] = (float2) (y1[2].x * tw_tmp2.x - y1[2].y * tw_tmp2.y, y1[2].x * tw_tmp2.y + y1[2].y * tw_tmp2.x);
        y1[3] = x[0] + p122 - p222 + p123 - p223;
        float2 tw_tmp3 = (float2) (tw_j1[15].x, tw_j1[15].y);
        y1[3] = (float2) (y1[3].x * tw_tmp3.x - y1[3].y * tw_tmp3.y, y1[3].x * tw_tmp3.y + y1[3].y * tw_tmp3.x);
        y1[4] = x[0] + p120 - p220 + p121 - p221;
        float2 tw_tmp4 = (float2) (tw_j1[20].x, tw_j1[20].y);
        y1[4] = (float2) (y1[4].x * tw_tmp4.x - y1[4].y * tw_tmp4.y, y1[4].x * tw_tmp4.y + y1[4].y * tw_tmp4.x);
        x[0] = y1[0];
        x[1] = y1[1];
        x[2] = y1[2];
        x[3] = y1[3];
        x[4] = y1[4];
        y1[0] = x[5] + (float2) (x[6].x, x[6].y) + (float2) (x[7].x, x[7].y) + (float2) (x[8].x, x[8].y) + (float2) (x[9].x, x[9].y);
        float2 tw_tmp5 = (float2) (tw_j1[1].x, tw_j1[1].y);
        y1[0] = (float2) (y1[0].x * tw_tmp5.x - y1[0].y * tw_tmp5.y, y1[0].x * tw_tmp5.y + y1[0].y * tw_tmp5.x);
        float2 p124 = -0x1.9e3779b97f4a8p-1f * (x[8] + x[7]);
        float2 p224 = 0x1.2cf2304755a5ep-1f * (float2) (x[7].y - x[8].y, x[8].x - x[7].x);
        float2 p125 = 0x1.3c6ef372fe95p-2f * (x[9] + x[6]);
        float2 p225 = 0x1.e6f0e134454ffp-1f * (float2) (x[6].y - x[9].y, x[9].x - x[6].x);
        y1[1] = x[5] + p124 + p224 + p125 + p225;
        float2 tw_tmp6 = (float2) (tw_j1[6].x, tw_j1[6].y);
        y1[1] = (float2) (y1[1].x * tw_tmp6.x - y1[1].y * tw_tmp6.y, y1[1].x * tw_tmp6.y + y1[1].y * tw_tmp6.x);
        float2 p126 = 0x1.3c6ef372fe95p-2f * (x[7] + x[8]);
        float2 p226 = 0x1.e6f0e134454ffp-1f * (float2) (x[8].y - x[7].y, x[7].x - x[8].x);
        float2 p127 = -0x1.9e3779b97f4a8p-1f * (x[9] + x[6]);
        float2 p227 = 0x1.2cf2304755a5ep-1f * (float2) (x[6].y - x[9].y, x[9].x - x[6].x);
        y1[2] = x[5] + p126 + p226 + p127 + p227;
        float2 tw_tmp7 = (float2) (tw_j1[11].x, tw_j1[11].y);
        y1[2] = (float2) (y1[2].x * tw_tmp7.x - y1[2].y * tw_tmp7.y, y1[2].x * tw_tmp7.y + y1[2].y * tw_tmp7.x);
        y1[3] = x[5] + p126 - p226 + p127 - p227;
        float2 tw_tmp8 = (float2) (tw_j1[16].x, tw_j1[16].y);
        y1[3] = (float2) (y1[3].x * tw_tmp8.x - y1[3].y * tw_tmp8.y, y1[3].x * tw_tmp8.y + y1[3].y * tw_tmp8.x);
        y1[4] = x[5] + p124 - p224 + p125 - p225;
        float2 tw_tmp9 = (float2) (tw_j1[21].x, tw_j1[21].y);
        y1[4] = (float2) (y1[4].x * tw_tmp9.x - y1[4].y * tw_tmp9.y, y1[4].x * tw_tmp9.y + y1[4].y * tw_tmp9.x);
        x[5] = y1[0];
        x[6] = y1[1];
        x[7] = y1[2];
        x[8] = y1[3];
        x[9] = y1[4];
        y1[0] = x[10] + (float2) (x[11].x, x[11].y) + (float2) (x[12].x, x[12].y) + (float2) (x[13].x, x[13].y) + (float2) (x[14].x, x[14].y);
        float2 tw_tmp10 = (float2) (tw_j1[2].x, tw_j1[2].y);
        y1[0] = (float2) (y1[0].x * tw_tmp10.x - y1[0].y * tw_tmp10.y, y1[0].x * tw_tmp10.y + y1[0].y * tw_tmp10.x);
        float2 p128 = -0x1.9e3779b97f4a8p-1f * (x[13] + x[12]);
        float2 p228 = 0x1.2cf2304755a5ep-1f * (float2) (x[12].y - x[13].y, x[13].x - x[12].x);
        float2 p129 = 0x1.3c6ef372fe95p-2f * (x[14] + x[11]);
        float2 p229 = 0x1.e6f0e134454ffp-1f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        y1[1] = x[10] + p128 + p228 + p129 + p229;
        float2 tw_tmp11 = (float2) (tw_j1[7].x, tw_j1[7].y);
        y1[1] = (float2) (y1[1].x * tw_tmp11.x - y1[1].y * tw_tmp11.y, y1[1].x * tw_tmp11.y + y1[1].y * tw_tmp11.x);
        float2 p130 = 0x1.3c6ef372fe95p-2f * (x[12] + x[13]);
        float2 p230 = 0x1.e6f0e134454ffp-1f * (float2) (x[13].y - x[12].y, x[12].x - x[13].x);
        float2 p131 = -0x1.9e3779b97f4a8p-1f * (x[14] + x[11]);
        float2 p231 = 0x1.2cf2304755a5ep-1f * (float2) (x[11].y - x[14].y, x[14].x - x[11].x);
        y1[2] = x[10] + p130 + p230 + p131 + p231;
        float2 tw_tmp12 = (float2) (tw_j1[12].x, tw_j1[12].y);
        y1[2] = (float2) (y1[2].x * tw_tmp12.x - y1[2].y * tw_tmp12.y, y1[2].x * tw_tmp12.y + y1[2].y * tw_tmp12.x);
        y1[3] = x[10] + p130 - p230 + p131 - p231;
        float2 tw_tmp13 = (float2) (tw_j1[17].x, tw_j1[17].y);
        y1[3] = (float2) (y1[3].x * tw_tmp13.x - y1[3].y * tw_tmp13.y, y1[3].x * tw_tmp13.y + y1[3].y * tw_tmp13.x);
        y1[4] = x[10] + p128 - p228 + p129 - p229;
        float2 tw_tmp14 = (float2) (tw_j1[22].x, tw_j1[22].y);
        y1[4] = (float2) (y1[4].x * tw_tmp14.x - y1[4].y * tw_tmp14.y, y1[4].x * tw_tmp14.y + y1[4].y * tw_tmp14.x);
        x[10] = y1[0];
        x[11] = y1[1];
        x[12] = y1[2];
        x[13] = y1[3];
        x[14] = y1[4];
        y1[0] = x[15] + (float2) (x[16].x, x[16].y) + (float2) (x[17].x, x[17].y) + (float2) (x[18].x, x[18].y) + (float2) (x[19].x, x[19].y);
        float2 tw_tmp15 = (float2) (tw_j1[3].x, tw_j1[3].y);
        y1[0] = (float2) (y1[0].x * tw_tmp15.x - y1[0].y * tw_tmp15.y, y1[0].x * tw_tmp15.y + y1[0].y * tw_tmp15.x);
        float2 p132 = -0x1.9e3779b97f4a8p-1f * (x[18] + x[17]);
        float2 p232 = 0x1.2cf2304755a5ep-1f * (float2) (x[17].y - x[18].y, x[18].x - x[17].x);
        float2 p133 = 0x1.3c6ef372fe95p-2f * (x[19] + x[16]);
        float2 p233 = 0x1.e6f0e134454ffp-1f * (float2) (x[16].y - x[19].y, x[19].x - x[16].x);
        y1[1] = x[15] + p132 + p232 + p133 + p233;
        float2 tw_tmp16 = (float2) (tw_j1[8].x, tw_j1[8].y);
        y1[1] = (float2) (y1[1].x * tw_tmp16.x - y1[1].y * tw_tmp16.y, y1[1].x * tw_tmp16.y + y1[1].y * tw_tmp16.x);
        float2 p134 = 0x1.3c6ef372fe95p-2f * (x[17] + x[18]);
        float2 p234 = 0x1.e6f0e134454ffp-1f * (float2) (x[18].y - x[17].y, x[17].x - x[18].x);
        float2 p135 = -0x1.9e3779b97f4a8p-1f * (x[19] + x[16]);
        float2 p235 = 0x1.2cf2304755a5ep-1f * (float2) (x[16].y - x[19].y, x[19].x - x[16].x);
        y1[2] = x[15] + p134 + p234 + p135 + p235;
        float2 tw_tmp17 = (float2) (tw_j1[13].x, tw_j1[13].y);
        y1[2] = (float2) (y1[2].x * tw_tmp17.x - y1[2].y * tw_tmp17.y, y1[2].x * tw_tmp17.y + y1[2].y * tw_tmp17.x);
        y1[3] = x[15] + p134 - p234 + p135 - p235;
        float2 tw_tmp18 = (float2) (tw_j1[18].x, tw_j1[18].y);
        y1[3] = (float2) (y1[3].x * tw_tmp18.x - y1[3].y * tw_tmp18.y, y1[3].x * tw_tmp18.y + y1[3].y * tw_tmp18.x);
        y1[4] = x[15] + p132 - p232 + p133 - p233;
        float2 tw_tmp19 = (float2) (tw_j1[23].x, tw_j1[23].y);
        y1[4] = (float2) (y1[4].x * tw_tmp19.x - y1[4].y * tw_tmp19.y, y1[4].x * tw_tmp19.y + y1[4].y * tw_tmp19.x);
        x[15] = y1[0];
        x[16] = y1[1];
        x[17] = y1[2];
        x[18] = y1[3];
        x[19] = y1[4];
        y1[0] = x[20] + (float2) (x[21].x, x[21].y) + (float2) (x[22].x, x[22].y) + (float2) (x[23].x, x[23].y) + (float2) (x[24].x, x[24].y);
        float2 tw_tmp20 = (float2) (tw_j1[4].x, tw_j1[4].y);
        y1[0] = (float2) (y1[0].x * tw_tmp20.x - y1[0].y * tw_tmp20.y, y1[0].x * tw_tmp20.y + y1[0].y * tw_tmp20.x);
        float2 p136 = -0x1.9e3779b97f4a8p-1f * (x[23] + x[22]);
        float2 p236 = 0x1.2cf2304755a5ep-1f * (float2) (x[22].y - x[23].y, x[23].x - x[22].x);
        float2 p137 = 0x1.3c6ef372fe95p-2f * (x[24] + x[21]);
        float2 p237 = 0x1.e6f0e134454ffp-1f * (float2) (x[21].y - x[24].y, x[24].x - x[21].x);
        y1[1] = x[20] + p136 + p236 + p137 + p237;
        float2 tw_tmp21 = (float2) (tw_j1[9].x, tw_j1[9].y);
        y1[1] = (float2) (y1[1].x * tw_tmp21.x - y1[1].y * tw_tmp21.y, y1[1].x * tw_tmp21.y + y1[1].y * tw_tmp21.x);
        float2 p138 = 0x1.3c6ef372fe95p-2f * (x[22] + x[23]);
        float2 p238 = 0x1.e6f0e134454ffp-1f * (float2) (x[23].y - x[22].y, x[22].x - x[23].x);
        float2 p139 = -0x1.9e3779b97f4a8p-1f * (x[24] + x[21]);
        float2 p239 = 0x1.2cf2304755a5ep-1f * (float2) (x[21].y - x[24].y, x[24].x - x[21].x);
        y1[2] = x[20] + p138 + p238 + p139 + p239;
        float2 tw_tmp22 = (float2) (tw_j1[14].x, tw_j1[14].y);
        y1[2] = (float2) (y1[2].x * tw_tmp22.x - y1[2].y * tw_tmp22.y, y1[2].x * tw_tmp22.y + y1[2].y * tw_tmp22.x);
        y1[3] = x[20] + p138 - p238 + p139 - p239;
        float2 tw_tmp23 = (float2) (tw_j1[19].x, tw_j1[19].y);
        y1[3] = (float2) (y1[3].x * tw_tmp23.x - y1[3].y * tw_tmp23.y, y1[3].x * tw_tmp23.y + y1[3].y * tw_tmp23.x);
        y1[4] = x[20] + p136 - p236 + p137 - p237;
        float2 tw_tmp24 = (float2) (tw_j1[24].x, tw_j1[24].y);
        y1[4] = (float2) (y1[4].x * tw_tmp24.x - y1[4].y * tw_tmp24.y, y1[4].x * tw_tmp24.y + y1[4].y * tw_tmp24.x);
        x[20] = y1[0];
        x[21] = y1[1];
        x[22] = y1[2];
        x[23] = y1[3];
        x[24] = y1[4];
        local float2* sub2 = sub + j1j2 * 16u;
        sub2[0u] = x[0u];
        sub2[20 * 16u] = x[5u];
        sub2[2u * (20 * 16u)] = x[10u];
        sub2[3u * (20 * 16u)] = x[15u];
        sub2[4u * (20 * 16u)] = x[20u];
        sub2[5u * (20 * 16u)] = x[1];
        sub2[6u * (20 * 16u)] = x[6u];
        sub2[7u * (20 * 16u)] = x[11u];
        sub2[8u * (20 * 16u)] = x[16u];
        sub2[9u * (20 * 16u)] = x[21u];
        sub2[10u * (20 * 16u)] = x[2u];
        sub2[11u * (20 * 16u)] = x[7u];
        sub2[12u * (20 * 16u)] = x[12u];
        sub2[13u * (20 * 16u)] = x[17u];
        sub2[14u * (20 * 16u)] = x[22u];
        sub2[15u * (20 * 16u)] = x[3u];
        sub2[16u * (20 * 16u)] = x[8u];
        sub2[17u * (20 * 16u)] = x[13u];
        sub2[18u * (20 * 16u)] = x[18u];
        sub2[19u * (20 * 16u)] = x[23u];
        sub2[20u * (20 * 16u)] = x[4u];
        sub2[21u * (20 * 16u)] = x[9u];
        sub2[22u * (20 * 16u)] = x[14u];
        sub2[23u * (20 * 16u)] = x[19u];
        sub2[24u * (20 * 16u)] = x[24u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (short j1j2 = n_local; j1j2 < 25; j1j2 += 32u) {
        float2 x[20];
        local float2* sub3 = sub + j1j2 * (20 * 16u);
        __attribute__((opencl_unroll_hint(20)))
        for (short j1 = 0; j1 < 20; ++j1) {
            x[j1] = sub3[j1 * 16u];
        }
        float2 y[5];
        y[0] = x[0] + (float2) (x[4].x, x[4].y) + (float2) (x[8].x, x[8].y) + (float2) (x[12].x, x[12].y) + (float2) (x[16].x, x[16].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p1 = -0x1.9e3779b97f4a8p-1f * (x[12] + x[8]);
        float2 p2 = 0x1.2cf2304755a5ep-1f * (float2) (x[8].y - x[12].y, x[12].x - x[8].x);
        float2 p11 = 0x1.3c6ef372fe95p-2f * (x[16] + x[4]);
        float2 p21 = 0x1.e6f0e134454ffp-1f * (float2) (x[4].y - x[16].y, x[16].x - x[4].x);
        y[1] = x[0] + p1 + p2 + p11 + p21;
        y[1] = (float2) (y[1].x, y[1].y);
        float2 p12 = 0x1.3c6ef372fe95p-2f * (x[8] + x[12]);
        float2 p22 = 0x1.e6f0e134454ffp-1f * (float2) (x[12].y - x[8].y, x[8].x - x[12].x);
        float2 p13 = -0x1.9e3779b97f4a8p-1f * (x[16] + x[4]);
        float2 p23 = 0x1.2cf2304755a5ep-1f * (float2) (x[4].y - x[16].y, x[16].x - x[4].x);
        y[2] = x[0] + p12 + p22 + p13 + p23;
        y[2] = (float2) (y[2].x, y[2].y);
        y[3] = x[0] + p12 - p22 + p13 - p23;
        y[3] = (float2) (y[3].x, y[3].y);
        y[4] = x[0] + p1 - p2 + p11 - p21;
        y[4] = (float2) (y[4].x, y[4].y);
        x[0] = y[0];
        x[4] = y[1];
        x[8] = y[2];
        x[12] = y[3];
        x[16] = y[4];
        y[0] = x[1] + (float2) (x[5].x, x[5].y) + (float2) (x[9].x, x[9].y) + (float2) (x[13].x, x[13].y) + (float2) (x[17].x, x[17].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p14 = -0x1.9e3779b97f4a8p-1f * (x[13] + x[9]);
        float2 p24 = 0x1.2cf2304755a5ep-1f * (float2) (x[9].y - x[13].y, x[13].x - x[9].x);
        float2 p15 = 0x1.3c6ef372fe95p-2f * (x[17] + x[5]);
        float2 p25 = 0x1.e6f0e134454ffp-1f * (float2) (x[5].y - x[17].y, x[17].x - x[5].x);
        y[1] = x[1] + p14 + p24 + p15 + p25;
        y[1] = (float2) (y[1].x * 0x1.e6f0e134454ffp-1f - y[1].y * -0x1.3c6ef372fe95p-2f, y[1].x * -0x1.3c6ef372fe95p-2f + y[1].y * 0x1.e6f0e134454ffp-1f);
        float2 p16 = 0x1.3c6ef372fe95p-2f * (x[9] + x[13]);
        float2 p26 = 0x1.e6f0e134454ffp-1f * (float2) (x[13].y - x[9].y, x[9].x - x[13].x);
        float2 p17 = -0x1.9e3779b97f4a8p-1f * (x[17] + x[5]);
        float2 p27 = 0x1.2cf2304755a5ep-1f * (float2) (x[5].y - x[17].y, x[17].x - x[5].x);
        y[2] = x[1] + p16 + p26 + p17 + p27;
        y[2] = (float2) (y[2].x * 0x1.9e3779b97f4a8p-1f - y[2].y * -0x1.2cf2304755a5ep-1f, y[2].x * -0x1.2cf2304755a5ep-1f + y[2].y * 0x1.9e3779b97f4a8p-1f);
        y[3] = x[1] + p16 - p26 + p17 - p27;
        y[3] = (float2) (y[3].x * 0x1.2cf2304755a5ep-1f - y[3].y * -0x1.9e3779b97f4a8p-1f, y[3].x * -0x1.9e3779b97f4a8p-1f + y[3].y * 0x1.2cf2304755a5ep-1f);
        y[4] = x[1] + p14 - p24 + p15 - p25;
        y[4] = (float2) (y[4].x * 0x1.3c6ef372fe95p-2f - y[4].y * -0x1.e6f0e134454ffp-1f, y[4].x * -0x1.e6f0e134454ffp-1f + y[4].y * 0x1.3c6ef372fe95p-2f);
        x[1] = y[0];
        x[5] = y[1];
        x[9] = y[2];
        x[13] = y[3];
        x[17] = y[4];
        y[0] = x[2] + (float2) (x[6].x, x[6].y) + (float2) (x[10].x, x[10].y) + (float2) (x[14].x, x[14].y) + (float2) (x[18].x, x[18].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p18 = -0x1.9e3779b97f4a8p-1f * (x[14] + x[10]);
        float2 p28 = 0x1.2cf2304755a5ep-1f * (float2) (x[10].y - x[14].y, x[14].x - x[10].x);
        float2 p19 = 0x1.3c6ef372fe95p-2f * (x[18] + x[6]);
        float2 p29 = 0x1.e6f0e134454ffp-1f * (float2) (x[6].y - x[18].y, x[18].x - x[6].x);
        y[1] = x[2] + p18 + p28 + p19 + p29;
        y[1] = (float2) (y[1].x * 0x1.9e3779b97f4a8p-1f - y[1].y * -0x1.2cf2304755a5ep-1f, y[1].x * -0x1.2cf2304755a5ep-1f + y[1].y * 0x1.9e3779b97f4a8p-1f);
        float2 p110 = 0x1.3c6ef372fe95p-2f * (x[10] + x[14]);
        float2 p210 = 0x1.e6f0e134454ffp-1f * (float2) (x[14].y - x[10].y, x[10].x - x[14].x);
        float2 p111 = -0x1.9e3779b97f4a8p-1f * (x[18] + x[6]);
        float2 p211 = 0x1.2cf2304755a5ep-1f * (float2) (x[6].y - x[18].y, x[18].x - x[6].x);
        y[2] = x[2] + p110 + p210 + p111 + p211;
        y[2] = (float2) (y[2].x * 0x1.3c6ef372fe95p-2f - y[2].y * -0x1.e6f0e134454ffp-1f, y[2].x * -0x1.e6f0e134454ffp-1f + y[2].y * 0x1.3c6ef372fe95p-2f);
        y[3] = x[2] + p110 - p210 + p111 - p211;
        y[3] = (float2) (y[3].x * -0x1.3c6ef372fe95p-2f - y[3].y * -0x1.e6f0e134454ffp-1f, y[3].x * -0x1.e6f0e134454ffp-1f + y[3].y * -0x1.3c6ef372fe95p-2f);
        y[4] = x[2] + p18 - p28 + p19 - p29;
        y[4] = (float2) (y[4].x * -0x1.9e3779b97f4a8p-1f - y[4].y * -0x1.2cf2304755a5ep-1f, y[4].x * -0x1.2cf2304755a5ep-1f + y[4].y * -0x1.9e3779b97f4a8p-1f);
        x[2] = y[0];
        x[6] = y[1];
        x[10] = y[2];
        x[14] = y[3];
        x[18] = y[4];
        y[0] = x[3] + (float2) (x[7].x, x[7].y) + (float2) (x[11].x, x[11].y) + (float2) (x[15].x, x[15].y) + (float2) (x[19].x, x[19].y);
        y[0] = (float2) (y[0].x, y[0].y);
        float2 p112 = -0x1.9e3779b97f4a8p-1f * (x[15] + x[11]);
        float2 p212 = 0x1.2cf2304755a5ep-1f * (float2) (x[11].y - x[15].y, x[15].x - x[11].x);
        float2 p113 = 0x1.3c6ef372fe95p-2f * (x[19] + x[7]);
        float2 p213 = 0x1.e6f0e134454ffp-1f * (float2) (x[7].y - x[19].y, x[19].x - x[7].x);
        y[1] = x[3] + p112 + p212 + p113 + p213;
        y[1] = (float2) (y[1].x * 0x1.2cf2304755a5ep-1f - y[1].y * -0x1.9e3779b97f4a8p-1f, y[1].x * -0x1.9e3779b97f4a8p-1f + y[1].y * 0x1.2cf2304755a5ep-1f);
        float2 p114 = 0x1.3c6ef372fe95p-2f * (x[11] + x[15]);
        float2 p214 = 0x1.e6f0e134454ffp-1f * (float2) (x[15].y - x[11].y, x[11].x - x[15].x);
        float2 p115 = -0x1.9e3779b97f4a8p-1f * (x[19] + x[7]);
        float2 p215 = 0x1.2cf2304755a5ep-1f * (float2) (x[7].y - x[19].y, x[19].x - x[7].x);
        y[2] = x[3] + p114 + p214 + p115 + p215;
        y[2] = (float2) (y[2].x * -0x1.3c6ef372fe95p-2f - y[2].y * -0x1.e6f0e134454ffp-1f, y[2].x * -0x1.e6f0e134454ffp-1f + y[2].y * -0x1.3c6ef372fe95p-2f);
        y[3] = x[3] + p114 - p214 + p115 - p215;
        y[3] = (float2) (y[3].x * -0x1.e6f0e134454ffp-1f - y[3].y * -0x1.3c6ef372fe95p-2f, y[3].x * -0x1.3c6ef372fe95p-2f + y[3].y * -0x1.e6f0e134454ffp-1f);
        y[4] = x[3] + p112 - p212 + p113 - p213;
        y[4] = (float2) (y[4].x * -0x1.9e3779b97f4a8p-1f - y[4].y * 0x1.2cf2304755a5ep-1f, y[4].x * 0x1.2cf2304755a5ep-1f + y[4].y * -0x1.9e3779b97f4a8p-1f);
        x[3] = y[0];
        x[7] = y[1];
        x[11] = y[2];
        x[15] = y[3];
        x[19] = y[4];
        float2 y1[2];
        y1[0] = x[0] + (float2) (x[2].x, x[2].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[0] + (float2) (x[2].x * -0x1p+0f, x[2].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[0] = y1[0];
        x[2] = y1[1];
        y1[0] = x[4] + (float2) (x[6].x, x[6].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[4] + (float2) (x[6].x * -0x1p+0f, x[6].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[4] = y1[0];
        x[6] = y1[1];
        y1[0] = x[8] + (float2) (x[10].x, x[10].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[8] + (float2) (x[10].x * -0x1p+0f, x[10].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[8] = y1[0];
        x[10] = y1[1];
        y1[0] = x[12] + (float2) (x[14].x, x[14].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[12] + (float2) (x[14].x * -0x1p+0f, x[14].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[12] = y1[0];
        x[14] = y1[1];
        y1[0] = x[16] + (float2) (x[18].x, x[18].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[16] + (float2) (x[18].x * -0x1p+0f, x[18].y * -0x1p+0f);
        y1[1] = (float2) (y1[1].x, y1[1].y);
        x[16] = y1[0];
        x[18] = y1[1];
        y1[0] = x[1] + (float2) (x[3].x, x[3].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[1] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[1] = y1[0];
        x[3] = y1[1];
        y1[0] = x[5] + (float2) (x[7].x, x[7].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[5] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[5] = y1[0];
        x[7] = y1[1];
        y1[0] = x[9] + (float2) (x[11].x, x[11].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[9] + (float2) (x[11].x * -0x1p+0f, x[11].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[9] = y1[0];
        x[11] = y1[1];
        y1[0] = x[13] + (float2) (x[15].x, x[15].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[13] + (float2) (x[15].x * -0x1p+0f, x[15].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[13] = y1[0];
        x[15] = y1[1];
        y1[0] = x[17] + (float2) (x[19].x, x[19].y);
        y1[0] = (float2) (y1[0].x, y1[0].y);
        y1[1] = x[17] + (float2) (x[19].x * -0x1p+0f, x[19].y * -0x1p+0f);
        y1[1] = (float2) (-(y1[1].y * -0x1p+0f), y1[1].x * -0x1p+0f);
        x[17] = y1[0];
        x[19] = y1[1];
        float2 y2[2];
        y2[0] = x[0] + (float2) (x[1].x, x[1].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[0] + (float2) (x[1].x * -0x1p+0f, x[1].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[0] = y2[0];
        x[1] = y2[1];
        y2[0] = x[2] + (float2) (x[3].x, x[3].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[2] + (float2) (x[3].x * -0x1p+0f, x[3].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[2] = y2[0];
        x[3] = y2[1];
        y2[0] = x[4] + (float2) (x[5].x, x[5].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[4] + (float2) (x[5].x * -0x1p+0f, x[5].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[4] = y2[0];
        x[5] = y2[1];
        y2[0] = x[6] + (float2) (x[7].x, x[7].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[6] + (float2) (x[7].x * -0x1p+0f, x[7].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[6] = y2[0];
        x[7] = y2[1];
        y2[0] = x[8] + (float2) (x[9].x, x[9].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[8] + (float2) (x[9].x * -0x1p+0f, x[9].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[8] = y2[0];
        x[9] = y2[1];
        y2[0] = x[10] + (float2) (x[11].x, x[11].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[10] + (float2) (x[11].x * -0x1p+0f, x[11].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[10] = y2[0];
        x[11] = y2[1];
        y2[0] = x[12] + (float2) (x[13].x, x[13].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[12] + (float2) (x[13].x * -0x1p+0f, x[13].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[12] = y2[0];
        x[13] = y2[1];
        y2[0] = x[14] + (float2) (x[15].x, x[15].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[14] + (float2) (x[15].x * -0x1p+0f, x[15].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[14] = y2[0];
        x[15] = y2[1];
        y2[0] = x[16] + (float2) (x[17].x, x[17].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[16] + (float2) (x[17].x * -0x1p+0f, x[17].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[16] = y2[0];
        x[17] = y2[1];
        y2[0] = x[18] + (float2) (x[19].x, x[19].y);
        y2[0] = (float2) (y2[0].x, y2[0].y);
        y2[1] = x[18] + (float2) (x[19].x * -0x1p+0f, x[19].y * -0x1p+0f);
        y2[1] = (float2) (y2[1].x, y2[1].y);
        x[18] = y2[0];
        x[19] = y2[1];
        local float2* sub4 = sub + j1j2 * (20 * 16u);
        sub4[0u] = x[0u];
        sub4[16u] = x[4u];
        sub4[2u * 16u] = x[8u];
        sub4[3u * 16u] = x[12u];
        sub4[4u * 16u] = x[16u];
        sub4[5u * 16u] = x[2u];
        sub4[6u * 16u] = x[6u];
        sub4[7u * 16u] = x[10u];
        sub4[8u * 16u] = x[14u];
        sub4[9u * 16u] = x[18u];
        sub4[10u * 16u] = x[1];
        sub4[11u * 16u] = x[5u];
        sub4[12u * 16u] = x[9u];
        sub4[13u * 16u] = x[13u];
        sub4[14u * 16u] = x[17u];
        sub4[15u * 16u] = x[3u];
        sub4[16u * 16u] = x[7u];
        sub4[17u * 16u] = x[11u];
        sub4[18u * 16u] = x[15u];
        sub4[19u * 16u] = x[19u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (short j2 = n_local; j2 < 25; j2 += 32u) {
        local float2* sub5 = sub + j2 * (20 * 16u);
        if (mm < 268800u && kk < K) {
            global float2* sub6 = out + (mm + j2 * 268800u + kk * 134400000u);
            __attribute__((opencl_unroll_hint(2)))
            for (short j1 = 0; j1 < 20; ++j1) {
                sub6[j1 * (25 * 268800u)] = sub5[j1 * 16u] / 500;
            }
        }
    }
}

kernel void r2c_post(global float2* input, global float2* output) {
    const size_t N = 500 * 512 * 525;
    const size_t N_stride = N + 1;
    size_t n = get_global_id(0);
    size_t n_other = N - n;
    size_t k = get_global_id(1);
    
    size_t n_load = n % N;
    size_t n_other_load = n_other % N;
    
    float2 y1 = input[n_load + k * N];
    float2 y2 = input[n_other_load + k * N];
    y2 = (float2) (y2.x, -y2.y);
    // We usually need to divide by 2, but so far we normalized by N/2 only
    // so we add the missing factor 2 here
    float2 a = (y2 + y1) / 4;
    float2 b = (y2 - y1) / 4;
    float arg = -(((float) 6.28318530717958647693) / (2 * N)) * n;
    float2 tw_i = (float2) (-native_sin(arg), native_cos(arg));
    b = (float2) (b.x * tw_i.x - b.y * tw_i.y, b.x * tw_i.y + b.y * tw_i.x);
    output[n + k * N_stride] = a + b;
    output[n_other + k * N_stride] = (float2) (a.x - b.x, b.y - a.y);
    if (n == 0) { // write n == N/2 case when n ==0
        output[N / 2 + k * N_stride] = input[N / 2 + k * N];
    }
}
)OpenCL";

using namespace bbfft;

fft1d_custom::fft1d_custom(bbfft::configuration const &cfg, cl_command_queue queue,
                           cl_context context, cl_device_id device)
    : queue_(queue) {
    if (cfg.dim != 1 || cfg.shape[0] != 1 || cfg.dir == direction::backward || cfg.callbacks) {
        throw std::runtime_error("Unsupported configuration");
    }

    bool inplace = false;
    if (cfg.istride == default_istride(1, cfg.shape, cfg.type, true) &&
        cfg.ostride == default_ostride(1, cfg.shape, cfg.type, true)) {
        inplace = true;
    } else if (cfg.istride == default_istride(1, cfg.shape, cfg.type, false) &&
               cfg.ostride == default_ostride(1, cfg.shape, cfg.type, false)) {
        inplace = false;
    } else {
        throw std::runtime_error("Non-default strides are unsupported");
    }

    auto N = cfg.shape[1];
    bool is_r2c = false;
    if (cfg.type == transform_type::r2c) {
        N /= 2;
        is_r2c = true;
    }
    if (N != 134400000) {
        throw std::runtime_error("Unsupported configuration");
    }

    cl_int err;

    const std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);
    buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                             2 * sizeof_real * N * cfg.shape[2], nullptr, &err);
    CL_CHECK(err);

    char const *code[] = {kernels};
    const std::size_t lengths[] = {sizeof(kernels)};
    program_ = clCreateProgramWithSource(context, 1, code, lengths, &err);
    CL_CHECK(err);
    err = clBuildProgram(program_, 1, &device, "-cl-std=CL2.0 -cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string log;
        std::size_t log_size;
        CL_CHECK(
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        log.resize(log_size);
        CL_CHECK(clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                       nullptr));
        throw std::runtime_error(log.c_str());
    }

    auto const create_twiddle = [](std::vector<int> const &factorization, cl_context context) {
        constexpr double tau = 6.28318530717958647693;

        int N = factorization[0] * factorization[1];
        int tw_size = N;
        auto const L = factorization.size();
        for (std::size_t k = 2; k < L; ++k) {
            N *= factorization[k];
            tw_size += N;
        }
        tw_size *= 2;

        auto twiddle = std::vector<float>(tw_size);
        auto tw_ptr = twiddle.data();
        int J1 = N;
        for (int f = L - 1; f >= 1; --f) {
            auto const Nf = factorization[f];
            J1 /= Nf;
            for (int i = 0; i < J1; ++i) {
                for (int j = 0; j < Nf; ++j) {
                    auto arg = -1 * tau / (J1 * Nf) * i * j;
                    tw_ptr[2 * (j + Nf * i)] = std::cos(arg);
                    tw_ptr[2 * (j + Nf * i) + 1] = std::sin(arg);
                }
            }
            tw_ptr += 2 * J1 * Nf;
        }

        cl_int err;
        cl_mem twiddle_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            twiddle.size() * sizeof(float), twiddle.data(), &err);
        CL_CHECK(err);
        return twiddle_dev;
    };

    cl_kernel kernel = clCreateKernel(program_, "stage0", &err);
    CL_CHECK(err);
    plans_.emplace_back(
        plan{kernel, create_twiddle({21, 25}, context), {500 * 512, 32, cfg.shape[2]}, {8, 32, 1}});
    std::uint64_t K = cfg.shape[2];
    clSetKernelArg(plans_.back().kernel, 2, sizeof(cl_mem), &plans_.back().twiddle);
    clSetKernelArg(plans_.back().kernel, 3, sizeof(K), &K);

    kernel = clCreateKernel(program_, "stage1", &err);
    CL_CHECK(err);
    plans_.emplace_back(
        plan{kernel, create_twiddle({8, 8, 8}, context), {528, 64, 500}, {8, 64, 1}});
    clSetKernelArg(plans_.back().kernel, 2, sizeof(cl_mem), &plans_.back().twiddle);
    auto const K0 = 500;
    clSetKernelArg(plans_.back().kernel, 3, sizeof(K0), &K0);

    kernel = clCreateKernel(program_, "stage2", &err);
    CL_CHECK(err);
    plans_.emplace_back(plan{
        kernel, create_twiddle({20, 25}, context), {525 * 512, 32, cfg.shape[2]}, {16, 32, 1}});
    clSetKernelArg(plans_.back().kernel, 2, sizeof(cl_mem), &plans_.back().twiddle);
    clSetKernelArg(plans_.back().kernel, 3, sizeof(K), &K);

    if (is_r2c) {
        r2c_post_ = clCreateKernel(program_, "r2c_post", &err);
        CL_CHECK(err);
    } else {
        r2c_post_ = nullptr;
    }
    r2c_post_gws_ = {N / 2, cfg.shape[2]};
}

fft1d_custom::~fft1d_custom() {
    clReleaseMemObject(buffer_);
    for (auto &p : plans_) {
        clReleaseKernel(p.kernel);
        clReleaseMemObject(p.twiddle);
    }
    if (r2c_post_) {
        clReleaseKernel(r2c_post_);
    }
    clReleaseProgram(program_);
}

auto fft1d_custom::execute(cl_mem in, cl_mem out, std::vector<cl_event> const &dep_events)
    -> cl_event {
    const bool is_r2c = r2c_post_ != nullptr;
    auto plan2_out = is_r2c ? out : buffer_;
    auto bit_reversal_out = is_r2c ? buffer_ : out;

    cl_event e0, e1, e2;

    CL_CHECK(clSetKernelArg(plans_[0].kernel, 0, sizeof(cl_mem), &in));
    CL_CHECK(clSetKernelArg(plans_[0].kernel, 1, sizeof(cl_mem), &buffer_));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, plans_[0].kernel, 3, nullptr, plans_[0].gws.data(),
                                    plans_[0].lws.data(), dep_events.size(), dep_events.data(),
                                    &e0));

    CL_CHECK(clSetKernelArg(plans_[1].kernel, 0, sizeof(cl_mem), &buffer_));
    CL_CHECK(clSetKernelArg(plans_[1].kernel, 1, sizeof(cl_mem), &out));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, plans_[1].kernel, 3, nullptr, plans_[1].gws.data(),
                                    plans_[1].lws.data(), 1, &e0, &e1));

    CL_CHECK(clSetKernelArg(plans_[2].kernel, 0, sizeof(cl_mem), &out));
    CL_CHECK(clSetKernelArg(plans_[2].kernel, 1, sizeof(cl_mem), &out));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, plans_[2].kernel, 3, nullptr, plans_[2].gws.data(),
                                    plans_[2].lws.data(), 1, &e1, &e2));

    if (is_r2c) {
        cl_event e3;
        CL_CHECK(clSetKernelArg(r2c_post_, 0, sizeof(cl_mem), &out));
        CL_CHECK(clSetKernelArg(r2c_post_, 1, sizeof(cl_mem), &out));
        CL_CHECK(clEnqueueNDRangeKernel(queue_, r2c_post_, 2, nullptr, r2c_post_gws_.data(),
                                        nullptr, 1, &e2, &e3));
        CL_CHECK(clReleaseEvent(e2));
        e2 = e3;
    }

    CL_CHECK(clReleaseEvent(e1));
    CL_CHECK(clReleaseEvent(e0));
    return e2;
}
