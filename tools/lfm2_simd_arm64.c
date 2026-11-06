#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "include/lfm2_simd_arm64.h"

/* ============================================================================
 * 1. Shared Bit-Unpacking & FP16 Helpers
 * ============================================================================ */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

static inline int get_scale_iq4_xs(int ib, const uint8_t scales_l[4], uint16_t scales_h) {
    const int low4  = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int high2 = ((scales_h >> (2 * ib)) & 0x03) << 4;
    return (low4 | high2) - 32;
}

static inline float fp16_to_fp32_arm(uint16_t h) {
#if defined(__aarch64__)
    /* Single-instruction hardware NEON FP16 -> FP32 conversion */
    uint16x4_t v16 = vdup_n_u16(h);
    float32x4_t v32 = vcvt_f32_f16(vreinterpret_f16_u16(v16));
    return vgetq_lane_f32(v32, 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
#endif
}

/* ============================================================================
 * 2. Portable Scalar Reference Kernels (Used as Fallback & Parity Oracle)
 * ============================================================================ */
void lfm2_quantize_row_q8_K_scalar(const float * restrict x,
                                   block_q8_K_t * restrict y,
                                   uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * 256;
        float max_abs = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float a = fabsf(xb[i]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            continue;
        }
        y[b].d = max_abs / 127.0f;
        const float iscale = 127.0f / max_abs;
        for (int j = 0; j < 16; ++j) {
            int32_t bsum = 0;
            for (int l = 0; l < 16; ++l) {
                int32_t q = (int32_t)roundf(xb[j * 16 + l] * iscale);
                if (q > 127)  q = 127;
                if (q < -127) q = -127;
                y[b].qs[j * 16 + l] = (int8_t)q;
                bsum += q;
            }
            y[b].bsums[j] = (int16_t)bsum;
        }
    }
}

float lfm2_vec_dot_q4_K_q8_K_scalar(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_q4_K_t *w_row = (const block_q4_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d    = fp16_to_fp32_arm(w_row[b].d)    * d8;
        const float dmin = fp16_to_fp32_arm(w_row[b].dmin) * d8;
        const uint8_t *q = w_row[b].qs;
        const int8_t  *x = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int32_t sum_q4q8 = 0, sum_mins = 0;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);
            int32_t s1 = 0, s2 = 0;
            for (int l = 0; l < 32; ++l) {
                s1 += (int32_t)(q[l] & 0x0F) * (int32_t)x[j + l];
                s2 += (int32_t)(q[l] >> 4)   * (int32_t)x[j + 32 + l];
            }
            sum_q4q8 += (int32_t)sc1 * s1 + (int32_t)sc2 * s2;
            q += 32;
            is += 2;
        }
        sumf += d * (float)sum_q4q8 - dmin * (float)sum_mins;
    }
    return sumf;
}

float lfm2_vec_dot_q6_K_q8_K_scalar(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_q6_K_t *w_row = (const block_q6_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d = fp16_to_fp32_arm(w_row[b].d) * d8;
        const uint8_t *ql = w_row[b].ql;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *sc = w_row[b].scales;
        const int8_t  *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int32_t sum_off = 0, sum_u6q8 = 0;
        for (int is = 0; is < 16; ++is) sum_off += (int32_t)sc[is] * (int32_t)bs[is];
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int32_t u1 = (ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4);
                int32_t u2 = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4);
                int32_t u3 = (ql[l +  0] >> 4)   | (((qh[l] >> 4) & 3) << 4);
                int32_t u4 = (ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4);
                sum_u6q8 += (int32_t)sc[is + 0] * u1 * (int32_t)q8[l +  0]
                          + (int32_t)sc[is + 2] * u2 * (int32_t)q8[l + 32]
                          + (int32_t)sc[is + 4] * u3 * (int32_t)q8[l + 64]
                          + (int32_t)sc[is + 6] * u4 * (int32_t)q8[l + 96];
            }
            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
        sumf += d * (float)(sum_u6q8 - 32 * sum_off);
    }
    return sumf;
}

float lfm2_vec_dot_q5_K_q8_K_scalar(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_q5_K_t *w_row = (const block_q5_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d    = fp16_to_fp32_arm(w_row[b].d)    * d8;
        const float dmin = fp16_to_fp32_arm(w_row[b].dmin) * d8;
        const uint8_t *ql = w_row[b].qs;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int32_t sum_q5q8 = 0, sum_mins = 0;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);
            int32_t s1 = 0, s2 = 0;
            for (int l = 0; l < 32; ++l) {
                int32_t q_lo = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                int32_t q_hi = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                s1 += q_lo * (int32_t)q8[j + l];
                s2 += q_hi * (int32_t)q8[j + 32 + l];
            }
            sum_q5q8 += (int32_t)sc1 * s1 + (int32_t)sc2 * s2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
        sumf += d * (float)sum_q5q8 - dmin * (float)sum_mins;
    }
    return sumf;
}

float lfm2_vec_dot_iq4_xs_q8_K_scalar(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_iq4_xs_t *w_row = (const block_iq4_xs_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d = fp16_to_fp32_arm(w_row[b].d) * d8;
        const uint8_t *qs = w_row[b].qs;
        const int8_t  *q8 = x_q8[b].qs;
        int32_t block_acc = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int32_t ls = get_scale_iq4_xs(ib, w_row[b].scales_l, w_row[b].scales_h);
            int32_t sub_dot = 0;
            for (int j = 0; j < 16; ++j) {
                sub_dot += (int32_t)kvalues_iq4nl[qs[j] & 0x0F] * (int32_t)q8[j +  0]
                         + (int32_t)kvalues_iq4nl[qs[j] >> 4]   * (int32_t)q8[j + 16];
            }
            block_acc += ls * sub_dot;
            qs += 16; q8 += 32;
        }
        sumf += d * (float)block_acc;
    }
    return sumf;
}

float lfm2_vec_dot_q4_0_q8_K_scalar(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_q4_0_t *w_row = (const block_q4_0_t *)w_raw;
    const uint32_t nb_k = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const block_q4_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        const int16_t      *bs    = x_q8[b].bsums;
        float block_sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float dw = fp16_to_fp32_arm(w_sub[j].d);
            const uint8_t *qs = w_sub[j].qs;
            const int8_t  *x_sub = q8 + j * 32;
            int32_t sum_u4q8 = 0;
            for (int l = 0; l < 16; ++l) {
                sum_u4q8 += (int32_t)(qs[l] & 0x0F) * (int32_t)x_sub[l +  0]
                          + (int32_t)(qs[l] >> 4)   * (int32_t)x_sub[l + 16];
            }
            const int32_t sum_q8 = (int32_t)bs[2 * j + 0] + (int32_t)bs[2 * j + 1];
            block_sum += dw * (float)(sum_u4q8 - 8 * sum_q8);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

float lfm2_vec_dot_q8_0_q8_K_scalar(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_q8_0_t *w_row = (const block_q8_0_t *)w_raw;
    const uint32_t nb_k = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const block_q8_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        float block_sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float dw = fp16_to_fp32_arm(w_sub[j].d);
            const int8_t *qs = w_sub[j].qs;
            const int8_t *x_sub = q8 + j * 32;
            int32_t dot = 0;
            for (int l = 0; l < 32; ++l) dot += (int32_t)qs[l] * (int32_t)x_sub[l];
            block_sum += dw * (float)dot;
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

static void scalar_1x4_q4_K(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_q4_K_q8_K_scalar((const block_q4_K_t *)w + k * rb, x, n);
}
static void scalar_1x4_q6_K(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_q6_K_q8_K_scalar((const block_q6_K_t *)w + k * rb, x, n);
}
static void scalar_1x4_q5_K(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_q5_K_q8_K_scalar((const block_q5_K_t *)w + k * rb, x, n);
}
static void scalar_1x4_iq4_xs(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_iq4_xs_q8_K_scalar((const block_iq4_xs_t *)w + k * rb, x, n);
}
static void scalar_1x4_q4_0(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_q4_0_q8_K_scalar((const block_q4_0_t *)w + k * rb, x, n);
}
static void scalar_1x4_q8_0(const void *w, size_t rb, const block_q8_K_t *x, uint32_t n, float o[4]) {
    for (int k = 0; k < 4; ++k) o[k] = lfm2_vec_dot_q8_0_q8_K_scalar((const block_q8_0_t *)w + k * rb, x, n);
}

/* ============================================================================
 * 3. Native ARM64 NEON + DotProd (`vdotq_s32`) + FP16 Kernels
 * ============================================================================ */
#if defined(__aarch64__)

#define LFM2_TARGET_ARM_DOTPROD __attribute__((target("arch=armv8.2-a+dotprod+fp16")))

/* Reduces 4 x int32x4_t accumulators into a single int32x4_t [sum0, sum1, sum2, sum3]
 * in 3 single-cycle NEON pairwise addition (`ADDP`) instructions */
static inline int32x4_t hsum4_s32_neon(int32x4_t r0, int32x4_t r1,
                                       int32x4_t r2, int32x4_t r3) {
    int32x4_t p01 = vpaddq_s32(r0, r1);
    int32x4_t p23 = vpaddq_s32(r2, r3);
    return vpaddq_s32(p01, p23);
}

/* Vectorized ARM64 NEON FP32 -> Q8_K Activation Quantizer */
void lfm2_quantize_row_q8_K_neon(const float * restrict x,
                                 block_q8_K_t * restrict y,
                                 uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;

    for (uint32_t b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * 256;

        float32x4_t vmax0 = vdupq_n_f32(0.0f);
        float32x4_t vmax1 = vdupq_n_f32(0.0f);
        float32x4_t vmax2 = vdupq_n_f32(0.0f);
        float32x4_t vmax3 = vdupq_n_f32(0.0f);

        for (int i = 0; i < 256; i += 16) {
            vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(xb + i +  0)));
            vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(xb + i +  4)));
            vmax2 = vmaxq_f32(vmax2, vabsq_f32(vld1q_f32(xb + i +  8)));
            vmax3 = vmaxq_f32(vmax3, vabsq_f32(vld1q_f32(xb + i + 12)));
        }
        vmax0 = vmaxq_f32(vmaxq_f32(vmax0, vmax1), vmaxq_f32(vmax2, vmax3));
        const float max_abs = vmaxvq_f32(vmax0);

        if (max_abs == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            continue;
        }

        y[b].d = max_abs / 127.0f;
        const float32x4_t vscale = vdupq_n_f32(127.0f / max_abs);

        for (int j = 0; j < 16; ++j) {
            const float *x16 = xb + j * 16;
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x16 +  0), vscale));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x16 +  4), vscale));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x16 +  8), vscale));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x16 + 12), vscale));

            int32x4_t sum01 = vaddq_s32(i0, i1);
            int32x4_t sum23 = vaddq_s32(i2, i3);
            y[b].bsums[j] = (int16_t)vaddvq_s32(vaddq_s32(sum01, sum23));

            int16x8_t s16_lo = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s16_hi = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            int8x16_t s8     = vcombine_s8(vqmovn_s16(s16_lo), vqmovn_s16(s16_hi));
            vst1q_s8(y[b].qs + j * 16, s8);
        }
    }
}

/* ----------------------------------------------------------------------------
 * 3.1 ARM64 DotProd Q4_K x Q8_K (1x1 and 1x4 Row-Unrolled)
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_q4_K_q8_K_neon_dp(const void * restrict w_raw,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols) {
    const block_q4_K_t * restrict w_row = (const block_q4_K_t *)w_raw;
    const uint32_t nb      = n_cols / 256;
    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = fp16_to_fp32_arm(w_row[b].d)    * d8;
        const float dmin = fp16_to_fp32_arm(w_row[b].dmin) * d8;

        const uint8_t * restrict q4 = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        int32x4_t acc_i32 = vzero;
        int32_t sum_mins  = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const uint8x16_t q4_0 = vld1q_u8(q4 +  0);
            const uint8x16_t q4_1 = vld1q_u8(q4 + 16);

            const int8x16_t u_lo0 = vreinterpretq_s8_u8(vandq_u8(q4_0, m_0f));
            const int8x16_t u_lo1 = vreinterpretq_s8_u8(vandq_u8(q4_1, m_0f));
            const int8x16_t u_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_0, 4));
            const int8x16_t u_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_1, 4));

            const int8x16_t x_lo0 = vld1q_s8(q8 + j +  0);
            const int8x16_t x_lo1 = vld1q_s8(q8 + j + 16);
            const int8x16_t x_hi0 = vld1q_s8(q8 + j + 32);
            const int8x16_t x_hi1 = vld1q_s8(q8 + j + 48);

            const int32x4_t dot_lo = vdotq_s32(vdotq_s32(vzero, u_lo0, x_lo0), u_lo1, x_lo1);
            const int32x4_t dot_hi = vdotq_s32(vdotq_s32(vzero, u_hi0, x_hi0), u_hi1, x_hi1);

            acc_i32 = vmlaq_n_s32(acc_i32, dot_lo, (int32_t)sc1);
            acc_i32 = vmlaq_n_s32(acc_i32, dot_hi, (int32_t)sc2);

            q4 += 32;
            is += 2;
        }

        sumf += d * (float)vaddvq_s32(acc_i32) - dmin * (float)sum_mins;
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_q4_K_q8_K_1x4_neon_dp(const void * restrict w_raw0,
                                        size_t row_blocks,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols,
                                        float out4[4]) {
    const block_q4_K_t * restrict w0 = (const block_q4_K_t *)w_raw0;
    const block_q4_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q4_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q4_K_t * restrict w3 = w0 + 3 * row_blocks;

    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float32x4_t v_sumf     = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < row_blocks; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        int32x4_t acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;
        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            const int8x16_t x_lo0 = vld1q_s8(q8 + j +  0);
            const int8x16_t x_lo1 = vld1q_s8(q8 + j + 16);
            const int8x16_t x_hi0 = vld1q_s8(q8 + j + 32);
            const int8x16_t x_hi1 = vld1q_s8(q8 + j + 48);
            const int32_t bsum0   = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1   = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_ARM_Q4K_ROW(w_ptr, acc_reg, mins_acc) do {                     \
                uint8_t sc1, m1, sc2, m2;                                               \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc1, &m1);                 \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc2, &m2);                 \
                (mins_acc) += (int32_t)m1 * bsum0 + (int32_t)m2 * bsum1;                \
                const uint8_t *qp   = (w_ptr)[b].qs + (j >> 1);                         \
                const uint8x16_t q0 = vld1q_u8(qp +  0);                                \
                const uint8x16_t q1 = vld1q_u8(qp + 16);                                \
                const int32x4_t dlo = vdotq_s32(                                        \
                    vdotq_s32(vzero, vreinterpretq_s8_u8(vandq_u8(q0, m_0f)), x_lo0),   \
                    vreinterpretq_s8_u8(vandq_u8(q1, m_0f)), x_lo1);                    \
                const int32x4_t dhi = vdotq_s32(                                        \
                    vdotq_s32(vzero, vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), x_hi0),    \
                    vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), x_hi1);                     \
                (acc_reg) = vmlaq_n_s32((acc_reg), dlo, (int32_t)sc1);                  \
                (acc_reg) = vmlaq_n_s32((acc_reg), dhi, (int32_t)sc2);                  \
            } while (0)

            LFM2_ARM_Q4K_ROW(w0, acc0, mins0);
            LFM2_ARM_Q4K_ROW(w1, acc1, mins1);
            LFM2_ARM_Q4K_ROW(w2, acc2, mins2);
            LFM2_ARM_Q4K_ROW(w3, acc3, mins3);

            #undef LFM2_ARM_Q4K_ROW
            is += 2;
        }

        const float32x4_t v_dots = vcvtq_f32_s32(hsum4_s32_neon(acc0, acc1, acc2, acc3));
        const int32_t mins_arr[4] = { mins0, mins1, mins2, mins3 };
        const float32x4_t v_mins = vcvtq_f32_s32(vld1q_s32(mins_arr));

        const uint16_t d_u16[4]    = { w0[b].d,    w1[b].d,    w2[b].d,    w3[b].d    };
        const uint16_t dmin_u16[4] = { w0[b].dmin, w1[b].dmin, w2[b].dmin, w3[b].dmin };
        const float32x4_t v_d    = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d_u16))),    d8);
        const float32x4_t v_dmin = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(dmin_u16))), d8);

        v_sumf = vfmaq_f32(vfmsq_f32(v_sumf, v_dmin, v_mins), v_d, v_dots);
    }

    (void)n_cols;
    vst1q_f32(out4, v_sumf);
}

/* ----------------------------------------------------------------------------
 * 3.2 ARM64 DotProd Q6_K x Q8_K (1x1 and 1x4 Row-Unrolled)
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_q6_K_q8_K_neon_dp(const void * restrict w_raw,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols) {
    const block_q6_K_t * restrict w_row = (const block_q6_K_t *)w_raw;
    const uint32_t nb      = n_cols / 256;
    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const uint8x16_t m_30  = vdupq_n_u8(0x30);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = fp16_to_fp32_arm(w_row[b].d) * d8;
        const uint8_t * restrict ql = w_row[b].ql;
        const uint8_t * restrict qh = w_row[b].qh;
        const int8_t  * restrict sc = w_row[b].scales;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        int32_t sum_off = 0;
        for (int is = 0; is < 16; ++is) sum_off += (int32_t)sc[is] * (int32_t)bs[is];

        int32x4_t acc_i32 = vzero;

        for (int half = 0; half < 2; ++half) {
            const uint8x16_t ql0_0 = vld1q_u8(ql +  0), ql0_1 = vld1q_u8(ql + 16);
            const uint8x16_t ql1_0 = vld1q_u8(ql + 32), ql1_1 = vld1q_u8(ql + 48);
            const uint8x16_t qh_0  = vld1q_u8(qh +  0), qh_1  = vld1q_u8(qh + 16);

            const int8x16_t u0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0_0, m_0f), vandq_u8(vshlq_n_u8(qh_0, 4), m_30)));
            const int8x16_t u1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0_1, m_0f), vandq_u8(vshlq_n_u8(qh_1, 4), m_30)));
            const int8x16_t u2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1_0, m_0f), vandq_u8(vshlq_n_u8(qh_0, 2), m_30)));
            const int8x16_t u3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1_1, m_0f), vandq_u8(vshlq_n_u8(qh_1, 2), m_30)));
            const int8x16_t u4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0_0, 4),  vandq_u8(qh_0, m_30)));
            const int8x16_t u5 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0_1, 4),  vandq_u8(qh_1, m_30)));
            const int8x16_t u6 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1_0, 4),  vandq_u8(vshrq_n_u8(qh_0, 2), m_30)));
            const int8x16_t u7 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1_1, 4),  vandq_u8(vshrq_n_u8(qh_1, 2), m_30)));

            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u0, vld1q_s8(q8 +   0)), (int32_t)sc[0]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u1, vld1q_s8(q8 +  16)), (int32_t)sc[1]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u2, vld1q_s8(q8 +  32)), (int32_t)sc[2]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u3, vld1q_s8(q8 +  48)), (int32_t)sc[3]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u4, vld1q_s8(q8 +  64)), (int32_t)sc[4]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u5, vld1q_s8(q8 +  80)), (int32_t)sc[5]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u6, vld1q_s8(q8 +  96)), (int32_t)sc[6]);
            acc_i32 = vmlaq_n_s32(acc_i32, vdotq_s32(vzero, u7, vld1q_s8(q8 + 112)), (int32_t)sc[7]);

            ql += 64; qh += 32; sc += 8; q8 += 128;
        }

        sumf += d * (float)(vaddvq_s32(acc_i32) - 32 * sum_off);
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_q6_K_q8_K_1x4_neon_dp(const void * restrict w_raw0, size_t rb,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols, float out4[4]) {
    const block_q6_K_t *w = (const block_q6_K_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q6_K_q8_K_neon_dp(w + k * rb, x_q8, n_cols);
}

/* ----------------------------------------------------------------------------
 * 3.3 ARM64 DotProd Q5_K x Q8_K (1x1 and 1x4 Row-Unrolled)
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_q5_K_q8_K_neon_dp(const void * restrict w_raw,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols) {
    const block_q5_K_t * restrict w_row = (const block_q5_K_t *)w_raw;
    const uint32_t nb      = n_cols / 256;
    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const uint8x16_t m_01  = vdupq_n_u8(0x01);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = fp16_to_fp32_arm(w_row[b].d)    * d8;
        const float dmin = fp16_to_fp32_arm(w_row[b].dmin) * d8;

        const uint8_t * restrict ql = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        uint8x16_t qh0 = vld1q_u8(w_row[b].qh +  0);
        uint8x16_t qh1 = vld1q_u8(w_row[b].qh + 16);

        int32x4_t acc_i32 = vzero;
        int32_t sum_mins  = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const uint8x16_t ql0 = vld1q_u8(ql +  0);
            const uint8x16_t ql1 = vld1q_u8(ql + 16);

            const uint8x16_t hi_lo0 = vshlq_n_u8(vandq_u8(qh0, m_01), 4);
            const uint8x16_t hi_lo1 = vshlq_n_u8(vandq_u8(qh1, m_01), 4);
            qh0 = vshrq_n_u8(qh0, 1);
            qh1 = vshrq_n_u8(qh1, 1);

            const uint8x16_t hi_hi0 = vshlq_n_u8(vandq_u8(qh0, m_01), 4);
            const uint8x16_t hi_hi1 = vshlq_n_u8(vandq_u8(qh1, m_01), 4);
            qh0 = vshrq_n_u8(qh0, 1);
            qh1 = vshrq_n_u8(qh1, 1);

            const int8x16_t u5_lo0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m_0f), hi_lo0));
            const int8x16_t u5_lo1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m_0f), hi_lo1));
            const int8x16_t u5_hi0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4),  hi_hi0));
            const int8x16_t u5_hi1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4),  hi_hi1));

            const int32x4_t dot_lo = vdotq_s32(
                vdotq_s32(vzero, u5_lo0, vld1q_s8(q8 + j +  0)), u5_lo1, vld1q_s8(q8 + j + 16));
            const int32x4_t dot_hi = vdotq_s32(
                vdotq_s32(vzero, u5_hi0, vld1q_s8(q8 + j + 32)), u5_hi1, vld1q_s8(q8 + j + 48));

            acc_i32 = vmlaq_n_s32(acc_i32, dot_lo, (int32_t)sc1);
            acc_i32 = vmlaq_n_s32(acc_i32, dot_hi, (int32_t)sc2);

            ql += 32;
            is += 2;
        }

        sumf += d * (float)vaddvq_s32(acc_i32) - dmin * (float)sum_mins;
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_q5_K_q8_K_1x4_neon_dp(const void * restrict w_raw0, size_t rb,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols, float out4[4]) {
    const block_q5_K_t *w = (const block_q5_K_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q5_K_q8_K_neon_dp(w + k * rb, x_q8, n_cols);
}

/* ----------------------------------------------------------------------------
 * 3.4 ARM64 DotProd IQ4_XS x Q8_K using 1-Cycle `vqtbl1q_s8` (TBL) Lookup
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_iq4_xs_q8_K_neon_dp(const void * restrict w_raw,
                                       const block_q8_K_t * restrict x_q8,
                                       uint32_t n_cols) {
    const block_iq4_xs_t * restrict w_row = (const block_iq4_xs_t *)w_raw;
    const uint32_t nb      = n_cols / 256;
    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const int8x16_t  lut   = vld1q_s8(kvalues_iq4nl);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = fp16_to_fp32_arm(w_row[b].d) * d8;
        const uint8_t * restrict qs = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;

        int32x4_t acc_i32 = vzero;

        for (int ib = 0; ib < 8; ++ib) {
            const int32_t ls     = get_scale_iq4_xs(ib, w_row[b].scales_l, w_row[b].scales_h);
            const uint8x16_t q4b = vld1q_u8(qs + ib * 16);

            const int8x16_t w_lo = vqtbl1q_s8(lut, vandq_u8(q4b, m_0f));
            const int8x16_t w_hi = vqtbl1q_s8(lut, vshrq_n_u8(q4b, 4));

            const int32x4_t sub_dot = vdotq_s32(
                vdotq_s32(vzero, w_lo, vld1q_s8(q8 + ib * 32 +  0)),
                w_hi, vld1q_s8(q8 + ib * 32 + 16));

            acc_i32 = vmlaq_n_s32(acc_i32, sub_dot, ls);
        }

        sumf += d * (float)vaddvq_s32(acc_i32);
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_iq4_xs_q8_K_1x4_neon_dp(const void * restrict w_raw0, size_t rb,
                                          const block_q8_K_t * restrict x_q8,
                                          uint32_t n_cols, float out4[4]) {
    const block_iq4_xs_t *w = (const block_iq4_xs_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_iq4_xs_q8_K_neon_dp(w + k * rb, x_q8, n_cols);
}

/* ----------------------------------------------------------------------------
 * 3.5 ARM64 DotProd Q4_0 and Q8_0 Kernels
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_q4_0_q8_K_neon_dp(const void * restrict w_raw,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols) {
    const block_q4_0_t * restrict w_row = (const block_q4_0_t *)w_raw;
    const uint32_t nb_k    = n_cols / 256;
    const uint8x16_t m_0f  = vdupq_n_u8(0x0F);
    const int32x4_t  vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q4_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        const int16_t      *bs    = x_q8[b].bsums;
        float block_sum = 0.0f;

        for (int j = 0; j < 8; ++j) {
            const float dw = fp16_to_fp32_arm(w_sub[j].d);
            const uint8x16_t q16  = vld1q_u8(w_sub[j].qs);
            const int8x16_t  u_lo = vreinterpretq_s8_u8(vandq_u8(q16, m_0f));
            const int8x16_t  u_hi = vreinterpretq_s8_u8(vshrq_n_u8(q16, 4));

            const int32x4_t dot = vdotq_s32(
                vdotq_s32(vzero, u_lo, vld1q_s8(q8 + j * 32 +  0)),
                u_hi, vld1q_s8(q8 + j * 32 + 16));

            const int32_t sum_q8 = (int32_t)bs[2 * j + 0] + (int32_t)bs[2 * j + 1];
            block_sum += dw * (float)(vaddvq_s32(dot) - 8 * sum_q8);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_q4_0_q8_K_1x4_neon_dp(const void * restrict w_raw0, size_t rb,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols, float out4[4]) {
    const block_q4_0_t *w = (const block_q4_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q4_0_q8_K_neon_dp(w + k * rb, x_q8, n_cols);
}

LFM2_TARGET_ARM_DOTPROD
float lfm2_vec_dot_q8_0_q8_K_neon_dp(const void * restrict w_raw,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols) {
    const block_q8_0_t * restrict w_row = (const block_q8_0_t *)w_raw;
    const uint32_t nb_k   = n_cols / 256;
    const int32x4_t vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q8_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        float block_sum = 0.0f;

        for (int j = 0; j < 8; ++j) {
            const float dw = fp16_to_fp32_arm(w_sub[j].d);
            const int8x16_t w0 = vld1q_s8(w_sub[j].qs +  0);
            const int8x16_t w1 = vld1q_s8(w_sub[j].qs + 16);
            const int8x16_t x0 = vld1q_s8(q8 + j * 32 +  0);
            const int8x16_t x1 = vld1q_s8(q8 + j * 32 + 16);

            const int32x4_t dot = vdotq_s32(vdotq_s32(vzero, w0, x0), w1, x1);
            block_sum += dw * (float)vaddvq_s32(dot);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

LFM2_TARGET_ARM_DOTPROD
void lfm2_vec_dot_q8_0_q8_K_1x4_neon_dp(const void * restrict w_raw0, size_t rb,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols, float out4[4]) {
    const block_q8_0_t *w = (const block_q8_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q8_0_q8_K_neon_dp(w + k * rb, x_q8, n_cols);
}

void lfm2_store_fp32_to_fp16_neon(uint16_t * restrict dst_f16,
                                  const float * restrict src_f32,
                                  uint32_t n_elems) {
    for (uint32_t i = 0; i < n_elems; i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src_f32 + i + 0));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src_f32 + i + 4));
        vst1q_u16(dst_f16 + i, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
    }
}

#endif /* __aarch64__ */

/* ============================================================================
 * 4. Runtime CPU Capability Detector & Dispatch Initializer
 * ============================================================================ */
void lfm2_init_mobile_kernels(LFM2MobileKernelTable *kt) {
    memset(kt, 0, sizeof(*kt));

    /* Default to portable C11 scalar pointers */
    kt->isa_name                        = "PORTABLE_C11_SCALAR";
    kt->quantize_q8_K                   = lfm2_quantize_row_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_Q4_K]     = lfm2_vec_dot_q4_K_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_Q6_K]     = lfm2_vec_dot_q6_K_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_Q5_K]     = lfm2_vec_dot_q5_K_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_IQ4_XS]   = lfm2_vec_dot_iq4_xs_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_Q4_0]     = lfm2_vec_dot_q4_0_q8_K_scalar;
    kt->vec_dot_1x1[LFM2_GGML_TYPE_Q8_0]     = lfm2_vec_dot_q8_0_q8_K_scalar;

    kt->vec_dot_1x4[LFM2_GGML_TYPE_Q4_K]     = scalar_1x4_q4_K;
    kt->vec_dot_1x4[LFM2_GGML_TYPE_Q6_K]     = scalar_1x4_q6_K;
    kt->vec_dot_1x4[LFM2_GGML_TYPE_Q5_K]     = scalar_1x4_q5_K;
    kt->vec_dot_1x4[LFM2_GGML_TYPE_IQ4_XS]   = scalar_1x4_iq4_xs;
    kt->vec_dot_1x4[LFM2_GGML_TYPE_Q4_0]     = scalar_1x4_q4_0;
    kt->vec_dot_1x4[LFM2_GGML_TYPE_Q8_0]     = scalar_1x4_q8_0;

#if defined(__aarch64__)
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    kt->has_neon    = (hwcap  & HWCAP_ASIMD)   != 0;
    kt->has_fp16    = (hwcap  & HWCAP_FPHP)    != 0;
    kt->has_dotprod = (hwcap  & HWCAP_ASIMDDP) != 0;
    kt->has_i8mm    = (hwcap2 & HWCAP2_I8MM)   != 0;

    const char *force_isa = getenv("LFM2_FORCE_ISA");
    if (force_isa && strcmp(force_isa, "scalar") == 0) {
        return;
    }

    if (kt->has_neon) {
        kt->quantize_q8_K = lfm2_quantize_row_q8_K_neon;
    }

    if (kt->has_dotprod) {
        kt->isa_name                        = kt->has_i8mm
                                              ? "ARM64_NEON + DOTPROD + FP16 (I8MM Ready)"
                                              : "ARM64_NEON + DOTPROD + FP16";
        kt->vec_dot_1x1[LFM2_GGML_TYPE_Q4_K]     = lfm2_vec_dot_q4_K_q8_K_neon_dp;
        kt->vec_dot_1x1[LFM2_GGML_TYPE_Q6_K]     = lfm2_vec_dot_q6_K_q8_K_neon_dp;
        kt->vec_dot_1x1[LFM2_GGML_TYPE_Q5_K]     = lfm2_vec_dot_q5_K_q8_K_neon_dp;
        kt->vec_dot_1x1[LFM2_GGML_TYPE_IQ4_XS]   = lfm2_vec_dot_iq4_xs_q8_K_neon_dp;
        kt->vec_dot_1x1[LFM2_GGML_TYPE_Q4_0]     = lfm2_vec_dot_q4_0_q8_K_neon_dp;
        kt->vec_dot_1x1[LFM2_GGML_TYPE_Q8_0]     = lfm2_vec_dot_q8_0_q8_K_neon_dp;

        kt->vec_dot_1x4[LFM2_GGML_TYPE_Q4_K]     = lfm2_vec_dot_q4_K_q8_K_1x4_neon_dp;
        kt->vec_dot_1x4[LFM2_GGML_TYPE_Q6_K]     = lfm2_vec_dot_q6_K_q8_K_1x4_neon_dp;
        kt->vec_dot_1x4[LFM2_GGML_TYPE_Q5_K]     = lfm2_vec_dot_q5_K_q8_K_1x4_neon_dp;
        kt->vec_dot_1x4[LFM2_GGML_TYPE_IQ4_XS]   = lfm2_vec_dot_iq4_xs_q8_K_1x4_neon_dp;
        kt->vec_dot_1x4[LFM2_GGML_TYPE_Q4_0]     = lfm2_vec_dot_q4_0_q8_K_1x4_neon_dp;
        kt->vec_dot_1x4[LFM2_GGML_TYPE_Q8_0]     = lfm2_vec_dot_q8_0_q8_K_1x4_neon_dp;
    }
#endif
}

/* ============================================================================
 * 5. Standalone Verification & Benchmark Entry Point
 * ============================================================================ */
#ifndef LFM2_SIMD_ARM64_NO_MAIN

static void *aligned_alloc_64(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (bytes + 63) & ~63ULL) != 0) return NULL;
    return p;
}

int main(void) {
    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);

    printf("=== LFM2 Android ARM64 NEON + DotProd (vdotq_s32) Kernel Suite ===\n");
    printf("Active Kernel ISA : %s\n", kt.isa_name);
    printf("ARM64 HWCAP Flags : NEON=%d | FP16=%d | DotProd(SDOT)=%d | I8MM=%d\n\n",
           kt.has_neon, kt.has_fp16, kt.has_dotprod, kt.has_i8mm);

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 4;
    const uint32_t nb_k   = n_cols / 256;
    const uint32_t nb_32  = n_cols / 32;

    float          *x      = (float *)aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t   *x_q8   = (block_q8_K_t *)aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q4_K_t   *w_q4k  = (block_q4_K_t *)aligned_alloc_64(n_rows * nb_k  * sizeof(block_q4_K_t));
    block_q6_K_t   *w_q6k  = (block_q6_K_t *)aligned_alloc_64(n_rows * nb_k  * sizeof(block_q6_K_t));
    block_q5_K_t   *w_q5k  = (block_q5_K_t *)aligned_alloc_64(n_rows * nb_k  * sizeof(block_q5_K_t));
    block_iq4_xs_t *w_iq4  = (block_iq4_xs_t *)aligned_alloc_64(n_rows * nb_k * sizeof(block_iq4_xs_t));
    block_q4_0_t   *w_q4_0 = (block_q4_0_t *)aligned_alloc_64(n_rows * nb_32 * sizeof(block_q4_0_t));
    block_q8_0_t   *w_q8_0 = (block_q8_0_t *)aligned_alloc_64(n_rows * nb_32 * sizeof(block_q8_0_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.019f);

    for (size_t i = 0; i < n_rows * nb_k; ++i) {
        w_q4k[i].d = 0x2E00; w_q4k[i].dmin = 0x2600;
        w_q5k[i].d = 0x2E00; w_q5k[i].dmin = 0x2600;
        w_q6k[i].d = 0x2C00;
        w_iq4[i].d = 0x2800; w_iq4[i].scales_h = (uint16_t)((i * 73) & 0xFFFF);

        for (int j = 0; j < 12; ++j) {
            w_q4k[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
            w_q5k[i].scales[j] = (uint8_t)((i * 5 + j * 9) & 0x3F);
        }
        for (int j = 0; j < 16; ++j)  w_q6k[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 4; ++j)   w_iq4[i].scales_l[j] = (uint8_t)((i * 19 + j * 41) & 0xFF);
        for (int j = 0; j < 32; ++j)  w_q5k[i].qh[j]       = (uint8_t)((i * 11 + j * 19) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_q6k[i].qh[j]       = (uint8_t)((i * 23 + j * 17) & 0xFF);
        for (int j = 0; j < 128; ++j) {
            uint8_t v = (uint8_t)((i * 17 + j * 31) & 0xFF);
            w_q4k[i].qs[j] = v;
            w_q5k[i].qs[j] = v;
            w_q6k[i].ql[j] = v;
            w_iq4[i].qs[j] = v;
        }
    }
    for (size_t i = 0; i < n_rows * nb_32; ++i) {
        w_q4_0[i].d = 0x2E00;
        w_q8_0[i].d = 0x2E00;
        for (int j = 0; j < 16; ++j) w_q4_0[i].qs[j] = (uint8_t)((i * 13 + j * 29) & 0xFF);
        for (int j = 0; j < 32; ++j) w_q8_0[i].qs[j] = (int8_t)(((int)(i * 7 + j * 11) % 251) - 125);
    }

    kt.quantize_q8_K(x, x_q8, n_cols);

    float o_q4k[4], o_q6k[4], o_q5k[4], o_iq4[4], o_q40[4], o_q80[4];
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_K](w_q4k,   nb_k,  x_q8, n_cols, o_q4k);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q6_K](w_q6k,   nb_k,  x_q8, n_cols, o_q6k);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q5_K](w_q5k,   nb_k,  x_q8, n_cols, o_q5k);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_IQ4_XS](w_iq4, nb_k,  x_q8, n_cols, o_iq4);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_0](w_q4_0,  nb_32, x_q8, n_cols, o_q40);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q8_0](w_q8_0,  nb_32, x_q8, n_cols, o_q80);

    float e_q4k = 0.0f, e_q6k = 0.0f, e_q5k = 0.0f, e_iq4 = 0.0f, e_q40 = 0.0f, e_q80 = 0.0f;
    for (int r = 0; r < 4; ++r) {
        float d1 = fabsf(o_q4k[r] - lfm2_vec_dot_q4_K_q8_K_scalar(w_q4k + r * nb_k, x_q8, n_cols));
        float d2 = fabsf(o_q6k[r] - lfm2_vec_dot_q6_K_q8_K_scalar(w_q6k + r * nb_k, x_q8, n_cols));
        float d3 = fabsf(o_q5k[r] - lfm2_vec_dot_q5_K_q8_K_scalar(w_q5k + r * nb_k, x_q8, n_cols));
        float d4 = fabsf(o_iq4[r] - lfm2_vec_dot_iq4_xs_q8_K_scalar(w_iq4 + r * nb_k, x_q8, n_cols));
        float d5 = fabsf(o_q40[r] - lfm2_vec_dot_q4_0_q8_K_scalar(w_q4_0 + r * nb_32, x_q8, n_cols));
        float d6 = fabsf(o_q80[r] - lfm2_vec_dot_q8_0_q8_K_scalar(w_q8_0 + r * nb_32, x_q8, n_cols));
        if (d1 > e_q4k) e_q4k = d1;
        if (d2 > e_q6k) e_q6k = d2;
        if (d3 > e_q5k) e_q5k = d3;
        if (d4 > e_iq4) e_iq4 = d4;
        if (d5 > e_q40) e_q40 = d5;
        if (d6 > e_q80) e_q80 = d6;
    }

    printf("1x4 Quantized Kernel Numerical Parity (Max Abs Diff vs Reference):\n");
    printf("  Q4_K   : %.8f | row[0] = %+10.4f\n", e_q4k, o_q4k[0]);
    printf("  Q6_K   : %.8f | row[0] = %+10.4f\n", e_q6k, o_q6k[0]);
    printf("  Q5_K   : %.8f | row[0] = %+10.4f\n", e_q5k, o_q5k[0]);
    printf("  IQ4_XS : %.8f | row[0] = %+10.4f\n", e_iq4, o_iq4[0]);
    printf("  Q4_0   : %.8f | row[0] = %+10.4f\n", e_q40, o_q40[0]);
    printf("  Q8_0   : %.8f | row[0] = %+10.4f\n", e_q80, o_q80[0]);

    free(x); free(x_q8); free(w_q4k); free(w_q6k);
    free(w_q5k); free(w_iq4); free(w_q4_0); free(w_q8_0);
    return EXIT_SUCCESS;
}

#endif /* LFM2_SIMD_ARM64_NO_MAIN */
