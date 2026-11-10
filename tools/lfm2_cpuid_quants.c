#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef LFM2_CPUID_QUANTS_INCLUDED
#define LFM2_CPUID_QUANTS_INCLUDED 1

#ifndef LFM2_LOADER_INCLUDED
#define LFM2_LOADER_NO_MAIN 1
#if __has_include("tools/lfm2_loader.c")
#include "tools/lfm2_loader.c"
#elif __has_include("lfm2_loader.c")
#include "lfm2_loader.c"
#else
#include "../tools/lfm2_loader.c"
#endif
#endif

#ifndef LFM2_SIMD_AVX2_H
#if __has_include("include/lfm2_simd_avx2.h")
#include "include/lfm2_simd_avx2.h"
#elif __has_include("lfm2_simd_avx2.h")
#include "lfm2_simd_avx2.h"
#else
#include "../include/lfm2_simd_avx2.h"
#endif
#endif

/* ============================================================================
 * 1. Complete GGUF / GGML Quantization Block Definitions
 * ============================================================================ */
#pragma pack(push, 1)

#ifndef LFM2_BLOCK_Q4_0_DEFINED
#define LFM2_BLOCK_Q4_0_DEFINED
/* Q4_0: 32 weights per block (18 bytes) */
typedef struct {
    uint16_t d;           /* FP16 delta scale */
    uint8_t  qs[16];      /* 32 4-bit unsigned nibbles (offset by -8) */
} block_q4_0_t;
#endif

#ifndef LFM2_BLOCK_Q8_0_DEFINED
#define LFM2_BLOCK_Q8_0_DEFINED
/* Q8_0: 32 weights per block (34 bytes) */
typedef struct {
    uint16_t d;           /* FP16 delta scale */
    int8_t   qs[32];      /* 32 signed 8-bit weights */
} block_q8_0_t;
#endif

#ifndef LFM2_BLOCK_Q4_K_DEFINED
#define LFM2_BLOCK_Q4_K_DEFINED
/* Q4_K: 256 weights per super-block (144 bytes) */
typedef struct {
    uint16_t d;           /* FP16 super-block scale for quantized scales */
    uint16_t dmin;        /* FP16 super-block scale for quantized mins   */
    uint8_t  scales[12];  /* 6-bit packed scales and mins (8 sub-blocks) */
    uint8_t  qs[128];     /* 256 4-bit unsigned weights                  */
} block_q4_K_t;
#endif

#ifndef LFM2_BLOCK_Q5_K_DEFINED
#define LFM2_BLOCK_Q5_K_DEFINED
/* Q5_K: 256 weights per super-block (176 bytes) */
typedef struct {
    uint16_t d;           /* FP16 super-block scale for quantized scales */
    uint16_t dmin;        /* FP16 super-block scale for quantized mins   */
    uint8_t  scales[12];  /* 6-bit packed scales and mins (8 sub-blocks) */
    uint8_t  qh[32];      /* 5th high bit for each of the 256 weights    */
    uint8_t  qs[128];     /* Low 4 bits for each of the 256 weights      */
} block_q5_K_t;
#endif

#ifndef LFM2_BLOCK_Q6_K_DEFINED
#define LFM2_BLOCK_Q6_K_DEFINED
/* Q6_K: 256 weights per super-block (210 bytes) */
typedef struct {
    uint8_t  ql[128];     /* Lower 4 bits of 256 6-bit weights           */
    uint8_t  qh[64];      /* Upper 2 bits of 256 6-bit weights           */
    int8_t   scales[16];  /* 8-bit signed sub-block scales               */
    uint16_t d;           /* FP16 super-block scale                      */
} block_q6_K_t;
#endif

#ifndef LFM2_BLOCK_IQ4_XS_DEFINED
#define LFM2_BLOCK_IQ4_XS_DEFINED
/* IQ4_XS: 256 weights per super-block (136 bytes) */
typedef struct {
    uint16_t d;           /* FP16 super-block scale                      */
    uint16_t scales_h;    /* High 2 bits of 8 6-bit sub-block scales     */
    uint8_t  scales_l[4]; /* Low 4 bits of 8 6-bit sub-block scales      */
    uint8_t  qs[128];     /* 256 4-bit indices into kvalues_iq4nl[16]    */
} block_iq4_xs_t;
#endif

#ifndef LFM2_BLOCK_Q8_K_DEFINED
#define LFM2_BLOCK_Q8_K_DEFINED
/* Q8_K: 256-element pre-quantized activation super-block (292 bytes) */
typedef struct {
    float   d;            /* FP32 scale: max_abs / 127.0f                */
    int8_t  qs[256];      /* 256 signed 8-bit quantized activations      */
    int16_t bsums[16];    /* Precomputed sum of each 16-element sub-block*/
} block_q8_K_t;
#endif

#pragma pack(pop)

#ifndef LFM2_KVALUES_IQ4NL_DEFINED
#define LFM2_KVALUES_IQ4NL_DEFINED
/* Official GGML non-linear 4-bit importance grid for IQ4_NL and IQ4_XS */
static const _Alignas(16) int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};
#endif

/* Safe 64-byte aligned allocation helper (size always rounded up to 64 bytes) */
#ifndef lfm2_aligned_alloc_64
#define lfm2_aligned_alloc_64 lfm2_aligned_alloc_64
static inline void *lfm2_aligned_alloc_64(size_t size) {
    size_t rounded = (size + 63) & ~((size_t)63);
    return aligned_alloc(64, rounded);
}
#define lfm2_cpuid_alloc_64 lfm2_aligned_alloc_64
#else
static inline void *lfm2_cpuid_alloc_64(size_t size) {
    return lfm2_aligned_alloc_64(size);
}
#endif

/* ============================================================================
 * 2. Portable Scalar Helpers (Safe on Any x86_64 CPU)
 * ============================================================================ */
static inline float fp16_to_fp32_scalar(uint16_t h) {
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
}

#ifndef get_scale_min_k4
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}
#endif

#ifndef LFM2_GET_SCALE_IQ4_XS_DEFINED
#define LFM2_GET_SCALE_IQ4_XS_DEFINED
#define get_scale_iq4_xs get_scale_iq4_xs
/* Unpacks the signed 6-bit sub-block scale (-32..31) for sub-block ib (0..7) in IQ4_XS */
static inline int get_scale_iq4_xs(int ib, const uint8_t scales_l[4], uint16_t scales_h) {
    const int low4  = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int high2 = ((scales_h >> (2 * ib)) & 0x03) << 4;
    return (low4 | high2) - 32;
}
#endif

#if defined(__aarch64__)
#define LFM2_SIMD_ARM64_NO_MAIN 1
#if __has_include("tools/lfm2_simd_arm64.c")
#include "tools/lfm2_simd_arm64.c"
#elif __has_include("lfm2_simd_arm64.c")
#include "lfm2_simd_arm64.c"
#else
#include "../tools/lfm2_simd_arm64.c"
#endif
#endif

/* ============================================================================
 * 3. Portable Scalar Fallback Kernels (ISA_SCALAR)
 * ============================================================================ */
#ifndef LFM2_SIMD_ARM64_INCLUDED
static inline void lfm2_quantize_row_q8_K_scalar(const float * restrict x,
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

static inline float lfm2_vec_dot_q4_0_q8_K_scalar(const void * restrict w_raw,
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
            const float dw = fp16_to_fp32_scalar(w_sub[j].d);
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

static inline float lfm2_vec_dot_q5_K_q8_K_scalar(const void * restrict w_raw,
                                                  const block_q8_K_t * restrict x_q8,
                                                  uint32_t n_cols) {
    const block_q5_K_t *w_row = (const block_q5_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = fp16_to_fp32_scalar(w_row[b].d)    * d8;
        const float dmin = fp16_to_fp32_scalar(w_row[b].dmin) * d8;
        const uint8_t *ql = w_row[b].qs;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;

        int32_t sum_q5q8 = 0;
        int32_t sum_mins = 0;
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

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += d * (float)sum_q5q8 - dmin * (float)sum_mins;
    }
    return sumf;
}

static inline float lfm2_vec_dot_iq4_xs_q8_K_scalar(const void * restrict w_raw,
                                                    const block_q8_K_t * restrict x_q8,
                                                    uint32_t n_cols) {
    const block_iq4_xs_t *w_row = (const block_iq4_xs_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = fp16_to_fp32_scalar(w_row[b].d) * d8;
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
            qs += 16;
            q8 += 32;
        }
        sumf += d * (float)block_acc;
    }
    return sumf;
}
#endif /* !LFM2_SIMD_ARM64_INCLUDED */

static inline void lfm2_vec_dot_q4_0_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                     const block_q8_K_t * restrict x_q8,
                                                     uint32_t n_cols, float out4[4]) {
    const block_q4_0_t *w = (const block_q4_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q4_0_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

static inline void lfm2_vec_dot_q5_K_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                     const block_q8_K_t * restrict x_q8,
                                                     uint32_t n_cols, float out4[4]) {
    const block_q5_K_t *w = (const block_q5_K_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q5_K_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

static inline void lfm2_vec_dot_iq4_xs_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                       const block_q8_K_t * restrict x_q8,
                                                       uint32_t n_cols, float out4[4]) {
    const block_iq4_xs_t *w = (const block_iq4_xs_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_iq4_xs_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

#ifndef LFM2_SIMD_ARM64_INCLUDED
static inline float lfm2_vec_dot_q8_0_q8_K_scalar(const void * restrict w_raw,
                                                  const block_q8_K_t * restrict x_q8,
                                                  uint32_t n_cols) {
    const block_q8_0_t *w_row = (const block_q8_0_t *)w_raw;
    const uint32_t nb_k = (n_cols + 255) / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const block_q8_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t *q8 = x_q8[b].qs;
        float block_sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float dw = fp16_to_fp32_scalar(w_sub[j].d);
            int32_t dot = 0;
            for (int l = 0; l < 32; ++l) dot += (int32_t)w_sub[j].qs[l] * (int32_t)q8[j * 32 + l];
            block_sum += dw * (float)dot;
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}
#endif /* !LFM2_SIMD_ARM64_INCLUDED */

static inline void lfm2_vec_dot_q8_0_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                     const block_q8_K_t * restrict x_q8,
                                                     uint32_t n_cols, float out4[4]) {
    const block_q8_0_t *w = (const block_q8_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q8_0_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

#ifndef LFM2_SIMD_ARM64_INCLUDED
static inline float lfm2_vec_dot_q4_K_q8_K_scalar(const void * restrict w_raw,
                                                  const block_q8_K_t * restrict x_q8,
                                                  uint32_t n_cols) {
    const block_q4_K_t *w_row = (const block_q4_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d = fp16_to_fp32_scalar(w_row[b].d) * d8;
        const float dmin = fp16_to_fp32_scalar(w_row[b].dmin) * d8;
        const uint8_t *q4 = w_row[b].qs;
        const int8_t *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int32_t acc_i32 = 0;
        int32_t sum_mins = 0;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);
            int32_t dot1 = 0, dot2 = 0;
            for (int l = 0; l < 32; ++l) {
                dot1 += (int32_t)(q4[l] & 0x0F) * (int32_t)q8[j + l];
                dot2 += (int32_t)(q4[l] >> 4)   * (int32_t)q8[j + 32 + l];
            }
            acc_i32 += dot1 * (int32_t)sc1 + dot2 * (int32_t)sc2;
            q4 += 32;
            is += 2;
        }
        sumf += d * (float)acc_i32 - dmin * (float)sum_mins;
    }
    return sumf;
}
#endif /* !LFM2_SIMD_ARM64_INCLUDED */

static inline void lfm2_vec_dot_q4_K_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                     const block_q8_K_t * restrict x_q8,
                                                     uint32_t n_cols, float out4[4]) {
    const block_q4_K_t *w = (const block_q4_K_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q4_K_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

#ifndef LFM2_SIMD_ARM64_INCLUDED
static inline float lfm2_vec_dot_q6_K_q8_K_scalar(const void * restrict w_raw,
                                                  const block_q8_K_t * restrict x_q8,
                                                  uint32_t n_cols) {
    const block_q6_K_t *w_row = (const block_q6_K_t *)w_raw;
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;
        const float d = fp16_to_fp32_scalar(w_row[b].d) * d8;
        const uint8_t *ql = w_row[b].ql;
        const uint8_t *qh = w_row[b].qh;
        const int8_t *sc = w_row[b].scales;
        const int8_t *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int32_t sum_off = 0;
        for (int is = 0; is < 16; ++is) sum_off += (int32_t)sc[is] * (int32_t)bs[is];
        int32_t acc_i32 = 0;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8;
            const int8_t *q8_h = q8 + half * 128;
            for (int l = 0; l < 32; ++l) {
                int8_t u1 = (int8_t)((ql_h[l] & 0x0F) | ((qh_h[l] & 0x03) << 4));
                int8_t u2 = (int8_t)((ql_h[l + 32] & 0x0F) | ((qh_h[l] & 0x0C) << 2));
                int8_t u3 = (int8_t)((ql_h[l] >> 4) | ((qh_h[l] & 0x30) << 0));
                int8_t u4 = (int8_t)((ql_h[l + 32] >> 4) | ((qh_h[l] & 0xC0) >> 2));
                int32_t s01 = (l < 16) ? sc_h[0] : sc_h[1];
                int32_t s23 = (l < 16) ? sc_h[2] : sc_h[3];
                int32_t s45 = (l < 16) ? sc_h[4] : sc_h[5];
                int32_t s67 = (l < 16) ? sc_h[6] : sc_h[7];
                acc_i32 += (int32_t)u1 * (int32_t)q8_h[l + 0]  * s01
                         + (int32_t)u2 * (int32_t)q8_h[l + 32] * s23
                         + (int32_t)u3 * (int32_t)q8_h[l + 64] * s45
                         + (int32_t)u4 * (int32_t)q8_h[l + 96] * s67;
            }
        }
        sumf += d * (float)(acc_i32 - 32 * sum_off);
    }
    return sumf;
}
#endif /* !LFM2_SIMD_ARM64_INCLUDED */

static inline void lfm2_vec_dot_q6_K_q8_K_1x4_scalar(const void * restrict w_raw0, size_t rb,
                                                     const block_q8_K_t * restrict x_q8,
                                                     uint32_t n_cols, float out4[4]) {
    const block_q6_K_t *w = (const block_q6_K_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q6_K_q8_K_scalar(w + k * rb, x_q8, n_cols);
}

#if defined(__x86_64__) || defined(_M_X64)
/* ============================================================================
 * 4. Target-Attributed AVX2 + FMA + F16C Kernels (Safe Multi-ISA Target)
 * ============================================================================ */
#define LFM2_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#ifndef LFM2_TARGET_AVXVNNI
#define LFM2_TARGET_AVXVNNI __attribute__((target("avx2,fma,f16c,avxvnni")))
#endif

#ifndef LFM2_PREFETCH_1X4_NTA
#define LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks) do {       \
    if ((b) + 1 < (row_blocks)) {                                             \
        _mm_prefetch((const char *)&(x_q8)[(b) + 1], _MM_HINT_T0);            \
        _mm_prefetch((const char *)&(w0)[(b) + 1],   _MM_HINT_NTA);           \
        _mm_prefetch((const char *)&(w1)[(b) + 1],   _MM_HINT_NTA);           \
        _mm_prefetch((const char *)&(w2)[(b) + 1],   _MM_HINT_NTA);           \
        _mm_prefetch((const char *)&(w3)[(b) + 1],   _MM_HINT_NTA);           \
    }                                                                         \
} while (0)
#endif

#ifndef kvalues_iq4nl_u8_defined
#define kvalues_iq4nl_u8_defined
static const _Alignas(16) uint8_t kvalues_iq4nl_u8[16] = {
      1,  24,  45,  63,  79,  93, 106, 118,
    129, 141, 153, 166, 181, 197, 217, 241
};
#endif

#ifndef mm256_set_m128i_avx2
#define mm256_set_m128i_avx2 mm256_set_m128i_cpuid
#endif
#ifndef hsum_epi32_avx2
#define hsum_epi32_avx2 hsum_epi32_cpuid
#endif

static LFM2_TARGET_AVX2 inline __m256i mm256_set_m128i_cpuid(__m128i hi, __m128i lo) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static LFM2_TARGET_AVX2 inline int32_t hsum_epi32_cpuid(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

#ifndef hsum4_epi32_avx2_defined
#define hsum4_epi32_avx2_defined
static LFM2_TARGET_AVX2 inline __m128i hsum4_epi32_avx2(__m256i r0, __m256i r1,
                                                         __m256i r2, __m256i r3) {
    __m256i h01   = _mm256_hadd_epi32(r0, r1);
    __m256i h23   = _mm256_hadd_epi32(r2, r3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123),
                         _mm256_extracti128_si256(h0123, 1));
}
#define hsum4_epi32_cpuid hsum4_epi32_avx2
#endif

#ifndef lfm2_quantize_row_q8_K_avx2
#define lfm2_quantize_row_q8_K_avx2 lfm2_quantize_row_q8_K_avx2_cpuid
#endif

LFM2_TARGET_AVX2
void lfm2_quantize_row_q8_K_avx2_cpuid(const float * restrict x,
                                       block_q8_K_t * restrict y,
                                       uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    for (uint32_t b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * 256;
        __m256 vmax = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 32) {
            __m256 v0 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i +  0));
            __m256 v1 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i +  8));
            __m256 v2 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i + 16));
            __m256 v3 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xb + i + 24));
            vmax = _mm256_max_ps(vmax, _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3)));
        }
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        const float max_abs = _mm_cvtss_f32(m128);

        if (max_abs == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            continue;
        }

        y[b].d = max_abs / 127.0f;
        const __m256 vscale = _mm256_set1_ps(127.0f / max_abs);
        for (int j = 0; j < 16; ++j) {
            const float *x16 = xb + j * 16;
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 0), vscale));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 8), vscale));
            y[b].bsums[j] = (int16_t)hsum_epi32_cpuid(_mm256_add_epi32(i0, i1));

            __m256i p16 = _mm256_packs_epi32(i0, i1);
            p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i p8  = _mm_packs_epi16(_mm256_castsi256_si128(p16),
                                          _mm256_extracti128_si256(p16, 1));
            _mm_storeu_si128((__m128i *)(y[b].qs + j * 16), p8);
        }
    }
}

/* ----------------------------------------------------------------------------
 * Kernel A: AVX2 Q4_0 x Q8_K (1x1 and 1x4 Row-Unrolled)
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_AVX2
float lfm2_vec_dot_q4_0_q8_K_avx2(const void * restrict w_raw,
                                  const block_q8_K_t * restrict x_q8,
                                  uint32_t n_cols) {
    const block_q4_0_t * restrict w_row = (const block_q4_0_t *)w_raw;
    const uint32_t nb_k   = n_cols / 256;
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q4_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        const int16_t      *bs    = x_q8[b].bsums;
        float block_sum = 0.0f;

        for (int j = 0; j < 8; ++j) {
            const float dw = _cvtsh_ss(w_sub[j].d);

            /* Load 16 packed bytes -> 16 low nibbles [0..15] + 16 high nibbles [16..31] */
            const __m128i q16 = _mm_loadu_si128((const __m128i *)w_sub[j].qs);
            const __m128i lo  = _mm_and_si128(q16, mask_0f);
            const __m128i hi  = _mm_and_si128(_mm_srli_epi16(q16, 4), mask_0f);
            const __m256i u4  = mm256_set_m128i_cpuid(hi, lo);

            const __m256i vx  = _mm256_loadu_si256((const __m256i *)(q8 + j * 32));
            const __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(u4, vx), ones_16);

            const int32_t sum_u4q8 = hsum_epi32_cpuid(dot);
            const int32_t sum_q8   = (int32_t)bs[2 * j + 0] + (int32_t)bs[2 * j + 1];
            block_sum += dw * (float)(sum_u4q8 - 8 * sum_q8);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

LFM2_TARGET_AVX2
void lfm2_vec_dot_q4_0_q8_K_1x4_avx2(const void * restrict w_raw0, size_t rb,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols, float out4[4]) {
    const block_q4_0_t *w = (const block_q4_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q4_0_q8_K_avx2(w + k * rb, x_q8, n_cols);
}

/* ----------------------------------------------------------------------------
 * Kernel B: AVX2 Q5_K x Q8_K (1x1 and 1x4 Row-Unrolled)
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_AVX2
float lfm2_vec_dot_q5_K_q8_K_avx2(const void * restrict w_raw,
                                  const block_q8_K_t * restrict x_q8,
                                  uint32_t n_cols) {
    const block_q5_K_t * restrict w_row = (const block_q5_K_t *)w_raw;
    const uint32_t nb     = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_01 = _mm256_set1_epi8(0x01);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = _cvtsh_ss(w_row[b].d)    * d8;
        const float dmin = _cvtsh_ss(w_row[b].dmin) * d8;

        const uint8_t * restrict ql = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i v_qh    = _mm256_loadu_si256((const __m256i *)w_row[b].qh);
        __m256i acc_i32 = _mm256_setzero_si256();
        int32_t sum_mins = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const __m256i q4_bytes = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i q4_lo    = _mm256_and_si256(q4_bytes, mask_0f);
            const __m256i q4_hi    = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), mask_0f);

            const __m256i qh_lo = _mm256_slli_epi16(_mm256_and_si256(v_qh, mask_01), 4);
            v_qh = _mm256_srli_epi16(v_qh, 1);
            const __m256i qh_hi = _mm256_slli_epi16(_mm256_and_si256(v_qh, mask_01), 4);
            v_qh = _mm256_srli_epi16(v_qh, 1);

            const __m256i u5_lo = _mm256_or_si256(q4_lo, qh_lo);
            const __m256i u5_hi = _mm256_or_si256(q4_hi, qh_hi);

            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));

            const __m256i dot_lo = _mm256_maddubs_epi16(u5_lo, q8_lo);
            const __m256i dot_hi = _mm256_maddubs_epi16(u5_hi, q8_hi);

            acc_i32 = _mm256_add_epi32(acc_i32,
                _mm256_madd_epi16(dot_lo, _mm256_set1_epi16((int16_t)sc1)));
            acc_i32 = _mm256_add_epi32(acc_i32,
                _mm256_madd_epi16(dot_hi, _mm256_set1_epi16((int16_t)sc2)));

            ql += 32;
            is += 2;
        }

        sumf += d * (float)hsum_epi32_cpuid(acc_i32) - dmin * (float)sum_mins;
    }
    return sumf;
}

LFM2_TARGET_AVX2
void lfm2_vec_dot_q5_K_q8_K_1x4_avx2(const void * restrict w_raw0,
                                     size_t row_blocks,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols,
                                     float out4[4]) {
    const block_q5_K_t * restrict w0 = (const block_q5_K_t *)w_raw0;
    const block_q5_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q5_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q5_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_01 = _mm256_set1_epi8(0x01);
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i vqh0 = _mm256_loadu_si256((const __m256i *)w0[b].qh);
        __m256i vqh1 = _mm256_loadu_si256((const __m256i *)w1[b].qh);
        __m256i vqh2 = _mm256_loadu_si256((const __m256i *)w2[b].qh);
        __m256i vqh3 = _mm256_loadu_si256((const __m256i *)w3[b].qh);

        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));
            const int32_t bsum0 = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1 = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_Q5K_STEP_ROW(w_ptr, vqh_reg, acc_reg, mins_acc) do {                  \
                uint8_t sc_lo, m_lo, sc_hi, m_hi;                                              \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc_lo, &m_lo);                    \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc_hi, &m_hi);                    \
                (mins_acc) += (int32_t)m_lo * bsum0 + (int32_t)m_hi * bsum1;                   \
                const __m256i q4b   = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + (j >> 1))); \
                const __m256i qh_lo = _mm256_slli_epi16(_mm256_and_si256((vqh_reg), mask_01), 4);      \
                (vqh_reg) = _mm256_srli_epi16((vqh_reg), 1);                                   \
                const __m256i qh_hi = _mm256_slli_epi16(_mm256_and_si256((vqh_reg), mask_01), 4);      \
                (vqh_reg) = _mm256_srli_epi16((vqh_reg), 1);                                   \
                const __m256i u5_lo = _mm256_or_si256(_mm256_and_si256(q4b, mask_0f), qh_lo);  \
                const __m256i u5_hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f), qh_hi); \
                const __m256i d_lo  = _mm256_maddubs_epi16(u5_lo, q8_lo);                      \
                const __m256i d_hi  = _mm256_maddubs_epi16(u5_hi, q8_hi);                      \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_lo, _mm256_set1_epi16((int16_t)sc_lo))); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_hi, _mm256_set1_epi16((int16_t)sc_hi))); \
            } while (0)

            LFM2_Q5K_STEP_ROW(w0, vqh0, acc0, mins0);
            LFM2_Q5K_STEP_ROW(w1, vqh1, acc1, mins1);
            LFM2_Q5K_STEP_ROW(w2, vqh2, acc2, mins2);
            LFM2_Q5K_STEP_ROW(w3, vqh3, acc3, mins3);

            #undef LFM2_Q5K_STEP_ROW
            is += 2;
        }

        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_cpuid(acc0, acc1, acc2, acc3));
        const __m128 v_mins = _mm_cvtepi32_ps(_mm_set_epi32(mins3, mins2, mins1, mins0));
        const __m128 v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        const __m128 v_dmin = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].dmin), _cvtsh_ss(w2[b].dmin), _cvtsh_ss(w1[b].dmin), _cvtsh_ss(w0[b].dmin)));

        v_sumf = _mm_fmadd_ps(v_d, v_dots, _mm_fnmadd_ps(v_dmin, v_mins, v_sumf));
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* ----------------------------------------------------------------------------
 * Kernel C: AVX2 IQ4_XS x Q8_K using `_mm256_shuffle_epi8` (vpshufb) Lookup
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_AVX2
float lfm2_vec_dot_iq4_xs_q8_K_avx2(const void * restrict w_raw,
                                    const block_q8_K_t * restrict x_q8,
                                    uint32_t n_cols) {
    const block_iq4_xs_t * restrict w_row = (const block_iq4_xs_t *)w_raw;
    const uint32_t nb     = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    const __m256i lut_iq4 = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)kvalues_iq4nl));

    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = _cvtsh_ss(w_row[b].d) * d8;
        const uint8_t * restrict qs = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;

        __m256i acc_i32 = _mm256_setzero_si256();

        for (int ib = 0; ib < 8; ib += 2) {
            const int16_t ls0 = (int16_t)get_scale_iq4_xs(ib + 0, w_row[b].scales_l, w_row[b].scales_h);
            const int16_t ls1 = (int16_t)get_scale_iq4_xs(ib + 1, w_row[b].scales_l, w_row[b].scales_h);

            const __m256i q4b  = _mm256_loadu_si256((const __m256i *)(qs + ib * 16));
            const __m256i idx_lo = _mm256_and_si256(q4b, mask_0f);
            const __m256i idx_hi = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f);

            const __m256i val_lo = _mm256_shuffle_epi8(lut_iq4, idx_lo);
            const __m256i val_hi = _mm256_shuffle_epi8(lut_iq4, idx_hi);

            const __m256i w_sub0 = _mm256_permute2x128_si256(val_lo, val_hi, 0x20);
            const __m256i w_sub1 = _mm256_permute2x128_si256(val_lo, val_hi, 0x31);

            const __m256i x_sub0 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 0) * 32));
            const __m256i x_sub1 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 1) * 32));

            const __m256i dot0 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(w_sub0, w_sub0), _mm256_sign_epi8(x_sub0, w_sub0));
            const __m256i dot1 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(w_sub1, w_sub1), _mm256_sign_epi8(x_sub1, w_sub1));

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot0, _mm256_set1_epi16(ls0)));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1)));
        }

        sumf += d * (float)hsum_epi32_cpuid(acc_i32);
    }
    return sumf;
}

LFM2_TARGET_AVX2
void lfm2_vec_dot_iq4_xs_q8_K_1x4_avx2(const void * restrict w_raw0,
                                       size_t row_blocks,
                                       const block_q8_K_t * restrict x_q8,
                                       uint32_t n_cols,
                                       float out4[4]) {
    const block_iq4_xs_t * restrict w0 = (const block_iq4_xs_t *)w_raw0;
    const block_iq4_xs_t * restrict w1 = w0 + 1 * row_blocks;
    const block_iq4_xs_t * restrict w2 = w0 + 2 * row_blocks;
    const block_iq4_xs_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i lut_iq4 = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)kvalues_iq4nl));

    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t * restrict q8 = x_q8[b].qs;
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();

        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i x_sub0 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 0) * 32));
            const __m256i x_sub1 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 1) * 32));

            #define LFM2_IQ4XS_STEP_ROW(w_ptr, acc_reg) do {                                   \
                const int16_t ls0 = (int16_t)get_scale_iq4_xs(ib + 0, (w_ptr)[b].scales_l, (w_ptr)[b].scales_h); \
                const int16_t ls1 = (int16_t)get_scale_iq4_xs(ib + 1, (w_ptr)[b].scales_l, (w_ptr)[b].scales_h); \
                const __m256i q4b  = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + ib * 16));             \
                const __m256i vlo  = _mm256_shuffle_epi8(lut_iq4, _mm256_and_si256(q4b, mask_0f));               \
                const __m256i vhi  = _mm256_shuffle_epi8(lut_iq4, _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f)); \
                const __m256i ws0  = _mm256_permute2x128_si256(vlo, vhi, 0x20);                \
                const __m256i ws1  = _mm256_permute2x128_si256(vlo, vhi, 0x31);                \
                const __m256i d0   = _mm256_maddubs_epi16(_mm256_sign_epi8(ws0, ws0), _mm256_sign_epi8(x_sub0, ws0)); \
                const __m256i d1   = _mm256_maddubs_epi16(_mm256_sign_epi8(ws1, ws1), _mm256_sign_epi8(x_sub1, ws1)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d0, _mm256_set1_epi16(ls0)));          \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d1, _mm256_set1_epi16(ls1)));          \
            } while (0)

            LFM2_IQ4XS_STEP_ROW(w0, acc0);
            LFM2_IQ4XS_STEP_ROW(w1, acc1);
            LFM2_IQ4XS_STEP_ROW(w2, acc2);
            LFM2_IQ4XS_STEP_ROW(w3, acc3);

            #undef LFM2_IQ4XS_STEP_ROW
        }

        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_cpuid(acc0, acc1, acc2, acc3));
        const __m128 v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        v_sumf = _mm_fmadd_ps(v_d, v_dots, v_sumf);
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}


/* ----------------------------------------------------------------------------
 * AVX2 1x1 & 1x4 Baseline Kernels with NTA Prefetching for Q4_K, Q6_K, Q8_0
 * ---------------------------------------------------------------------------- */
LFM2_TARGET_AVX2
void lfm2_vec_dot_q4_K_q8_K_1x4_avx2_nta(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]);

LFM2_TARGET_AVX2
void lfm2_vec_dot_q6_K_q8_K_1x4_avx2_nta(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]);

#ifndef LFM2_Q8_0_1X4_AVX2_DEFINED
#define LFM2_Q8_0_1X4_AVX2_DEFINED
LFM2_TARGET_AVX2
void lfm2_vec_dot_q8_0_q8_K_1x4_avx2(const void * restrict w_raw0, size_t rb,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols, float out4[4]) {
    const block_q8_0_t *w = (const block_q8_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q8_0_q8_K_avx2(w + k * rb, x_q8, n_cols);
}
#endif

/* ============================================================================
 * 2. Upgraded AVX2 1x4 Kernels with _MM_HINT_NTA Weight Prefetching
 * ============================================================================ */
LFM2_TARGET_AVX2
void lfm2_vec_dot_q4_K_q8_K_1x4_avx2_nta(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]) {
    const block_q4_K_t * restrict w0 = (const block_q4_K_t *)w_raw0;
    const block_q4_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q4_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q4_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));
            const int32_t bsum0 = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1 = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_Q4K_NTA_ROW(w_ptr, acc_reg, mins_acc) do {                     \
                uint8_t sc_lo, m_lo, sc_hi, m_hi;                                       \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc_lo, &m_lo);             \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc_hi, &m_hi);             \
                (mins_acc) += (int32_t)m_lo * bsum0 + (int32_t)m_hi * bsum1;            \
                const __m256i q4b  = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + (j >> 1))); \
                const __m256i q4lo = _mm256_and_si256(q4b, mask_0f);                    \
                const __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f); \
                const __m256i d_lo = _mm256_maddubs_epi16(q4lo, q8_lo);                 \
                const __m256i d_hi = _mm256_maddubs_epi16(q4hi, q8_hi);                 \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_lo, _mm256_set1_epi16((int16_t)sc_lo))); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_hi, _mm256_set1_epi16((int16_t)sc_hi))); \
            } while (0)

            LFM2_Q4K_NTA_ROW(w0, acc0, mins0);
            LFM2_Q4K_NTA_ROW(w1, acc1, mins1);
            LFM2_Q4K_NTA_ROW(w2, acc2, mins2);
            LFM2_Q4K_NTA_ROW(w3, acc3, mins3);

            #undef LFM2_Q4K_NTA_ROW
            is += 2;
        }

        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_avx2(acc0, acc1, acc2, acc3));
        const __m128 v_mins = _mm_cvtepi32_ps(_mm_set_epi32(mins3, mins2, mins1, mins0));
        const __m128 v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        const __m128 v_dmin = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].dmin), _cvtsh_ss(w2[b].dmin), _cvtsh_ss(w1[b].dmin), _cvtsh_ss(w0[b].dmin)));

        v_sumf = _mm_add_ps(v_sumf, _mm_sub_ps(_mm_mul_ps(v_d, v_dots), _mm_mul_ps(v_dmin, v_mins)));
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

LFM2_TARGET_AVX2
void lfm2_vec_dot_q6_K_q8_K_1x4_avx2_nta(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]) {
    const block_q6_K_t * restrict w0 = (const block_q6_K_t *)w_raw0;
    const block_q6_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q6_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q6_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t * restrict q8 = x_q8[b].qs;
        const __m256i v_bs = _mm256_loadu_si256((const __m256i *)x_q8[b].bsums);

        #define LFM2_Q6K_BSUM(w_ptr) \
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w_ptr)[b].scales)), v_bs)

        __m128i v_off4 = _mm_slli_epi32(hsum4_epi32_avx2(
            LFM2_Q6K_BSUM(w0), LFM2_Q6K_BSUM(w1), LFM2_Q6K_BSUM(w2), LFM2_Q6K_BSUM(w3)), 5);
        #undef LFM2_Q6K_BSUM

        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const size_t ql_off = (size_t)half * 64;
            const size_t qh_off = (size_t)half * 32;
            const size_t sc_off = (size_t)half * 8;
            const int8_t *q8_h  = q8 + (size_t)half * 128;

            const __m256i x0 = _mm256_loadu_si256((const __m256i *)(q8_h +  0));
            const __m256i x1 = _mm256_loadu_si256((const __m256i *)(q8_h + 32));
            const __m256i x2 = _mm256_loadu_si256((const __m256i *)(q8_h + 64));
            const __m256i x3 = _mm256_loadu_si256((const __m256i *)(q8_h + 96));

            #define LFM2_Q6K_NTA_HALF(w_ptr, acc_reg) do {                              \
                const uint8_t *ql = (w_ptr)[b].ql + ql_off;                             \
                const uint8_t *qh = (w_ptr)[b].qh + qh_off;                             \
                const int8_t  *sc = (w_ptr)[b].scales + sc_off;                         \
                const __m256i v_ql0 = _mm256_loadu_si256((const __m256i *)(ql +  0));   \
                const __m256i v_ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));   \
                const __m256i v_qh  = _mm256_loadu_si256((const __m256i *)qh);          \
                const __m256i u1 = _mm256_or_si256(_mm256_and_si256(v_ql0, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 4), mask_30)); \
                const __m256i u2 = _mm256_or_si256(_mm256_and_si256(v_ql1, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 2), mask_30)); \
                const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql0, 4), mask_0f), _mm256_and_si256(v_qh, mask_30)); \
                const __m256i u4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql1, 4), mask_0f), _mm256_and_si256(_mm256_srli_epi16(v_qh, 2), mask_30)); \
                const __m256i s01 = mm256_set_m128i_avx2(_mm_set1_epi16((int16_t)sc[1]), _mm_set1_epi16((int16_t)sc[0])); \
                const __m256i s23 = mm256_set_m128i_avx2(_mm_set1_epi16((int16_t)sc[3]), _mm_set1_epi16((int16_t)sc[2])); \
                const __m256i s45 = mm256_set_m128i_avx2(_mm_set1_epi16((int16_t)sc[5]), _mm_set1_epi16((int16_t)sc[4])); \
                const __m256i s67 = mm256_set_m128i_avx2(_mm_set1_epi16((int16_t)sc[7]), _mm_set1_epi16((int16_t)sc[6])); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(_mm256_maddubs_epi16(u1, x0), s01)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(_mm256_maddubs_epi16(u2, x1), s23)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(_mm256_maddubs_epi16(u3, x2), s45)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(_mm256_maddubs_epi16(u4, x3), s67)); \
            } while (0)

            LFM2_Q6K_NTA_HALF(w0, acc0);
            LFM2_Q6K_NTA_HALF(w1, acc1);
            LFM2_Q6K_NTA_HALF(w2, acc2);
            LFM2_Q6K_NTA_HALF(w3, acc3);

            #undef LFM2_Q6K_NTA_HALF
        }

        const __m128i v_tot  = _mm_sub_epi32(hsum4_epi32_avx2(acc0, acc1, acc2, acc3), v_off4);
        const __m128  v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        v_sumf = _mm_add_ps(v_sumf, _mm_mul_ps(v_d, _mm_cvtepi32_ps(v_tot)));
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* ============================================================================
 * 3. Hardware AVX-VNNI (`_mm256_dpbusd_epi32` / `vpdpbusd`) Kernels
 * ============================================================================ */

/* 3.1 AVX-VNNI Q4_K x Q8_K (1x1 and 1x4 with NTA Prefetching) */
LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_q4_K_q8_K_avx_vnni(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_q4_K_t * restrict w_row = (const block_q4_K_t *)w_raw;
    const uint32_t nb     = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i vzero   = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        if (b + 1 < nb) _mm_prefetch((const char *)&w_row[b + 1], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = _cvtsh_ss(w_row[b].d)    * d8;
        const float dmin = _cvtsh_ss(w_row[b].dmin) * d8;
        const uint8_t * restrict q4 = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc_i32 = vzero;
        int32_t sum_mins = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const __m256i q4b   = _mm256_loadu_si256((const __m256i *)q4);
            const __m256i u_lo  = _mm256_and_si256(q4b, mask_0f);
            const __m256i u_hi  = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f);
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));

            /* Single-instruction u8 * s8 -> i32 dot product via vpdpbusd */
            const __m256i dot_lo = _mm256_dpbusd_epi32(vzero, u_lo, q8_lo);
            const __m256i dot_hi = _mm256_dpbusd_epi32(vzero, u_hi, q8_hi);

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot_lo, _mm256_set1_epi32((int32_t)sc1)));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot_hi, _mm256_set1_epi32((int32_t)sc2)));

            q4 += 32;
            is += 2;
        }
        sumf += d * (float)hsum_epi32_avx2(acc_i32) - dmin * (float)sum_mins;
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_q4_K_q8_K_1x4_avx_vnni(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]) {
    const block_q4_K_t * restrict w0 = (const block_q4_K_t *)w_raw0;
    const block_q4_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q4_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q4_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i vzero   = _mm256_setzero_si256();
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;
        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));
            const int32_t bsum0 = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1 = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_Q4K_VNNI_ROW(w_ptr, acc_reg, mins_acc) do {                    \
                uint8_t sc_lo, m_lo, sc_hi, m_hi;                                       \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc_lo, &m_lo);             \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc_hi, &m_hi);             \
                (mins_acc) += (int32_t)m_lo * bsum0 + (int32_t)m_hi * bsum1;            \
                const __m256i q4b  = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + (j >> 1))); \
                const __m256i u_lo = _mm256_and_si256(q4b, mask_0f);                    \
                const __m256i u_hi = _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f); \
                const __m256i d_lo = _mm256_dpbusd_epi32(vzero, u_lo, q8_lo);           \
                const __m256i d_hi = _mm256_dpbusd_epi32(vzero, u_hi, q8_hi);           \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d_lo, _mm256_set1_epi32((int32_t)sc_lo))); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d_hi, _mm256_set1_epi32((int32_t)sc_hi))); \
            } while (0)

            LFM2_Q4K_VNNI_ROW(w0, acc0, mins0);
            LFM2_Q4K_VNNI_ROW(w1, acc1, mins1);
            LFM2_Q4K_VNNI_ROW(w2, acc2, mins2);
            LFM2_Q4K_VNNI_ROW(w3, acc3, mins3);

            #undef LFM2_Q4K_VNNI_ROW
            is += 2;
        }

        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_avx2(acc0, acc1, acc2, acc3));
        const __m128 v_mins = _mm_cvtepi32_ps(_mm_set_epi32(mins3, mins2, mins1, mins0));
        const __m128 v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        const __m128 v_dmin = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].dmin), _cvtsh_ss(w2[b].dmin), _cvtsh_ss(w1[b].dmin), _cvtsh_ss(w0[b].dmin)));

        v_sumf = _mm_fmadd_ps(v_d, v_dots, _mm_fnmadd_ps(v_dmin, v_mins, v_sumf));
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* 3.2 AVX-VNNI Q5_K x Q8_K (1x1 and 1x4 with NTA Prefetching) */
LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_q5_K_q8_K_avx_vnni(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_q5_K_t * restrict w_row = (const block_q5_K_t *)w_raw;
    const uint32_t nb     = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_01 = _mm256_set1_epi8(0x01);
    const __m256i vzero   = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        if (b + 1 < nb) _mm_prefetch((const char *)&w_row[b + 1], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = _cvtsh_ss(w_row[b].d)    * d8;
        const float dmin = _cvtsh_ss(w_row[b].dmin) * d8;
        const uint8_t * restrict ql = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i v_qh    = _mm256_loadu_si256((const __m256i *)w_row[b].qh);
        __m256i acc_i32 = vzero;
        int32_t sum_mins = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const __m256i q4b   = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i qh_lo = _mm256_slli_epi16(_mm256_and_si256(v_qh, mask_01), 4);
            v_qh = _mm256_srli_epi16(v_qh, 1);
            const __m256i qh_hi = _mm256_slli_epi16(_mm256_and_si256(v_qh, mask_01), 4);
            v_qh = _mm256_srli_epi16(v_qh, 1);

            const __m256i u5_lo = _mm256_or_si256(_mm256_and_si256(q4b, mask_0f), qh_lo);
            const __m256i u5_hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f), qh_hi);

            const __m256i dot_lo = _mm256_dpbusd_epi32(vzero, u5_lo, _mm256_loadu_si256((const __m256i *)(q8 + j +  0)));
            const __m256i dot_hi = _mm256_dpbusd_epi32(vzero, u5_hi, _mm256_loadu_si256((const __m256i *)(q8 + j + 32)));

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot_lo, _mm256_set1_epi32((int32_t)sc1)));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot_hi, _mm256_set1_epi32((int32_t)sc2)));

            ql += 32;
            is += 2;
        }
        sumf += d * (float)hsum_epi32_avx2(acc_i32) - dmin * (float)sum_mins;
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_q5_K_q8_K_1x4_avx_vnni(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]) {
    const block_q5_K_t * restrict w0 = (const block_q5_K_t *)w_raw0;
    const block_q5_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q5_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q5_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_01 = _mm256_set1_epi8(0x01);
    const __m256i vzero   = _mm256_setzero_si256();
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i vqh0 = _mm256_loadu_si256((const __m256i *)w0[b].qh);
        __m256i vqh1 = _mm256_loadu_si256((const __m256i *)w1[b].qh);
        __m256i vqh2 = _mm256_loadu_si256((const __m256i *)w2[b].qh);
        __m256i vqh3 = _mm256_loadu_si256((const __m256i *)w3[b].qh);

        __m256i acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;
        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));
            const int32_t bsum0 = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1 = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_Q5K_VNNI_ROW(w_ptr, vqh_reg, acc_reg, mins_acc) do {                  \
                uint8_t sc_lo, m_lo, sc_hi, m_hi;                                              \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc_lo, &m_lo);                    \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc_hi, &m_hi);                    \
                (mins_acc) += (int32_t)m_lo * bsum0 + (int32_t)m_hi * bsum1;                   \
                const __m256i q4b   = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + (j >> 1))); \
                const __m256i qh_lo = _mm256_slli_epi16(_mm256_and_si256((vqh_reg), mask_01), 4);      \
                (vqh_reg) = _mm256_srli_epi16((vqh_reg), 1);                                   \
                const __m256i qh_hi = _mm256_slli_epi16(_mm256_and_si256((vqh_reg), mask_01), 4);      \
                (vqh_reg) = _mm256_srli_epi16((vqh_reg), 1);                                   \
                const __m256i u5_lo = _mm256_or_si256(_mm256_and_si256(q4b, mask_0f), qh_lo);  \
                const __m256i u5_hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f), qh_hi); \
                const __m256i d_lo  = _mm256_dpbusd_epi32(vzero, u5_lo, q8_lo);                \
                const __m256i d_hi  = _mm256_dpbusd_epi32(vzero, u5_hi, q8_hi);                \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d_lo, _mm256_set1_epi32((int32_t)sc_lo))); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d_hi, _mm256_set1_epi32((int32_t)sc_hi))); \
            } while (0)

            LFM2_Q5K_VNNI_ROW(w0, vqh0, acc0, mins0);
            LFM2_Q5K_VNNI_ROW(w1, vqh1, acc1, mins1);
            LFM2_Q5K_VNNI_ROW(w2, vqh2, acc2, mins2);
            LFM2_Q5K_VNNI_ROW(w3, vqh3, acc3, mins3);

            #undef LFM2_Q5K_VNNI_ROW
            is += 2;
        }

        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_avx2(acc0, acc1, acc2, acc3));
        const __m128 v_mins = _mm_cvtepi32_ps(_mm_set_epi32(mins3, mins2, mins1, mins0));
        const __m128 v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        const __m128 v_dmin = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].dmin), _cvtsh_ss(w2[b].dmin), _cvtsh_ss(w1[b].dmin), _cvtsh_ss(w0[b].dmin)));

        v_sumf = _mm_fmadd_ps(v_d, v_dots, _mm_fnmadd_ps(v_dmin, v_mins, v_sumf));
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* 3.3 AVX-VNNI Q6_K x Q8_K (1x1 and 1x4 with NTA Prefetching) */
LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_q6_K_q8_K_avx_vnni(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_q6_K_t * restrict w_row = (const block_q6_K_t *)w_raw;
    const uint32_t nb     = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);
    const __m256i vzero   = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        if (b + 1 < nb) _mm_prefetch((const char *)&w_row[b + 1], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = _cvtsh_ss(w_row[b].d) * d8;
        const uint8_t * restrict ql = w_row[b].ql;
        const uint8_t * restrict qh = w_row[b].qh;
        const int8_t  * restrict sc = w_row[b].scales;
        const int8_t  * restrict q8 = x_q8[b].qs;

        const __m256i v_sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        const __m256i v_bs16 = _mm256_loadu_si256((const __m256i *)x_q8[b].bsums);
        const int32_t sum_off = hsum_epi32_avx2(_mm256_madd_epi16(v_sc16, v_bs16));

        __m256i acc_i32 = vzero;

        for (int half = 0; half < 2; ++half) {
            const __m256i v_ql0 = _mm256_loadu_si256((const __m256i *)(ql +  0));
            const __m256i v_ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i v_qh  = _mm256_loadu_si256((const __m256i *)qh);

            const __m256i u1 = _mm256_or_si256(_mm256_and_si256(v_ql0, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 4), mask_30));
            const __m256i u2 = _mm256_or_si256(_mm256_and_si256(v_ql1, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 2), mask_30));
            const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql0, 4), mask_0f), _mm256_and_si256(v_qh, mask_30));
            const __m256i u4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql1, 4), mask_0f), _mm256_and_si256(_mm256_srli_epi16(v_qh, 2), mask_30));

            const __m256i d1 = _mm256_dpbusd_epi32(vzero, u1, _mm256_loadu_si256((const __m256i *)(q8 +  0)));
            const __m256i d2 = _mm256_dpbusd_epi32(vzero, u2, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
            const __m256i d3 = _mm256_dpbusd_epi32(vzero, u3, _mm256_loadu_si256((const __m256i *)(q8 + 64)));
            const __m256i d4 = _mm256_dpbusd_epi32(vzero, u4, _mm256_loadu_si256((const __m256i *)(q8 + 96)));

            const __m256i s01 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[1]), _mm_set1_epi32((int32_t)sc[0]));
            const __m256i s23 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[3]), _mm_set1_epi32((int32_t)sc[2]));
            const __m256i s45 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[5]), _mm_set1_epi32((int32_t)sc[4]));
            const __m256i s67 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[7]), _mm_set1_epi32((int32_t)sc[6]));

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(d1, s01));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(d2, s23));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(d3, s45));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(d4, s67));

            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
        sumf += d * (float)(hsum_epi32_avx2(acc_i32) - 32 * sum_off);
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_q6_K_q8_K_1x4_avx_vnni(const void * restrict w_raw0,
                                         size_t row_blocks,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols,
                                         float out4[4]) {
    const block_q6_K_t * restrict w0 = (const block_q6_K_t *)w_raw0;
    const block_q6_K_t * restrict w1 = w0 + 1 * row_blocks;
    const block_q6_K_t * restrict w2 = w0 + 2 * row_blocks;
    const block_q6_K_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);
    const __m256i vzero   = _mm256_setzero_si256();
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t * restrict q8 = x_q8[b].qs;
        const __m256i v_bs = _mm256_loadu_si256((const __m256i *)x_q8[b].bsums);

        #define LFM2_Q6K_BSUM(w_ptr) \
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w_ptr)[b].scales)), v_bs)

        __m128i v_off4 = _mm_slli_epi32(hsum4_epi32_avx2(
            LFM2_Q6K_BSUM(w0), LFM2_Q6K_BSUM(w1), LFM2_Q6K_BSUM(w2), LFM2_Q6K_BSUM(w3)), 5);
        #undef LFM2_Q6K_BSUM

        __m256i acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;

        for (int half = 0; half < 2; ++half) {
            const size_t ql_off = (size_t)half * 64;
            const size_t qh_off = (size_t)half * 32;
            const size_t sc_off = (size_t)half * 8;
            const int8_t *q8_h  = q8 + (size_t)half * 128;

            const __m256i x0 = _mm256_loadu_si256((const __m256i *)(q8_h +  0));
            const __m256i x1 = _mm256_loadu_si256((const __m256i *)(q8_h + 32));
            const __m256i x2 = _mm256_loadu_si256((const __m256i *)(q8_h + 64));
            const __m256i x3 = _mm256_loadu_si256((const __m256i *)(q8_h + 96));

            #define LFM2_Q6K_VNNI_HALF(w_ptr, acc_reg) do {                             \
                const uint8_t *ql = (w_ptr)[b].ql + ql_off;                             \
                const uint8_t *qh = (w_ptr)[b].qh + qh_off;                             \
                const int8_t  *sc = (w_ptr)[b].scales + sc_off;                         \
                const __m256i v_ql0 = _mm256_loadu_si256((const __m256i *)(ql +  0));   \
                const __m256i v_ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));   \
                const __m256i v_qh  = _mm256_loadu_si256((const __m256i *)qh);          \
                const __m256i u1 = _mm256_or_si256(_mm256_and_si256(v_ql0, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 4), mask_30)); \
                const __m256i u2 = _mm256_or_si256(_mm256_and_si256(v_ql1, mask_0f), _mm256_and_si256(_mm256_slli_epi16(v_qh, 2), mask_30)); \
                const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql0, 4), mask_0f), _mm256_and_si256(v_qh, mask_30)); \
                const __m256i u4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v_ql1, 4), mask_0f), _mm256_and_si256(_mm256_srli_epi16(v_qh, 2), mask_30)); \
                const __m256i s01 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[1]), _mm_set1_epi32((int32_t)sc[0])); \
                const __m256i s23 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[3]), _mm_set1_epi32((int32_t)sc[2])); \
                const __m256i s45 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[5]), _mm_set1_epi32((int32_t)sc[4])); \
                const __m256i s67 = mm256_set_m128i_avx2(_mm_set1_epi32((int32_t)sc[7]), _mm_set1_epi32((int32_t)sc[6])); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(_mm256_dpbusd_epi32(vzero, u1, x0), s01)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(_mm256_dpbusd_epi32(vzero, u2, x1), s23)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(_mm256_dpbusd_epi32(vzero, u3, x2), s45)); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(_mm256_dpbusd_epi32(vzero, u4, x3), s67)); \
            } while (0)

            LFM2_Q6K_VNNI_HALF(w0, acc0);
            LFM2_Q6K_VNNI_HALF(w1, acc1);
            LFM2_Q6K_VNNI_HALF(w2, acc2);
            LFM2_Q6K_VNNI_HALF(w3, acc3);

            #undef LFM2_Q6K_VNNI_HALF
        }

        const __m128i v_tot = _mm_sub_epi32(hsum4_epi32_avx2(acc0, acc1, acc2, acc3), v_off4);
        const __m128  v_d   = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        v_sumf = _mm_fmadd_ps(v_d, _mm_cvtepi32_ps(v_tot), v_sumf);
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* 3.4 AVX-VNNI IQ4_XS x Q8_K (Uses unsigned-shifted lut_u8 + vpdpbusd -> zero _mm256_sign_epi8!) */
LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_iq4_xs_q8_K_avx_vnni(const void * restrict w_raw,
                                        const block_q8_K_t * restrict x_q8,
                                        uint32_t n_cols) {
    const block_iq4_xs_t * restrict w_row = (const block_iq4_xs_t *)w_raw;
    const uint32_t nb      = n_cols / 256;
    const __m256i mask_0f  = _mm256_set1_epi8(0x0F);
    const __m256i lut_u8   = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)kvalues_iq4nl_u8));
    const __m256i vzero    = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        if (b + 1 < nb) _mm_prefetch((const char *)&w_row[b + 1], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = _cvtsh_ss(w_row[b].d) * d8;
        const uint8_t * restrict qs = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc_i32  = vzero;
        int32_t sum_off  = 0;

        for (int ib = 0; ib < 8; ib += 2) {
            const int32_t ls0 = get_scale_iq4_xs(ib + 0, w_row[b].scales_l, w_row[b].scales_h);
            const int32_t ls1 = get_scale_iq4_xs(ib + 1, w_row[b].scales_l, w_row[b].scales_h);

            const int32_t bs0 = (int32_t)bs[2 * (ib + 0) + 0] + (int32_t)bs[2 * (ib + 0) + 1];
            const int32_t bs1 = (int32_t)bs[2 * (ib + 1) + 0] + (int32_t)bs[2 * (ib + 1) + 1];
            sum_off += ls0 * bs0 + ls1 * bs1;

            const __m256i q4b   = _mm256_loadu_si256((const __m256i *)(qs + ib * 16));
            const __m256i vlo   = _mm256_shuffle_epi8(lut_u8, _mm256_and_si256(q4b, mask_0f));
            const __m256i vhi   = _mm256_shuffle_epi8(lut_u8, _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f));
            const __m256i wu0   = _mm256_permute2x128_si256(vlo, vhi, 0x20);
            const __m256i wu1   = _mm256_permute2x128_si256(vlo, vhi, 0x31);

            const __m256i dot0  = _mm256_dpbusd_epi32(vzero, wu0, _mm256_loadu_si256((const __m256i *)(q8 + (ib + 0) * 32)));
            const __m256i dot1  = _mm256_dpbusd_epi32(vzero, wu1, _mm256_loadu_si256((const __m256i *)(q8 + (ib + 1) * 32)));

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot0, _mm256_set1_epi32(ls0)));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(dot1, _mm256_set1_epi32(ls1)));
        }
        sumf += d * (float)(hsum_epi32_avx2(acc_i32) - 128 * sum_off);
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_iq4_xs_q8_K_1x4_avx_vnni(const void * restrict w_raw0,
                                           size_t row_blocks,
                                           const block_q8_K_t * restrict x_q8,
                                           uint32_t n_cols,
                                           float out4[4]) {
    const block_iq4_xs_t * restrict w0 = (const block_iq4_xs_t *)w_raw0;
    const block_iq4_xs_t * restrict w1 = w0 + 1 * row_blocks;
    const block_iq4_xs_t * restrict w2 = w0 + 2 * row_blocks;
    const block_iq4_xs_t * restrict w3 = w0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i lut_u8  = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)kvalues_iq4nl_u8));
    const __m256i vzero   = _mm256_setzero_si256();
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        LFM2_PREFETCH_1X4_NTA(w0, w1, w2, w3, x_q8, b, row_blocks);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;
        int32_t off0 = 0, off1 = 0, off2 = 0, off3 = 0;

        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i x_sub0 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 0) * 32));
            const __m256i x_sub1 = _mm256_loadu_si256((const __m256i *)(q8 + (ib + 1) * 32));
            const int32_t bsum0  = (int32_t)bs[2 * (ib + 0) + 0] + (int32_t)bs[2 * (ib + 0) + 1];
            const int32_t bsum1  = (int32_t)bs[2 * (ib + 1) + 0] + (int32_t)bs[2 * (ib + 1) + 1];

            #define LFM2_IQ4XS_VNNI_ROW(w_ptr, acc_reg, off_acc) do {                                   \
                const int32_t ls0 = get_scale_iq4_xs(ib + 0, (w_ptr)[b].scales_l, (w_ptr)[b].scales_h); \
                const int32_t ls1 = get_scale_iq4_xs(ib + 1, (w_ptr)[b].scales_l, (w_ptr)[b].scales_h); \
                (off_acc) += ls0 * bsum0 + ls1 * bsum1;                                                 \
                const __m256i q4b = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + ib * 16));     \
                const __m256i vlo = _mm256_shuffle_epi8(lut_u8, _mm256_and_si256(q4b, mask_0f));        \
                const __m256i vhi = _mm256_shuffle_epi8(lut_u8, _mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f)); \
                const __m256i wu0 = _mm256_permute2x128_si256(vlo, vhi, 0x20);                          \
                const __m256i wu1 = _mm256_permute2x128_si256(vlo, vhi, 0x31);                          \
                const __m256i d0  = _mm256_dpbusd_epi32(vzero, wu0, x_sub0);                            \
                const __m256i d1  = _mm256_dpbusd_epi32(vzero, wu1, x_sub1);                            \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d0, _mm256_set1_epi32(ls0)));\
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_mullo_epi32(d1, _mm256_set1_epi32(ls1)));\
            } while (0)

            LFM2_IQ4XS_VNNI_ROW(w0, acc0, off0);
            LFM2_IQ4XS_VNNI_ROW(w1, acc1, off1);
            LFM2_IQ4XS_VNNI_ROW(w2, acc2, off2);
            LFM2_IQ4XS_VNNI_ROW(w3, acc3, off3);

            #undef LFM2_IQ4XS_VNNI_ROW
        }

        const __m128i v_off4 = _mm_slli_epi32(_mm_set_epi32(off3, off2, off1, off0), 7);
        const __m128i v_tot  = _mm_sub_epi32(hsum4_epi32_avx2(acc0, acc1, acc2, acc3), v_off4);
        const __m128  v_d    = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            _cvtsh_ss(w3[b].d), _cvtsh_ss(w2[b].d), _cvtsh_ss(w1[b].d), _cvtsh_ss(w0[b].d)));
        v_sumf = _mm_fmadd_ps(v_d, _mm_cvtepi32_ps(v_tot), v_sumf);
    }
    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* 3.5 AVX-VNNI Q4_0 and Q8_0 (1x1 and 1x4 with NTA Prefetching) */
LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_q4_0_q8_K_avx_vnni(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_q4_0_t * restrict w_row = (const block_q4_0_t *)w_raw;
    const uint32_t nb_k   = n_cols / 256;
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m256i vzero   = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        if (b + 1 < nb_k) _mm_prefetch((const char *)&w_row[(b + 1) * 8], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q4_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        const int16_t      *bs    = x_q8[b].bsums;
        float block_sum = 0.0f;

        for (int j = 0; j < 8; ++j) {
            const float dw      = _cvtsh_ss(w_sub[j].d);
            const __m128i q16   = _mm_loadu_si128((const __m128i *)w_sub[j].qs);
            const __m256i u4    = mm256_set_m128i_avx2(
                _mm_and_si128(_mm_srli_epi16(q16, 4), mask_0f), _mm_and_si128(q16, mask_0f));
            const __m256i vx    = _mm256_loadu_si256((const __m256i *)(q8 + j * 32));
            const __m256i dot   = _mm256_dpbusd_epi32(vzero, u4, vx);
            const int32_t sum_q8 = (int32_t)bs[2 * j + 0] + (int32_t)bs[2 * j + 1];
            block_sum += dw * (float)(hsum_epi32_avx2(dot) - 8 * sum_q8);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_q4_0_q8_K_1x4_avx_vnni(const void * restrict w_raw0, size_t rb,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols, float out4[4]) {
    const block_q4_0_t *w = (const block_q4_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q4_0_q8_K_avx_vnni(w + k * rb, x_q8, n_cols);
}

LFM2_TARGET_AVXVNNI
float lfm2_vec_dot_q8_0_q8_K_avx_vnni(const void * restrict w_raw,
                                      const block_q8_K_t * restrict x_q8,
                                      uint32_t n_cols) {
    const block_q8_0_t * restrict w_row = (const block_q8_0_t *)w_raw;
    const uint32_t nb_k    = n_cols / 256;
    const __m256i bias_128 = _mm256_set1_epi8((char)0x80);
    const __m256i vzero    = _mm256_setzero_si256();
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        if (b + 1 < nb_k) _mm_prefetch((const char *)&w_row[(b + 1) * 8], _MM_HINT_NTA);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q8_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        const int16_t      *bs    = x_q8[b].bsums;
        float block_sum = 0.0f;

        for (int j = 0; j < 8; ++j) {
            const float dw       = _cvtsh_ss(w_sub[j].d);
            /* Bias signed int8 weight by +128 (XOR 0x80) into unsigned uint8 for vpdpbusd */
            const __m256i wu8    = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)w_sub[j].qs), bias_128);
            const __m256i vx     = _mm256_loadu_si256((const __m256i *)(q8 + j * 32));
            const __m256i dot    = _mm256_dpbusd_epi32(vzero, wu8, vx);
            const int32_t sum_q8 = (int32_t)bs[2 * j + 0] + (int32_t)bs[2 * j + 1];
            block_sum += dw * (float)(hsum_epi32_avx2(dot) - 128 * sum_q8);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

LFM2_TARGET_AVXVNNI
void lfm2_vec_dot_q8_0_q8_K_1x4_avx_vnni(const void * restrict w_raw0, size_t rb,
                                         const block_q8_K_t * restrict x_q8,
                                         uint32_t n_cols, float out4[4]) {
    const block_q8_0_t *w = (const block_q8_0_t *)w_raw0;
    for (int k = 0; k < 4; ++k) out4[k] = lfm2_vec_dot_q8_0_q8_K_avx_vnni(w + k * rb, x_q8, n_cols);
}
#endif /* __x86_64__ || _M_X64 */

/* ============================================================================
 * 5. Runtime CPUID Inspector & Multi-ISA Kernel Dispatch Table
 * ============================================================================ */
#ifndef LFM2_KERNEL_FN_TYPES_DEFINED
#define LFM2_KERNEL_FN_TYPES_DEFINED
typedef void  (*lfm2_quantize_q8_K_fn)(const float * restrict x,
                                       block_q8_K_t * restrict y,
                                       uint32_t n_cols);

typedef float (*lfm2_vec_dot_1x1_fn)(const void * restrict w_row,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols);

typedef void  (*lfm2_vec_dot_1x4_fn)(const void * restrict w_row0,
                                     size_t row_blocks,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols,
                                     float out4[4]);
#endif

typedef struct {
    const char *isa_name;
    char        cpu_brand[64];
    bool        has_sse41;
    bool        has_avx;
    bool        has_f16c;
    bool        has_fma;
    bool        has_avx2;
    bool        has_avx_vnni;

    lfm2_quantize_q8_K_fn quantize_q8_K;
    lfm2_vec_dot_1x1_fn   vec_dot_1x1[GGML_TYPE_COUNT];
    lfm2_vec_dot_1x4_fn   vec_dot_1x4[GGML_TYPE_COUNT];
} LFM2KernelTable;

#if defined(__x86_64__) || defined(_M_X64)
static inline void query_x86_brand_string(char out_brand[64]) {
    uint32_t regs[12] = {0};
    uint32_t max_ext = 0, unused = 0;
    __get_cpuid(0x80000000u, &max_ext, &unused, &unused, &unused);
    if (max_ext >= 0x80000004u) {
        __get_cpuid(0x80000002u, &regs[0], &regs[1], &regs[2],  &regs[3]);
        __get_cpuid(0x80000003u, &regs[4], &regs[5], &regs[6],  &regs[7]);
        __get_cpuid(0x80000004u, &regs[8], &regs[9], &regs[10], &regs[11]);
        memcpy(out_brand, regs, 48);
        out_brand[48] = '\0';
    } else {
        snprintf(out_brand, 64, "Generic x86_64 CPU");
    }
}
#endif

/* Queries hardware CPUID / HWCAP at startup and populates the optimal kernel pointers */
static inline void lfm2_init_cpu_kernels(LFM2KernelTable *kt) {
    memset(kt, 0, sizeof(*kt));

#if defined(__x86_64__) || defined(_M_X64)
    query_x86_brand_string(kt->cpu_brand);

    __builtin_cpu_init();
    kt->has_sse41 = __builtin_cpu_supports("sse4.1");
    kt->has_avx   = __builtin_cpu_supports("avx");
    kt->has_f16c  = __builtin_cpu_supports("f16c");
    kt->has_fma   = __builtin_cpu_supports("fma");
    kt->has_avx2  = __builtin_cpu_supports("avx2");

    /* Check CPUID Leaf 7 Subleaf 1 EAX bit 4 for AVX-VNNI */
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        kt->has_avx_vnni = (eax & (1u << 4)) != 0;
    }

    /* Allow user/test override via LFM2_FORCE_ISA=scalar */
    const char *force_isa = getenv("LFM2_FORCE_ISA");
    bool use_avx2 = (kt->has_avx2 && kt->has_fma && kt->has_f16c);
    if (force_isa && strcmp(force_isa, "scalar") == 0) {
        use_avx2 = false;
    }

    const bool use_vnni = use_avx2 && kt->has_avx_vnni &&
                          !(force_isa && strcmp(force_isa, "avx2") == 0);

    if (use_vnni) {
        kt->isa_name                            = "AVX-VNNI (vpdpbusd) + FMA + F16C + NTA";
        kt->quantize_q8_K                       = lfm2_quantize_row_q8_K_avx2_cpuid;
        kt->vec_dot_1x1[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_avx_vnni;
        kt->vec_dot_1x1[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_avx_vnni;
        kt->vec_dot_1x1[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_avx_vnni;
        kt->vec_dot_1x1[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_avx_vnni;
        kt->vec_dot_1x1[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_avx_vnni;
        kt->vec_dot_1x1[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_avx_vnni;

        kt->vec_dot_1x4[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_1x4_avx_vnni;
        kt->vec_dot_1x4[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_1x4_avx_vnni;
        kt->vec_dot_1x4[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_1x4_avx_vnni;
        kt->vec_dot_1x4[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_1x4_avx_vnni;
        kt->vec_dot_1x4[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_1x4_avx_vnni;
        kt->vec_dot_1x4[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_1x4_avx_vnni;
    } else if (use_avx2) {
        kt->isa_name                            = "AVX2 + FMA + F16C + NTA";
        kt->quantize_q8_K                       = lfm2_quantize_row_q8_K_avx2_cpuid;
        kt->vec_dot_1x1[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_avx2;
        kt->vec_dot_1x1[GGML_TYPE_Q8_0]         = (lfm2_vec_dot_1x1_fn)lfm2_vec_dot_q8_0_q8_K_avx2;
        kt->vec_dot_1x1[GGML_TYPE_Q4_K]         = (lfm2_vec_dot_1x1_fn)lfm2_vec_dot_q4_K_q8_K_avx2;
        kt->vec_dot_1x1[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_avx2;
        kt->vec_dot_1x1[GGML_TYPE_Q6_K]         = (lfm2_vec_dot_1x1_fn)lfm2_vec_dot_q6_K_q8_K_avx2;
        kt->vec_dot_1x1[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_avx2;

        kt->vec_dot_1x4[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_1x4_avx2;
        kt->vec_dot_1x4[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_1x4_avx2;
        kt->vec_dot_1x4[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_1x4_avx2_nta;
        kt->vec_dot_1x4[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_1x4_avx2;
        kt->vec_dot_1x4[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_1x4_avx2_nta;
        kt->vec_dot_1x4[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_1x4_avx2;
    } else {
        kt->isa_name                            = "GENERIC_SCALAR_X86_64";
        kt->quantize_q8_K                       = lfm2_quantize_row_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_scalar;
        kt->vec_dot_1x1[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_scalar;

        kt->vec_dot_1x4[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_1x4_scalar;
        kt->vec_dot_1x4[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_1x4_scalar;
        kt->vec_dot_1x4[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_1x4_scalar;
        kt->vec_dot_1x4[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_1x4_scalar;
        kt->vec_dot_1x4[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_1x4_scalar;
        kt->vec_dot_1x4[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_1x4_scalar;
    }
#elif defined(__aarch64__)
    LFM2MobileKernelTable mkt;
    lfm2_init_mobile_kernels(&mkt);
    const char *force_isa = getenv("LFM2_FORCE_ISA");
    const bool forced_scalar = (force_isa && strcmp(force_isa, "scalar") == 0);

    kt->isa_name = forced_scalar ? "GENERIC_SCALAR_ARM64" : mkt.isa_name;
    snprintf(kt->cpu_brand, sizeof(kt->cpu_brand), "ARM64 (%s)", kt->isa_name);
    kt->quantize_q8_K = mkt.quantize_q8_K;

    kt->vec_dot_1x1[GGML_TYPE_Q4_0]   = mkt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_0];
    kt->vec_dot_1x1[GGML_TYPE_Q8_0]   = mkt.vec_dot_1x1[LFM2_GGML_TYPE_Q8_0];
    kt->vec_dot_1x1[GGML_TYPE_Q4_K]   = mkt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_K];
    kt->vec_dot_1x1[GGML_TYPE_Q5_K]   = mkt.vec_dot_1x1[LFM2_GGML_TYPE_Q5_K];
    kt->vec_dot_1x1[GGML_TYPE_Q6_K]   = mkt.vec_dot_1x1[LFM2_GGML_TYPE_Q6_K];
    kt->vec_dot_1x1[GGML_TYPE_IQ4_XS] = mkt.vec_dot_1x1[LFM2_GGML_TYPE_IQ4_XS];

    kt->vec_dot_1x4[GGML_TYPE_Q4_0]   = mkt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_0];
    kt->vec_dot_1x4[GGML_TYPE_Q8_0]   = mkt.vec_dot_1x4[LFM2_GGML_TYPE_Q8_0];
    kt->vec_dot_1x4[GGML_TYPE_Q4_K]   = mkt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_K];
    kt->vec_dot_1x4[GGML_TYPE_Q5_K]   = mkt.vec_dot_1x4[LFM2_GGML_TYPE_Q5_K];
    kt->vec_dot_1x4[GGML_TYPE_Q6_K]   = mkt.vec_dot_1x4[LFM2_GGML_TYPE_Q6_K];
    kt->vec_dot_1x4[GGML_TYPE_IQ4_XS] = mkt.vec_dot_1x4[LFM2_GGML_TYPE_IQ4_XS];
#else
    snprintf(kt->cpu_brand, sizeof(kt->cpu_brand), "Generic Scalar CPU");
    kt->isa_name                            = "GENERIC_SCALAR";
    kt->quantize_q8_K                       = lfm2_quantize_row_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_scalar;
    kt->vec_dot_1x1[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_scalar;

    kt->vec_dot_1x4[GGML_TYPE_Q4_0]         = lfm2_vec_dot_q4_0_q8_K_1x4_scalar;
    kt->vec_dot_1x4[GGML_TYPE_Q8_0]         = lfm2_vec_dot_q8_0_q8_K_1x4_scalar;
    kt->vec_dot_1x4[GGML_TYPE_Q4_K]         = lfm2_vec_dot_q4_K_q8_K_1x4_scalar;
    kt->vec_dot_1x4[GGML_TYPE_Q5_K]         = lfm2_vec_dot_q5_K_q8_K_1x4_scalar;
    kt->vec_dot_1x4[GGML_TYPE_Q6_K]         = lfm2_vec_dot_q6_K_q8_K_1x4_scalar;
    kt->vec_dot_1x4[GGML_TYPE_IQ4_XS]       = lfm2_vec_dot_iq4_xs_q8_K_1x4_scalar;
#endif
}

/* Global singleton kernel table accessor */
static inline const LFM2KernelTable *lfm2_get_cpu_kernel_table(void) {
    static LFM2KernelTable g_kt;
    static bool g_initialized = false;
    if (!g_initialized) {
        lfm2_init_cpu_kernels(&g_kt);
        g_initialized = true;
    }
    return &g_kt;
}

#endif /* LFM2_CPUID_QUANTS_INCLUDED */

/* ============================================================================
 * 6. Standalone Verification & Bandwidth Benchmark
 * ============================================================================ */
#ifndef LFM2_CPUID_QUANTS_NO_MAIN
static double cpuid_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(void) {
    LFM2KernelTable kt;
    lfm2_init_cpu_kernels(&kt);

    printf("=== LFM2 Runtime CPUID Dispatcher & Quantized Kernel Verifier ===\n");
    printf("Detected CPU      : %s\n", kt.cpu_brand);
    printf("Hardware Flags    : SSE4.1=%d | AVX=%d | F16C=%d | FMA=%d | AVX2=%d | AVX-VNNI=%d\n",
           kt.has_sse41, kt.has_avx, kt.has_f16c, kt.has_fma, kt.has_avx2, kt.has_avx_vnni);
    printf("Active Kernel ISA : %s\n\n", kt.isa_name);

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 4096;
    const uint32_t nb_k   = n_cols / 256;
    const uint32_t nb_32  = n_cols / 32;

    float          *x       = (float *)lfm2_cpuid_alloc_64(n_cols * sizeof(float));
    block_q8_K_t   *x_q8    = (block_q8_K_t *)lfm2_cpuid_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q4_0_t   *w_q4_0  = (block_q4_0_t *)lfm2_cpuid_alloc_64((size_t)n_rows * nb_32 * sizeof(block_q4_0_t));
    block_q5_K_t   *w_q5_k  = (block_q5_K_t *)lfm2_cpuid_alloc_64((size_t)n_rows * nb_k  * sizeof(block_q5_K_t));
    block_iq4_xs_t *w_iq4   = (block_iq4_xs_t *)lfm2_cpuid_alloc_64((size_t)n_rows * nb_k * sizeof(block_iq4_xs_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.017f);

    for (size_t i = 0; i < (size_t)n_rows * nb_32; ++i) {
        w_q4_0[i].d = 0x2E00;
        for (int j = 0; j < 16; ++j) w_q4_0[i].qs[j] = (uint8_t)((i * 13 + j * 29) & 0xFF);
    }
    for (size_t i = 0; i < (size_t)n_rows * nb_k; ++i) {
        w_q5_k[i].d    = 0x2E00;
        w_q5_k[i].dmin = 0x2600;
        for (int j = 0; j < 12; ++j)  w_q5_k[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
        for (int j = 0; j < 32; ++j)  w_q5_k[i].qh[j]     = (uint8_t)((i * 11 + j * 19) & 0xFF);
        for (int j = 0; j < 128; ++j) w_q5_k[i].qs[j]     = (uint8_t)((i * 17 + j * 31) & 0xFF);

        w_iq4[i].d        = 0x2800;
        w_iq4[i].scales_h = (uint16_t)((i * 73) & 0xFFFF);
        for (int j = 0; j < 4; ++j)   w_iq4[i].scales_l[j] = (uint8_t)((i * 19 + j * 41) & 0xFF);
        for (int j = 0; j < 128; ++j) w_iq4[i].qs[j]       = (uint8_t)((i * 23 + j * 37) & 0xFF);
    }

    kt.quantize_q8_K(x, x_q8, n_cols);

    /* Verify exact bit-for-bit integer accumulator parity: Scalar vs AVX2 1x4 */
    float out4_q4_0[4], out4_q5_k[4], out4_iq4[4];
    kt.vec_dot_1x4[GGML_TYPE_Q4_0](w_q4_0, nb_32, x_q8, n_cols, out4_q4_0);
    kt.vec_dot_1x4[GGML_TYPE_Q5_K](w_q5_k, nb_k,  x_q8, n_cols, out4_q5_k);
    kt.vec_dot_1x4[GGML_TYPE_IQ4_XS](w_iq4, nb_k, x_q8, n_cols, out4_iq4);

    float err_q4_0 = 0.0f, err_q5_k = 0.0f, err_iq4 = 0.0f;
    for (int r = 0; r < 4; ++r) {
        float ref0 = lfm2_vec_dot_q4_0_q8_K_scalar(w_q4_0 + r * nb_32, x_q8, n_cols);
        float ref5 = lfm2_vec_dot_q5_K_q8_K_scalar(w_q5_k + r * nb_k,  x_q8, n_cols);
        float refi = lfm2_vec_dot_iq4_xs_q8_K_scalar(w_iq4 + r * nb_k, x_q8, n_cols);
        if (fabsf(ref0 - out4_q4_0[r]) > err_q4_0) err_q4_0 = fabsf(ref0 - out4_q4_0[r]);
        if (fabsf(ref5 - out4_q5_k[r]) > err_q5_k) err_q5_k = fabsf(ref5 - out4_q5_k[r]);
        if (fabsf(refi - out4_iq4[r])  > err_iq4)  err_iq4  = fabsf(refi - out4_iq4[r]);
    }

    printf("1. Q4_0   (1x4 vs Scalar Ref) Max Diff : %.8f\n", err_q4_0);
    printf("2. Q5_K   (1x4 vs Scalar Ref) Max Diff : %.8f\n", err_q5_k);
    printf("3. IQ4_XS (1x4 vs Scalar Ref) Max Diff : %.8f\n\n", err_iq4);

    /* Benchmark Single-Core 1x4 Throughput across [4096 x 2048] */
    float *dst = (float *)lfm2_cpuid_alloc_64(n_rows * sizeof(float));
    const int iters = 100;

    double t0 = cpuid_now_ms();
    for (int it = 0; it < iters; ++it) {
        for (uint32_t r = 0; r < n_rows; r += 4) {
            kt.vec_dot_1x4[GGML_TYPE_Q4_0](w_q4_0 + (size_t)r * nb_32, nb_32, x_q8, n_cols, dst + r);
        }
    }
    double ms_q4_0 = (cpuid_now_ms() - t0) / iters;

    t0 = cpuid_now_ms();
    for (int it = 0; it < iters; ++it) {
        for (uint32_t r = 0; r < n_rows; r += 4) {
            kt.vec_dot_1x4[GGML_TYPE_Q5_K](w_q5_k + (size_t)r * nb_k, nb_k, x_q8, n_cols, dst + r);
        }
    }
    double ms_q5_k = (cpuid_now_ms() - t0) / iters;

    t0 = cpuid_now_ms();
    for (int it = 0; it < iters; ++it) {
        for (uint32_t r = 0; r < n_rows; r += 4) {
            kt.vec_dot_1x4[GGML_TYPE_IQ4_XS](w_iq4 + (size_t)r * nb_k, nb_k, x_q8, n_cols, dst + r);
        }
    }
    double ms_iq4 = (cpuid_now_ms() - t0) / iters;

    printf("Single-Core 1x4 [4096 x 2048] MatMul Throughput:\n");
    printf("  Q4_0   : %.3f ms | %.2f GB/s\n",
           ms_q4_0, ((double)n_rows * nb_32 * sizeof(block_q4_0_t)) / (ms_q4_0 * 1e6));
    printf("  Q5_K   : %.3f ms | %.2f GB/s\n",
           ms_q5_k, ((double)n_rows * nb_k  * sizeof(block_q5_K_t)) / (ms_q5_k * 1e6));
    printf("  IQ4_XS : %.3f ms | %.2f GB/s\n",
           ms_iq4,  ((double)n_rows * nb_k  * sizeof(block_iq4_xs_t)) / (ms_iq4 * 1e6));

    free(x); free(x_q8); free(w_q4_0); free(w_q5_k); free(w_iq4); free(dst);
    return EXIT_SUCCESS;
}
#endif
