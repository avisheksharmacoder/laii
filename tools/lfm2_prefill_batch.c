#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef LFM2_PREFILL_BATCH_INCLUDED
#define LFM2_PREFILL_BATCH_INCLUDED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#ifndef LFM_DEFAULTS_H
#if defined(__has_include)
#  if __has_include("lfm_defaults.h")
#    include "lfm_defaults.h"
#  elif __has_include("include/lfm_defaults.h")
#    include "include/lfm_defaults.h"
#  elif __has_include("../include/lfm_defaults.h")
#    include "../include/lfm_defaults.h"
#  endif
#else
#  include "lfm_defaults.h"
#endif
#endif

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1

#if __has_include("tools/lfm2_forward.c")
#include "tools/lfm2_forward.c"
#elif __has_include("lfm2_forward.c")
#include "lfm2_forward.c"
#else
#include "../tools/lfm2_forward.c"
#endif

#ifndef LFM2_CPUID_QUANTS_INCLUDED
#define LFM2_CPUID_QUANTS_NO_MAIN 1
#if __has_include("tools/lfm2_cpuid_quants.c")
#include "tools/lfm2_cpuid_quants.c"
#elif __has_include("lfm2_cpuid_quants.c")
#include "lfm2_cpuid_quants.c"
#else
#include "../tools/lfm2_cpuid_quants.c"
#endif
#endif

#ifndef fp16_to_fp32
#define fp16_to_fp32 lfm2_fp16_to_fp32
#endif
#ifndef get_scale_min_k4
#define get_scale_min_k4 lfm2_get_scale_min_k4
#endif
#if LFM2_HAS_AVX2
#ifndef hsum_epi32_avx2
#define hsum_epi32_avx2 lfm2_hsum_epi32_avx2
#endif
#ifndef mm256_set_m128i
#define mm256_set_m128i lfm2_mm256_set_m128i
#endif
#endif


static inline void *lfm2_aligned_alloc(size_t align, size_t size) {
    size_t rounded = (size + align - 1) & ~(align - 1);
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, rounded) != 0) {
        return NULL;
    }
    memset(ptr, 0, rounded);
    return ptr;
}

/* Maximum micro-batch size for prompt prefill (8 tokens fits in L1 cache & YMM regs) */
#define LFM2_MAX_BATCH 8

/* ============================================================================
 * 1. Batched Q8_K Quantizer: Quantizes B Vectors [B x n_cols] at Once
 * ============================================================================ */
void lfm2_quantize_batch_q8_K_avx2(const float * restrict x_batch,
                                   block_q8_K_t * restrict y_batch,
                                   uint32_t B,
                                   uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    for (uint32_t t = 0; t < B; ++t) {
        kt->quantize_q8_K(x_batch + (size_t)t * n_cols,
                          y_batch + (size_t)t * nb,
                          n_cols);
    }
}

#if LFM2_HAS_AVX2
/* ============================================================================
 * 2. Batched AVX2 Q4_K x Q8_K GEMM Micro-Kernel (Unpack Weight Once for B Tokens)
 * ============================================================================
 * Computes dot product of ONE weight row `w_row` against `B` (1..8) quantized
 * token vectors in `x_q8_batch[t * nb + b]` and writes `out[t * n_rows + r]`.
 * ============================================================================ */
static inline void lfm2_gemm_row_q4_K_q8_K_avx2(const block_q4_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8_batch,
                                                uint32_t B,
                                                uint32_t n_cols,
                                                uint32_t n_rows,
                                                uint32_t row_idx,
                                                float * restrict out_batch) {
    const uint32_t nb = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    float sumf[LFM2_MAX_BATCH] = {0.0f};

    for (uint32_t b = 0; b < nb; ++b) {
        _mm_prefetch((const char *)(w_row + b + 2), _MM_HINT_T0);

        /* 1. Decode super-block FP16 scale & min ONCE for all B tokens */
        const float d_super    = fp16_to_fp32(w_row[b].d);
        const float dmin_super = fp16_to_fp32(w_row[b].dmin);
        const uint8_t * restrict q4 = w_row[b].qs;

        __m256i acc_i32[LFM2_MAX_BATCH];
        int32_t sum_mins[LFM2_MAX_BATCH];
        for (uint32_t t = 0; t < B; ++t) {
            acc_i32[t]  = _mm256_setzero_si256();
            sum_mins[t] = 0;
        }

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            /* 2. Unpack 6-bit sub-block scales & mins ONCE for all B tokens */
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            const __m256i v_sc1 = _mm256_set1_epi16((int16_t)sc1);
            const __m256i v_sc2 = _mm256_set1_epi16((int16_t)sc2);

            /* 3. Load & unpack 64 4-bit weights ONCE for all B tokens */
            const __m256i q4_bytes = _mm256_loadu_si256((const __m256i *)q4);
            const __m256i q4_lo    = _mm256_and_si256(q4_bytes, mask_0f);
            const __m256i q4_hi    = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), mask_0f);

            /* 4. Multiply unpacked weights against each token t in the batch */
            for (uint32_t t = 0; t < B; ++t) {
                const block_q8_K_t *xb_q8 = &x_q8_batch[(size_t)t * nb + b];
                const int16_t * restrict bs = xb_q8->bsums;

                sum_mins[t] += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                             + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(xb_q8->qs + j +  0));
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(xb_q8->qs + j + 32));

                const __m256i dot_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
                const __m256i dot_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);

                acc_i32[t] = _mm256_add_epi32(acc_i32[t], _mm256_madd_epi16(dot_lo, v_sc1));
                acc_i32[t] = _mm256_add_epi32(acc_i32[t], _mm256_madd_epi16(dot_hi, v_sc2));
            }

            q4 += 32;
            is += 2;
        }

        /* 5. Reduce integer accumulators and scale by each token's d8 */
        for (uint32_t t = 0; t < B; ++t) {
            const float d8 = x_q8_batch[(size_t)t * nb + b].d;
            if (d8 != 0.0f) {
                const int32_t sum_q4q8 = hsum_epi32_avx2(acc_i32[t]);
                const float d    = d_super    * d8;
                const float dmin = dmin_super * d8;
                sumf[t] += d * (float)sum_q4q8 - dmin * (float)sum_mins[t];
            }
        }
    }

    for (uint32_t t = 0; t < B; ++t) {
        out_batch[(size_t)t * n_rows + row_idx] = sumf[t];
    }
}

/* ============================================================================
 * 3. Batched AVX2 Q6_K x Q8_K GEMM Micro-Kernel (Unpack 6-Bit Weights Once)
 * ============================================================================ */
static inline void lfm2_gemm_row_q6_K_q8_K_avx2(const block_q6_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8_batch,
                                                uint32_t B,
                                                uint32_t n_cols,
                                                uint32_t n_rows,
                                                uint32_t row_idx,
                                                float * restrict out_batch) {
    const uint32_t nb = n_cols / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    const __m256i mask_0c = _mm256_set1_epi8(0x0C);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);

    float sumf[LFM2_MAX_BATCH] = {0.0f};

    for (uint32_t b = 0; b < nb; ++b) {
        _mm_prefetch((const char *)(w_row + b + 2), _MM_HINT_T0);

        const float d_super = fp16_to_fp32(w_row[b].d);
        const uint8_t * restrict ql = w_row[b].ql;
        const uint8_t * restrict qh = w_row[b].qh;
        const int8_t  * restrict sc = w_row[b].scales;

        int32_t sum_offset[LFM2_MAX_BATCH] = {0};
        __m256i acc_i32[LFM2_MAX_BATCH];

        for (uint32_t t = 0; t < B; ++t) {
            acc_i32[t] = _mm256_setzero_si256();
            const int16_t * restrict bs = x_q8_batch[(size_t)t * nb + b].bsums;
            int32_t off = 0;
            for (int is = 0; is < 16; ++is) {
                off += (int32_t)sc[is] * (int32_t)bs[is];
            }
            sum_offset[t] = off;
        }

        for (int half = 0; half < 2; ++half) {
            /* Unpack 128 unsigned 6-bit weights [0..63] and their scales ONCE */
            const __m256i ql0 = _mm256_loadu_si256((const __m256i *)(ql +  0));
            const __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);

            const __m256i u1 = _mm256_or_si256(
                _mm256_and_si256(ql0, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qhv, mask_03), 4));
            const __m256i u2 = _mm256_or_si256(
                _mm256_and_si256(ql1, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qhv, mask_0c), 2));
            const __m256i u3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_0f),
                _mm256_and_si256(qhv, mask_30));
            const __m256i u4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask_30));

            const __m256i sc01 = mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            const __m256i sc23 = mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            const __m256i sc45 = mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            const __m256i sc67 = mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));

            /* Multiply the unpacked 128 weights across all B tokens in L1 cache */
            const int half_off = half * 128;
            for (uint32_t t = 0; t < B; ++t) {
                const int8_t * restrict q8 = x_q8_batch[(size_t)t * nb + b].qs + half_off;

                const __m256i x0 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i x1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i x2 = _mm256_loadu_si256((const __m256i *)(q8 + 64));
                const __m256i x3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));

                const __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, x0), sc01);
                const __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u2, x1), sc23);
                const __m256i d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(u3, x2), sc45);
                const __m256i d3 = _mm256_madd_epi16(_mm256_maddubs_epi16(u4, x3), sc67);

                acc_i32[t] = _mm256_add_epi32(acc_i32[t],
                    _mm256_add_epi32(_mm256_add_epi32(d0, d1), _mm256_add_epi32(d2, d3)));
            }

            ql += 64;
            qh += 32;
            sc += 8;
        }

        for (uint32_t t = 0; t < B; ++t) {
            const float d8 = x_q8_batch[(size_t)t * nb + b].d;
            if (d8 != 0.0f) {
                const int32_t sum_u6q8 = hsum_epi32_avx2(acc_i32[t]);
                sumf[t] += (d_super * d8) * (float)(sum_u6q8 - 32 * sum_offset[t]);
            }
        }
    }

    for (uint32_t t = 0; t < B; ++t) {
        out_batch[(size_t)t * n_rows + row_idx] = sumf[t];
    }
}
#elif defined(__aarch64__)

/* ============================================================================
 * 2. Batched ARM64 DotProd Q4_K x Q8_K GEMM Micro-Kernel
 * ============================================================================ */
LFM2_TARGET_ARM_DOTPROD
static inline void lfm2_gemm_row_q4_K_q8_K_neon_dp(const block_q4_K_t * restrict w_row,
                                                   const block_q8_K_t * restrict x_q8_batch,
                                                   uint32_t B,
                                                   uint32_t n_cols,
                                                   uint32_t n_rows,
                                                   uint32_t row_idx,
                                                   float * restrict out_batch) {
    const uint32_t nb = n_cols / 256;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const int32x4_t  vzero   = vdupq_n_s32(0);
    float sumf[LFM2_MAX_BATCH] = {0.0f};

    for (uint32_t b = 0; b < nb; ++b) {
        const float d_super    = fp16_to_fp32_arm(w_row[b].d);
        const float dmin_super = fp16_to_fp32_arm(w_row[b].dmin);
        const uint8_t * restrict q4 = w_row[b].qs;

        int32x4_t acc_i32[LFM2_MAX_BATCH];
        int32_t   sum_mins[LFM2_MAX_BATCH];
        for (uint32_t t = 0; t < B; ++t) {
            acc_i32[t]  = vzero;
            sum_mins[t] = 0;
        }

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            const uint8x16_t q4_0 = vld1q_u8(q4 +  0);
            const uint8x16_t q4_1 = vld1q_u8(q4 + 16);
            const int8x16_t u_lo0 = vreinterpretq_s8_u8(vandq_u8(q4_0, mask_0f));
            const int8x16_t u_lo1 = vreinterpretq_s8_u8(vandq_u8(q4_1, mask_0f));
            const int8x16_t u_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_0, 4));
            const int8x16_t u_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_1, 4));

            for (uint32_t t = 0; t < B; ++t) {
                const block_q8_K_t *xb_q8 = &x_q8_batch[(size_t)t * nb + b];
                const int16_t * restrict bs = xb_q8->bsums;
                sum_mins[t] += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                             + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

                const int8_t * restrict q8 = xb_q8->qs + j;
                const int32x4_t dot_lo = vdotq_s32(
                    vdotq_s32(vzero, u_lo0, vld1q_s8(q8 +  0)), u_lo1, vld1q_s8(q8 + 16));
                const int32x4_t dot_hi = vdotq_s32(
                    vdotq_s32(vzero, u_hi0, vld1q_s8(q8 + 32)), u_hi1, vld1q_s8(q8 + 48));

                acc_i32[t] = vmlaq_n_s32(acc_i32[t], dot_lo, (int32_t)sc1);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], dot_hi, (int32_t)sc2);
            }

            q4 += 32;
            is += 2;
        }

        for (uint32_t t = 0; t < B; ++t) {
            const float d8 = x_q8_batch[(size_t)t * nb + b].d;
            if (d8 != 0.0f) {
                const int32_t sum_q4q8 = vaddvq_s32(acc_i32[t]);
                sumf[t] += (d_super * d8) * (float)sum_q4q8 - (dmin_super * d8) * (float)sum_mins[t];
            }
        }
    }

    for (uint32_t t = 0; t < B; ++t) {
        out_batch[(size_t)t * n_rows + row_idx] = sumf[t];
    }
}

/* ============================================================================
 * 3. Batched ARM64 DotProd Q6_K x Q8_K GEMM Micro-Kernel
 * ============================================================================ */
LFM2_TARGET_ARM_DOTPROD
static inline void lfm2_gemm_row_q6_K_q8_K_neon_dp(const block_q6_K_t * restrict w_row,
                                                   const block_q8_K_t * restrict x_q8_batch,
                                                   uint32_t B,
                                                   uint32_t n_cols,
                                                   uint32_t n_rows,
                                                   uint32_t row_idx,
                                                   float * restrict out_batch) {
    const uint32_t nb = n_cols / 256;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask_30 = vdupq_n_u8(0x30);
    const int32x4_t  vzero   = vdupq_n_s32(0);
    float sumf[LFM2_MAX_BATCH] = {0.0f};

    for (uint32_t b = 0; b < nb; ++b) {
        const float d_super = fp16_to_fp32_arm(w_row[b].d);
        const uint8_t * restrict ql = w_row[b].ql;
        const uint8_t * restrict qh = w_row[b].qh;
        const int8_t  * restrict sc = w_row[b].scales;

        int32_t sum_offset[LFM2_MAX_BATCH] = {0};
        int32x4_t acc_i32[LFM2_MAX_BATCH];

        for (uint32_t t = 0; t < B; ++t) {
            acc_i32[t] = vzero;
            const int16_t * restrict bs = x_q8_batch[(size_t)t * nb + b].bsums;
            int32_t off = 0;
            for (int is = 0; is < 16; ++is) off += (int32_t)sc[is] * (int32_t)bs[is];
            sum_offset[t] = off;
        }

        for (int half = 0; half < 2; ++half) {
            const uint8x16_t ql0_0 = vld1q_u8(ql +  0), ql0_1 = vld1q_u8(ql + 16);
            const uint8x16_t ql1_0 = vld1q_u8(ql + 32), ql1_1 = vld1q_u8(ql + 48);
            const uint8x16_t qh_0  = vld1q_u8(qh +  0), qh_1  = vld1q_u8(qh + 16);

            const int8x16_t u0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0_0, mask_0f), vandq_u8(vshlq_n_u8(qh_0, 4), mask_30)));
            const int8x16_t u1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0_1, mask_0f), vandq_u8(vshlq_n_u8(qh_1, 4), mask_30)));
            const int8x16_t u2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1_0, mask_0f), vandq_u8(vshlq_n_u8(qh_0, 2), mask_30)));
            const int8x16_t u3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1_1, mask_0f), vandq_u8(vshlq_n_u8(qh_1, 2), mask_30)));
            const int8x16_t u4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0_0, 4),  vandq_u8(qh_0, mask_30)));
            const int8x16_t u5 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0_1, 4),  vandq_u8(qh_1, mask_30)));
            const int8x16_t u6 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1_0, 4),  vandq_u8(vshrq_n_u8(qh_0, 2), mask_30)));
            const int8x16_t u7 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1_1, 4),  vandq_u8(vshrq_n_u8(qh_1, 2), mask_30)));

            const int half_off = half * 128;
            for (uint32_t t = 0; t < B; ++t) {
                const int8_t * restrict q8 = x_q8_batch[(size_t)t * nb + b].qs + half_off;
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u0, vld1q_s8(q8 +   0)), (int32_t)sc[0]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u1, vld1q_s8(q8 +  16)), (int32_t)sc[1]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u2, vld1q_s8(q8 +  32)), (int32_t)sc[2]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u3, vld1q_s8(q8 +  48)), (int32_t)sc[3]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u4, vld1q_s8(q8 +  64)), (int32_t)sc[4]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u5, vld1q_s8(q8 +  80)), (int32_t)sc[5]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u6, vld1q_s8(q8 +  96)), (int32_t)sc[6]);
                acc_i32[t] = vmlaq_n_s32(acc_i32[t], vdotq_s32(vzero, u7, vld1q_s8(q8 + 112)), (int32_t)sc[7]);
            }

            ql += 64;
            qh += 32;
            sc += 8;
        }

        for (uint32_t t = 0; t < B; ++t) {
            const float d8 = x_q8_batch[(size_t)t * nb + b].d;
            if (d8 != 0.0f) {
                const int32_t sum_u6q8 = vaddvq_s32(acc_i32[t]);
                sumf[t] += (d_super * d8) * (float)(sum_u6q8 - 32 * sum_offset[t]);
            }
        }
    }

    for (uint32_t t = 0; t < B; ++t) {
        out_batch[(size_t)t * n_rows + row_idx] = sumf[t];
    }
}

static inline void lfm2_gemm_row_f32_neon(const float * restrict w_row,
                                          const float * restrict x_f32_batch,
                                          uint32_t B,
                                          uint32_t n_cols,
                                          uint32_t n_rows,
                                          uint32_t row_idx,
                                          float * restrict out_batch) {
    float32x4_t acc[LFM2_MAX_BATCH];
    for (uint32_t t = 0; t < B; ++t) acc[t] = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 4 <= n_cols; i += 4) {
        const float32x4_t vw = vld1q_f32(w_row + i);
        for (uint32_t t = 0; t < B; ++t) {
            const float32x4_t vx = vld1q_f32(x_f32_batch + (size_t)t * n_cols + i);
            acc[t] = vfmaq_f32(acc[t], vw, vx);
        }
    }
    for (uint32_t t = 0; t < B; ++t) {
        float sum = vaddvq_f32(acc[t]);
        for (uint32_t k = i; k < n_cols; ++k) {
            sum += w_row[k] * x_f32_batch[(size_t)t * n_cols + k];
        }
        out_batch[(size_t)t * n_rows + row_idx] = sum;
    }
}
#endif

static inline void lfm2_matmul_batch_dispatch_row(const void * restrict w_data,
                                                  uint32_t w_type,
                                                  const float * restrict x_f32_batch,
                                                  const block_q8_K_t * restrict x_q8_batch,
                                                  uint32_t B,
                                                  uint32_t n_cols,
                                                  uint32_t n_rows,
                                                  uint32_t r,
                                                  float * restrict out_batch) {
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    const uint32_t nb_k = n_cols / 256;

    if (w_type == GGML_TYPE_Q4_K) {
        const size_t row_blocks = n_cols / 256;
        const block_q4_K_t *w_row = (const block_q4_K_t *)w_data + (size_t)r * row_blocks;
        if (B == 1) {
            out_batch[r] = kt->vec_dot_1x1[GGML_TYPE_Q4_K](w_row, x_q8_batch, n_cols);
        } else {
#if LFM2_HAS_AVX2
            if (kt->has_avx2 && kt->has_fma && kt->has_f16c &&
                kt->vec_dot_1x1[GGML_TYPE_Q4_K] != lfm2_vec_dot_q4_K_q8_K_scalar) {
                lfm2_gemm_row_q4_K_q8_K_avx2(w_row, x_q8_batch, B, n_cols, n_rows, r, out_batch);
                return;
            }
#elif defined(__aarch64__)
            if (kt->vec_dot_1x1[GGML_TYPE_Q4_K] != lfm2_vec_dot_q4_K_q8_K_scalar) {
                lfm2_gemm_row_q4_K_q8_K_neon_dp(w_row, x_q8_batch, B, n_cols, n_rows, r, out_batch);
                return;
            }
#endif
            for (uint32_t t = 0; t < B; ++t) {
                out_batch[(size_t)t * n_rows + r] =
                    kt->vec_dot_1x1[GGML_TYPE_Q4_K](w_row, x_q8_batch + (size_t)t * nb_k, n_cols);
            }
        }
    } else if (w_type == GGML_TYPE_Q6_K) {
        const size_t row_blocks = n_cols / 256;
        const block_q6_K_t *w_row = (const block_q6_K_t *)w_data + (size_t)r * row_blocks;
        if (B == 1) {
            out_batch[r] = kt->vec_dot_1x1[GGML_TYPE_Q6_K](w_row, x_q8_batch, n_cols);
        } else {
#if LFM2_HAS_AVX2
            if (kt->has_avx2 && kt->has_fma && kt->has_f16c &&
                kt->vec_dot_1x1[GGML_TYPE_Q6_K] != lfm2_vec_dot_q6_K_q8_K_scalar) {
                lfm2_gemm_row_q6_K_q8_K_avx2(w_row, x_q8_batch, B, n_cols, n_rows, r, out_batch);
                return;
            }
#elif defined(__aarch64__)
            if (kt->vec_dot_1x1[GGML_TYPE_Q6_K] != lfm2_vec_dot_q6_K_q8_K_scalar) {
                lfm2_gemm_row_q6_K_q8_K_neon_dp(w_row, x_q8_batch, B, n_cols, n_rows, r, out_batch);
                return;
            }
#endif
            for (uint32_t t = 0; t < B; ++t) {
                out_batch[(size_t)t * n_rows + r] =
                    kt->vec_dot_1x1[GGML_TYPE_Q6_K](w_row, x_q8_batch + (size_t)t * nb_k, n_cols);
            }
        }
    } else if (w_type == GGML_TYPE_Q8_0 || w_type == GGML_TYPE_Q4_0 ||
               w_type == GGML_TYPE_Q5_K || w_type == GGML_TYPE_IQ4_XS) {
        ggml_type_traits_t traits;
        get_ggml_type_traits(w_type, &traits);
        const size_t row_bytes = ((size_t)n_cols / traits.block_elements) * traits.block_bytes;
        const uint8_t *w_row = (const uint8_t *)w_data + (size_t)r * row_bytes;
        for (uint32_t t = 0; t < B; ++t) {
            out_batch[(size_t)t * n_rows + r] =
                kt->vec_dot_1x1[w_type](w_row, x_q8_batch + (size_t)t * nb_k, n_cols);
        }
    } else if (w_type == GGML_TYPE_F32) {
        const float *w_row = (const float *)w_data + (size_t)r * n_cols;
#if defined(__aarch64__)
        lfm2_gemm_row_f32_neon(w_row, x_f32_batch, B, n_cols, n_rows, r, out_batch);
#elif defined(__x86_64__) || defined(_M_X64)
        for (uint32_t t = 0; t < B; ++t) {
            out_batch[(size_t)t * n_rows + r] =
                lfm2_vec_dot_f32_avx2(w_row, x_f32_batch + (size_t)t * n_cols, n_cols);
        }
#else
        for (uint32_t t = 0; t < B; ++t) {
            float sum = 0.0f;
            const float *xb = x_f32_batch + (size_t)t * n_cols;
            for (uint32_t k = 0; k < n_cols; ++k) sum += w_row[k] * xb[k];
            out_batch[(size_t)t * n_rows + r] = sum;
        }
#endif
    }
}

/* ============================================================================
 * 4. Multi-Threaded Batched GEMM Dispatcher (Spin-Barrier Pool)
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *out_batch; /* [B, n_rows] */
    const void         *w_data;
    uint32_t            w_type;
    uint32_t            B;
    uint32_t            n_cols;
    uint32_t            n_rows;
    const float        *x_f32_batch;
    const block_q8_K_t *x_q8_batch;
} BatchMatMulTaskCtx;

static void lfm2_matmul_batch_worker_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    BatchMatMulTaskCtx *ctx  = (BatchMatMulTaskCtx *)arg;
    LFM2ThreadPool     *pool = ctx->pool;
    uint32_t r_start = 0, r_end = 0;

    while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
        for (uint32_t r = r_start; r < r_end; ++r) {
            lfm2_matmul_batch_dispatch_row(
                ctx->w_data, ctx->w_type,
                ctx->x_f32_batch, ctx->x_q8_batch,
                ctx->B, ctx->n_cols, ctx->n_rows, r, ctx->out_batch);
        }
    }
}

void lfm2_matmul_batch_avx2_mt(LFM2ThreadPool *pool,
                               float * restrict out_batch,
                               const GGUFTensor * restrict W,
                               const float * restrict x_batch,
                               uint32_t B,
                               block_q8_K_t * restrict xq_scratch) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    const uint32_t n_rows = (uint32_t)W->ne[1];

    if (W->type == GGML_TYPE_Q4_K || W->type == GGML_TYPE_Q6_K ||
        W->type == GGML_TYPE_Q8_0 || W->type == GGML_TYPE_Q4_0 ||
        W->type == GGML_TYPE_Q5_K || W->type == GGML_TYPE_IQ4_XS) {
        lfm2_quantize_batch_q8_K_avx2(x_batch, xq_scratch, B, n_cols);
    }

    if (!pool || pool->n_threads <= 1) {
        for (uint32_t r = 0; r < n_rows; ++r) {
            lfm2_matmul_batch_dispatch_row(
                W->data, W->type, x_batch, xq_scratch, B, n_cols, n_rows, r, out_batch);
        }
        return;
    }

    BatchMatMulTaskCtx ctx = {
        .pool        = pool,
        .out_batch   = out_batch,
        .w_data      = W->data,
        .w_type      = W->type,
        .B           = B,
        .n_cols      = n_cols,
        .n_rows      = n_rows,
        .x_f32_batch = x_batch,
        .x_q8_batch  = xq_scratch
    };

    lfm2_parallel_for(pool, n_rows, 16, lfm2_matmul_batch_worker_task, &ctx);
}

/* ============================================================================
 * 5. Batch-Capable LFM2 Runtime Arena (Supports B = 1 .. LFM2_MAX_BATCH)
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool pool;

    /* Batch-sized activation scratch buffers (LFM2_MAX_BATCH * dim) */
    float *x;           /* [B, n_embd] Residual stream                       */
    float *xb;          /* [B, n_embd] RMSNorm / operator output             */
    float *bcx;         /* [B, 3 * n_embd] ShortConv in_proj output          */
    float *conv_out;    /* [B, n_embd] ShortConv gated depthwise output      */
    float *q;           /* [B, n_embd] Query batch                           */
    float *k;           /* [B, n_embd] Key batch                             */
    float *v;           /* [B, n_embd] Value batch                           */
    float *att;         /* [n_head_q, max_seq_len] Attention score scratch   */
    float *hb;          /* [B, n_ff] SwiGLU gate & output                    */
    float *hb2;         /* [B, n_ff] SwiGLU up projection                    */
    block_q8_K_t *xq;   /* [B, max_cols / 256] Quantized activation buffer   */

    /* Single-token output logits (only computed for the final token!) */
    float *logits;      /* [n_vocab]                                         */

    /* Fast trigonometric tables for NeoX RoPE */
    float *rope_cos;    /* [max_seq_len * (head_dim / 2)]                    */
    float *rope_sin;    /* [max_seq_len * (head_dim / 2)]                    */

    /* Pre-unpacked 3-tap ShortConv filter weights */
    float *conv_w[LFM2_MAX_LAYERS][3];

    /* Persistent Hybrid State across Tokens */
    uint32_t  max_seq_len;
    float    *conv_state[LFM2_MAX_LAYERS]; /* [hist][n_embd] for ShortConv    */
    float    *k_cache[LFM2_MAX_LAYERS];    /* [max_seq_len][kv_dim] for GQA   */
    float    *v_cache[LFM2_MAX_LAYERS];    /* [max_seq_len][kv_dim] for GQA   */
    uint16_t *k_cache_f16[LFM2_MAX_LAYERS];/* [max_seq_len][kv_dim] FP16      */
    uint16_t *v_cache_f16[LFM2_MAX_LAYERS];/* [max_seq_len][kv_dim] FP16      */
} LFM2BatchState;

bool lfm2_batch_state_init(LFM2BatchState *s, const LFM2Model *m,
                           uint32_t max_seq_len, uint32_t n_threads) {
    memset(s, 0, sizeof(*s));
    lfm2_threadpool_init(&s->pool, n_threads);

    const LFM2Config *c = &m->cfg;
    s->max_seq_len = lfm_resolve_ctx_len(max_seq_len, c->n_ctx);

    const size_t B        = LFM2_MAX_BATCH;
    const size_t D        = c->n_embd;
    const size_t head_dim = c->n_head_q > 0 ? (D / c->n_head_q) : 64;
    const size_t half_dim = head_dim / 2;
    const size_t hist_len = (c->shortconv_l_cache > 1) ? (c->shortconv_l_cache - 1) : 2;

    uint32_t max_ff = c->n_ff;
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].ffn_gate) {
            uint32_t l_ff = (uint32_t)m->layers[l].ffn_gate->ne[1];
            if (l_ff > max_ff) max_ff = l_ff;
        }
    }
    if (max_ff == 0) max_ff = D * 4;

    size_t max_cols = (3 * D > max_ff) ? (3 * D) : max_ff;
    if (c->n_vocab > max_cols) max_cols = c->n_vocab;

    const size_t D_pad        = ((D + 255) / 256) * 256;
    const size_t max_ff_pad   = ((max_ff + 255) / 256) * 256;
    const size_t max_cols_pad = ((max_cols + 255) / 256) * 256;
    const size_t n_vocab_pad  = ((c->n_vocab + 255) / 256) * 256;

    s->x        = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->xb       = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->bcx      = (float *)lfm2_aligned_alloc(64, B * 3 * D_pad * sizeof(float));
    s->conv_out = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->q        = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->k        = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->v        = (float *)lfm2_aligned_alloc(64, B * D_pad * sizeof(float));
    s->att      = (float *)lfm2_aligned_alloc(64, (size_t)c->n_head_q * s->max_seq_len * sizeof(float));
    s->hb       = (float *)lfm2_aligned_alloc(64, B * max_ff_pad * sizeof(float));
    s->hb2      = (float *)lfm2_aligned_alloc(64, B * max_ff_pad * sizeof(float));
    s->xq       = (block_q8_K_t *)lfm2_aligned_alloc(64, B * (max_cols_pad / 256) * sizeof(block_q8_K_t));
    s->logits   = (float *)lfm2_aligned_alloc(64, n_vocab_pad * sizeof(float));

    s->rope_cos = (float *)lfm2_aligned_alloc(64, (size_t)s->max_seq_len * half_dim * sizeof(float));
    s->rope_sin = (float *)lfm2_aligned_alloc(64, (size_t)s->max_seq_len * half_dim * sizeof(float));

    if (!s->x || !s->xb || !s->bcx || !s->conv_out || !s->q ||
        !s->k || !s->v  || !s->att || !s->hb || !s->hb2 || !s->xq || !s->logits ||
        !s->rope_cos || !s->rope_sin) {
        return false;
    }

    /* Precompute RoPE cos/sin tables */
    const float freq_base = (c->rope_freq_base > 0.0f) ? c->rope_freq_base : 10000.0f;
    for (uint32_t i = 0; i < half_dim; ++i) {
        float freq = powf(freq_base, -((float)(2 * i) / (float)head_dim));
        for (uint32_t p = 0; p < s->max_seq_len; ++p) {
            float angle = (float)p * freq;
            s->rope_cos[(size_t)p * half_dim + i] = cosf(angle);
            s->rope_sin[(size_t)p * half_dim + i] = sinf(angle);
        }
    }

    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            s->conv_state[l] = (float *)calloc(hist_len * D, sizeof(float));
            if (!s->conv_state[l]) return false;

            const GGUFTensor *conv_t = m->layers[l].shortconv.conv;
            if (conv_t && conv_t->data) {
                const uint32_t l_cache = c->shortconv_l_cache ? c->shortconv_l_cache : 3;
                const bool stride_is_lcache = (conv_t->ne[0] == l_cache);
                for (int k = 0; k < 3; ++k) {
                    s->conv_w[l][k] = (float *)calloc(D, sizeof(float));
                    if (!s->conv_w[l][k]) return false;
                    for (uint32_t d = 0; d < D; ++d) {
                        float val = 0.0f;
                        if (conv_t->type == GGML_TYPE_F32) {
                            const float *w_base = (const float *)conv_t->data;
                            val = stride_is_lcache ? w_base[d * l_cache + k] : w_base[k * D + d];
                        } else if (conv_t->type == GGML_TYPE_F16) {
                            const uint16_t *w_base = (const uint16_t *)conv_t->data;
                            val = stride_is_lcache ? lfm2_fp16_to_fp32(w_base[d * l_cache + k]) : lfm2_fp16_to_fp32(w_base[k * D + d]);
                        }
                        s->conv_w[l][k][d] = val;
                    }
                }
            }
        } else {
            const size_t kv_dim = (size_t)c->n_head_kv[l] * head_dim;
            s->k_cache[l]     = (float *)calloc(s->max_seq_len * kv_dim, sizeof(float));
            s->v_cache[l]     = (float *)calloc(s->max_seq_len * kv_dim, sizeof(float));
            s->k_cache_f16[l] = (uint16_t *)calloc(s->max_seq_len * kv_dim, sizeof(uint16_t));
            s->v_cache_f16[l] = (uint16_t *)calloc(s->max_seq_len * kv_dim, sizeof(uint16_t));
#if defined(MADV_HUGEPAGE)
            madvise(s->k_cache_f16[l], s->max_seq_len * kv_dim * sizeof(uint16_t), MADV_HUGEPAGE);
            madvise(s->v_cache_f16[l], s->max_seq_len * kv_dim * sizeof(uint16_t), MADV_HUGEPAGE);
#endif
            if (!s->k_cache[l] || !s->v_cache[l] || !s->k_cache_f16[l] || !s->v_cache_f16[l]) return false;
        }
    }
    return true;
}

void lfm2_batch_state_free(LFM2BatchState *s) {
    lfm2_threadpool_free(&s->pool);
    free(s->x); free(s->xb); free(s->bcx); free(s->conv_out);
    free(s->q); free(s->k);  free(s->v);   free(s->att);
    free(s->hb); free(s->hb2); free(s->xq); free(s->logits);
    free(s->rope_cos); free(s->rope_sin);
    for (int l = 0; l < LFM2_MAX_LAYERS; ++l) {
        free(s->conv_state[l]);
        for (int k = 0; k < 3; ++k) free(s->conv_w[l][k]);
        free(s->k_cache[l]);
        free(s->v_cache[l]);
        free(s->k_cache_f16[l]);
        free(s->v_cache_f16[l]);
    }
    memset(s, 0, sizeof(*s));
}

/* ============================================================================
 * 6. Batched Hybrid Layer Operators (ShortConv, GQA, SwiGLU)
 * ============================================================================ */

static inline void lfm2_vec_add_f32(float * restrict dst, const float * restrict src, size_t n) {
    size_t i = 0;
#if LFM2_HAS_AVX2
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
#elif defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

static inline void lfm2_rmsnorm_batch(float *out_batch, const float *x_batch,
                                      const GGUFTensor *w_tensor, uint32_t B,
                                      uint32_t dim, float eps) {
    for (uint32_t t = 0; t < B; ++t) {
        lfm2_rmsnorm(out_batch + (size_t)t * dim,
                     x_batch + (size_t)t * dim,
                     w_tensor, dim, eps);
    }
}

static inline void lfm2_qk_norm_and_rope_table(const float *rope_cos, const float *rope_sin,
                                               float *vec, uint32_t n_heads, uint32_t head_dim,
                                               const GGUFTensor *norm_weight, float eps,
                                               uint32_t pos) {
    const uint32_t half_dim = head_dim / 2;
    const float *cos_table = rope_cos ? (rope_cos + (size_t)pos * half_dim) : NULL;
    const float *sin_table = rope_sin ? (rope_sin + (size_t)pos * half_dim) : NULL;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float *head = vec + (size_t)h * head_dim;

        /* 1. Per-Head RMSNorm */
        if (norm_weight != NULL) {
            lfm2_rmsnorm(head, head, norm_weight, head_dim, eps);
        }

        /* 2. NeoX RoPE */
        if (cos_table && sin_table) {
            uint32_t i = 0;
#if LFM2_HAS_AVX2
            for (; i + 8 <= half_dim; i += 8) {
                __m256 vx0  = _mm256_loadu_ps(head + i);
                __m256 vx1  = _mm256_loadu_ps(head + half_dim + i);
                __m256 vcos = _mm256_loadu_ps(cos_table + i);
                __m256 vsin = _mm256_loadu_ps(sin_table + i);

                __m256 vout0 = _mm256_fmsub_ps(vx0, vcos, _mm256_mul_ps(vx1, vsin));
                __m256 vout1 = _mm256_fmadd_ps(vx1, vcos, _mm256_mul_ps(vx0, vsin));

                _mm256_storeu_ps(head + i, vout0);
                _mm256_storeu_ps(head + half_dim + i, vout1);
            }
#elif defined(__aarch64__)
            for (; i + 4 <= half_dim; i += 4) {
                float32x4_t vx0  = vld1q_f32(head + i);
                float32x4_t vx1  = vld1q_f32(head + half_dim + i);
                float32x4_t vcos = vld1q_f32(cos_table + i);
                float32x4_t vsin = vld1q_f32(sin_table + i);

                float32x4_t vout0 = vfmsq_f32(vmulq_f32(vx0, vcos), vx1, vsin);
                float32x4_t vout1 = vfmaq_f32(vmulq_f32(vx1, vcos), vx0, vsin);

                vst1q_f32(head + i, vout0);
                vst1q_f32(head + half_dim + i, vout1);
            }
#endif
            for (; i < half_dim; ++i) {
                const float cos_a = cos_table[i];
                const float sin_a = sin_table[i];
                const float x0 = head[i];
                const float x1 = head[i + half_dim];
                head[i]            = x0 * cos_a - x1 * sin_a;
                head[i + half_dim] = x1 * cos_a + x0 * sin_a;
            }
        } else {
            for (uint32_t i = 0; i < half_dim; ++i) {
                const float freq  = powf(10000.0f, -((float)(2 * i) / (float)head_dim));
                const float angle = (float)pos * freq;
                const float cos_a = cosf(angle);
                const float sin_a = sinf(angle);
                const float x0 = head[i];
                const float x1 = head[i + half_dim];
                head[i]            = x0 * cos_a - x1 * sin_a;
                head[i + half_dim] = x1 * cos_a + x0 * sin_a;
            }
        }
    }
}

/* Batched ShortConv Block:
 * - in_proj & out_proj execute as B-token batched GEMMs (8x less RAM traffic)
 * - Depthwise 3-tap conv steps sequentially t = 0..B-1 in L1 cache for exact causality */
static void lfm2_forward_shortconv_batch(LFM2BatchState *s, const LFM2Model *m,
                                         const LFM2Layer *layer, uint32_t l_idx, uint32_t B) {
    const uint32_t D       = m->cfg.n_embd;
    const uint32_t l_cache = m->cfg.shortconv_l_cache ? m->cfg.shortconv_l_cache : 3;
    const uint32_t hist    = (l_cache > 1) ? (l_cache - 1) : 2;

    /* 1. Batched Input Projection: xb [B, D] -> bcx [B, 3*D] */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->bcx, layer->shortconv.in_proj, s->xb, B, s->xq);

    /* 2. Causal Depthwise Conv1D + Gates across t = 0 .. B - 1 */
    float *state = s->conv_state[l_idx]; /* [hist][D] */
    const float *w0 = s->conv_w[l_idx][0];
    const float *w1 = s->conv_w[l_idx][1];
    const float *w2 = s->conv_w[l_idx][2];

    for (uint32_t t = 0; t < B; ++t) {
        float *bcx_t    = s->bcx + (size_t)t * 3 * D;
        float *B_gate   = bcx_t;
        float *C_gate   = bcx_t + D;
        float *x_tilde  = bcx_t + 2 * D;
        float *out_t    = s->conv_out + (size_t)t * D;

        if (w0 && w1 && w2 && hist == 2) {
            const float *s0 = state;
            const float *s1 = state + D;
            uint32_t d = 0;
#if LFM2_HAS_AVX2
            for (; d + 8 <= D; d += 8) {
                __m256 vb = _mm256_loadu_ps(B_gate + d);
                __m256 vx = _mm256_loadu_ps(x_tilde + d);
                __m256 vu = _mm256_mul_ps(vb, vx);
                _mm256_storeu_ps(B_gate + d, vu);

                __m256 vw0 = _mm256_loadu_ps(w0 + d);
                __m256 vs0 = _mm256_loadu_ps(s0 + d);
                __m256 vw1 = _mm256_loadu_ps(w1 + d);
                __m256 vs1 = _mm256_loadu_ps(s1 + d);
                __m256 vw2 = _mm256_loadu_ps(w2 + d);
                __m256 vc  = _mm256_loadu_ps(C_gate + d);

                __m256 vacc = _mm256_fmadd_ps(vw0, vs0, _mm256_mul_ps(vw1, vs1));
                vacc = _mm256_fmadd_ps(vw2, vu, vacc);
                __m256 vout = _mm256_mul_ps(vc, vacc);
                _mm256_storeu_ps(out_t + d, vout);
            }
#elif defined(__aarch64__)
            for (; d + 4 <= D; d += 4) {
                float32x4_t vb = vld1q_f32(B_gate + d);
                float32x4_t vx = vld1q_f32(x_tilde + d);
                float32x4_t vu = vmulq_f32(vb, vx);
                vst1q_f32(B_gate + d, vu);

                float32x4_t vw0 = vld1q_f32(w0 + d);
                float32x4_t vs0 = vld1q_f32(s0 + d);
                float32x4_t vw1 = vld1q_f32(w1 + d);
                float32x4_t vs1 = vld1q_f32(s1 + d);
                float32x4_t vw2 = vld1q_f32(w2 + d);
                float32x4_t vc  = vld1q_f32(C_gate + d);

                float32x4_t vacc = vfmaq_f32(vmulq_f32(vw1, vs1), vw0, vs0);
                vacc = vfmaq_f32(vacc, vw2, vu);
                vst1q_f32(out_t + d, vmulq_f32(vc, vacc));
            }
#endif
            for (; d < D; ++d) {
                float u = B_gate[d] * x_tilde[d];
                B_gate[d] = u;
                float acc = w0[d] * s0[d] + w1[d] * s1[d] + w2[d] * u;
                out_t[d] = C_gate[d] * acc;
            }
        } else {
            const GGUFTensor *conv_t = layer->shortconv.conv;
            const bool stride_is_lcache = (conv_t->ne[0] == l_cache);
            for (uint32_t d = 0; d < D; ++d) {
                float u_t = B_gate[d] * x_tilde[d];
                B_gate[d] = u_t;

                float acc = 0.0f;
                if (conv_t->type == GGML_TYPE_F32) {
                    const float *w_base = (const float *)conv_t->data;
                    for (uint32_t k = 0; k < hist; ++k) {
                        float w_k = stride_is_lcache ? w_base[d * l_cache + k] : w_base[k * D + d];
                        acc += w_k * state[k * D + d];
                    }
                    float w_last = stride_is_lcache ? w_base[d * l_cache + hist] : w_base[hist * D + d];
                    acc += w_last * u_t;
                } else {
                    const uint16_t *w_base = (const uint16_t *)conv_t->data;
                    for (uint32_t k = 0; k < hist; ++k) {
                        float w_k = stride_is_lcache ? lfm2_fp16_to_fp32(w_base[d * l_cache + k]) : lfm2_fp16_to_fp32(w_base[k * D + d]);
                        acc += w_k * state[k * D + d];
                    }
                    float w_last = stride_is_lcache ? lfm2_fp16_to_fp32(w_base[d * l_cache + hist]) : lfm2_fp16_to_fp32(w_base[hist * D + d]);
                    acc += w_last * u_t;
                }
                out_t[d] = C_gate[d] * acc;
            }
        }

        /* Advance rolling 2-token state buffer for token t + 1 */
        if (hist > 1) {
            memmove(state, state + D, (size_t)(hist - 1) * D * sizeof(float));
        }
        memcpy(state + (size_t)(hist - 1) * D, B_gate, D * sizeof(float));
    }

    /* 3. Batched Output Projection: conv_out [B, D] -> xb [B, D] + Residual Add */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->xb, layer->shortconv.out_proj, s->conv_out, B, s->xq);
    lfm2_vec_add_f32(s->x, s->xb, (size_t)B * D);
}

/* Batched GQA Block:
 * - Q, K, V and attn_output run as B-token batched GEMMs
 * - Causal attention runs per token at position `start_pos + t` */
static void lfm2_forward_gqa_batch(LFM2BatchState *s, const LFM2Model *m,
                                   const LFM2Layer *layer, uint32_t l_idx,
                                   uint32_t start_pos, uint32_t B) {
    const uint32_t D         = m->cfg.n_embd;
    const uint32_t n_head_q  = m->cfg.n_head_q;
    const uint32_t n_head_kv = m->cfg.n_head_kv[l_idx];
    const uint32_t head_dim  = n_head_q > 0 ? (D / n_head_q) : 64;
    const uint32_t kv_dim    = n_head_kv * head_dim;
    const uint32_t kv_mul    = (n_head_kv > 0) ? (n_head_q / n_head_kv) : 1;
    const float    scale     = 1.0f / sqrtf((float)head_dim);

    /* 1. Batched Q, K, V Projections across all B tokens */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->q, layer->attn.wq, s->xb, B, s->xq);
    lfm2_matmul_batch_avx2_mt(&s->pool, s->k, layer->attn.wk, s->xb, B, s->xq);
    lfm2_matmul_batch_avx2_mt(&s->pool, s->v, layer->attn.wv, s->xb, B, s->xq);

    /* 2. Apply Per-Head QK-Norm + RoPE, store K/V into cache, and compute causal GQA */
    for (uint32_t t = 0; t < B; ++t) {
        const uint32_t pos = start_pos + t;
        float *q_t  = s->q  + (size_t)t * D;
        float *k_t  = s->k  + (size_t)t * kv_dim;
        float *v_t  = s->v  + (size_t)t * kv_dim;
        float *xb_t = s->xb + (size_t)t * D;

        lfm2_qk_norm_and_rope_table(s->rope_cos, s->rope_sin,
                                    q_t, n_head_q,  head_dim, layer->attn.q_norm,
                                    m->cfg.rms_norm_eps, pos);
        lfm2_qk_norm_and_rope_table(s->rope_cos, s->rope_sin,
                                    k_t, n_head_kv, head_dim, layer->attn.k_norm,
                                    m->cfg.rms_norm_eps, pos);

        if (s->k_cache[l_idx] && s->v_cache[l_idx]) {
            memcpy(s->k_cache[l_idx] + (size_t)pos * kv_dim, k_t, kv_dim * sizeof(float));
            memcpy(s->v_cache[l_idx] + (size_t)pos * kv_dim, v_t, kv_dim * sizeof(float));
        }
        if (s->k_cache_f16[l_idx] && s->v_cache_f16[l_idx]) {
            uint16_t *k_step = s->k_cache_f16[l_idx] + (size_t)pos * kv_dim;
            uint16_t *v_step = s->v_cache_f16[l_idx] + (size_t)pos * kv_dim;
            uint32_t i = 0;
#if defined(__F16C__)
            for (; i + 8 <= kv_dim; i += 8) {
                __m256 vk = _mm256_loadu_ps(k_t + i);
                __m256 vv = _mm256_loadu_ps(v_t + i);
                _mm_storeu_si128((__m128i *)(k_step + i), _mm256_cvtps_ph(vk, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm_storeu_si128((__m128i *)(v_step + i), _mm256_cvtps_ph(vv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            }
#elif defined(__aarch64__)
            for (; i + 8 <= kv_dim; i += 8) {
                float16x4_t hk0 = vcvt_f16_f32(vld1q_f32(k_t + i + 0));
                float16x4_t hk1 = vcvt_f16_f32(vld1q_f32(k_t + i + 4));
                float16x4_t hv0 = vcvt_f16_f32(vld1q_f32(v_t + i + 0));
                float16x4_t hv1 = vcvt_f16_f32(vld1q_f32(v_t + i + 4));
                vst1q_u16(k_step + i, vreinterpretq_u16_f16(vcombine_f16(hk0, hk1)));
                vst1q_u16(v_step + i, vreinterpretq_u16_f16(vcombine_f16(hv0, hv1)));
            }
#endif
            for (; i < kv_dim; ++i) {
                k_step[i] = lfm2_fp32_to_fp16(k_t[i]);
                v_step[i] = lfm2_fp32_to_fp16(v_t[i]);
            }
        }

        GQAAttnTaskCtx gqa_ctx = {
            .pool        = &s->pool,
            .q           = q_t,
            .k_cache     = s->k_cache[l_idx],
            .v_cache     = s->v_cache[l_idx],
            .k_cache_f16 = s->k_cache_f16[l_idx],
            .v_cache_f16 = s->v_cache_f16[l_idx],
            .att         = s->att,
            .xb          = xb_t,
            .n_head_q    = n_head_q,
            .kv_mul      = kv_mul,
            .head_dim    = head_dim,
            .kv_dim      = kv_dim,
            .max_seq_len = s->max_seq_len,
            .pos         = pos,
            .scale       = scale
        };
        lfm2_parallel_for(&s->pool, n_head_q, 1, lfm2_gqa_heads_worker_task, &gqa_ctx);
    }

    /* 3. Batched Output Projection: xb [B, D] -> conv_out [B, D] + Residual Add */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->conv_out, layer->attn.wo, s->xb, B, s->xq);
    lfm2_vec_add_f32(s->x, s->conv_out, (size_t)B * D);
}

/* Batched SwiGLU FFN Block */
static void lfm2_forward_ffn_batch(LFM2BatchState *s, const LFM2Model *m,
                                   const LFM2Layer *layer, uint32_t B) {
    const uint32_t D    = m->cfg.n_embd;
    const uint32_t n_ff = (uint32_t)layer->ffn_gate->ne[1];

    lfm2_rmsnorm_batch(s->xb, s->x, layer->ffn_norm, B, D, m->cfg.rms_norm_eps);

    /* Batched Gate & Up Projections: xb [B, D] -> hb [B, n_ff], hb2 [B, n_ff] */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->hb,  layer->ffn_gate, s->xb, B, s->xq);
    lfm2_matmul_batch_avx2_mt(&s->pool, s->hb2, layer->ffn_up,   s->xb, B, s->xq);

    const size_t total_ff = (size_t)B * n_ff;
    for (size_t i = 0; i < total_ff; ++i) {
        const float g = s->hb[i];
        s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
    }

    /* Batched Down Projection: hb [B, n_ff] -> xb [B, D] + Residual Add */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->xb, layer->ffn_down, s->hb, B, s->xq);
    lfm2_vec_add_f32(s->x, s->xb, (size_t)B * D);
}

/* ============================================================================
 * 7. Top-Level Batched Forward Pass & Prompt Prefill Engine
 * ============================================================================
 * Processes `B` (1..LFM2_MAX_BATCH) tokens starting at `start_pos`.
 * If `compute_last_logits == false`, skips `output_norm` and `output.weight`
 * completely. If `true`, projects ONLY the final token `B - 1` into `s->logits`!
 * ============================================================================ */
float *lfm2_forward_batch(const LFM2Model *m,
                          LFM2BatchState *s,
                          const uint32_t *tokens,
                          uint32_t B,
                          uint32_t start_pos,
                          bool compute_last_logits) {
    if (B == 0 || B > LFM2_MAX_BATCH || start_pos + B > s->max_seq_len) return NULL;
    const uint32_t D = m->cfg.n_embd;

    /* 1. Dequantize B token embeddings into s->x[B, D] */
    for (uint32_t t = 0; t < B; ++t) {
        lfm2_dequantize_row(m->token_embd, tokens[t], s->x + (size_t)t * D);
    }

    /* 2. Execute All Hybrid Layers Across the B-Token Batch */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];
        lfm2_rmsnorm_batch(s->xb, s->x, layer->attn_norm, B, D, m->cfg.rms_norm_eps);

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv_batch(s, m, layer, l, B);
        } else {
            lfm2_forward_gqa_batch(s, m, layer, l, start_pos, B);
        }

        lfm2_forward_ffn_batch(s, m, layer, B);
    }

    /* 3. Skip output_norm & output.weight unless this is the final token of prefill */
    if (!compute_last_logits) {
        return NULL;
    }

    /* Project ONLY the last token in the batch (index B - 1) through the LM head */
    float *last_x = s->x + (size_t)(B - 1) * D;
    lfm2_rmsnorm(last_x, last_x, m->output_norm, D, m->cfg.rms_norm_eps);
    if (m->output->type == GGML_TYPE_Q4_K || m->output->type == GGML_TYPE_Q6_K ||
        m->output->type == GGML_TYPE_Q8_0 || m->output->type == GGML_TYPE_Q4_0 ||
        m->output->type == GGML_TYPE_Q5_K || m->output->type == GGML_TYPE_IQ4_XS ||
        m->output->type == GGML_TYPE_F32) {
        lfm2_matmul_batch_avx2_mt(&s->pool, s->logits, m->output, last_x, 1, (block_q8_K_t *)s->xq);
    } else {
        lfm2_matmul_vec(s->logits, m->output, last_x, (block_q8_K_t *)s->xq, &s->pool);
    }
    return s->logits;
}

/* High-level prompt prefill driver: chunks arbitrary prompt lengths into
 * micro-batches of 8 tokens and computes logits ONLY on the final token. */
float *lfm2_prefill_prompt(const LFM2Model *m,
                           LFM2BatchState *s,
                           const uint32_t *prompt_tokens,
                           uint32_t n_prompt,
                           uint32_t *inout_pos) {
    float *logits = NULL;
    uint32_t i = 0;

    while (i < n_prompt) {
        uint32_t chunk = n_prompt - i;
        if (chunk > LFM2_MAX_BATCH) chunk = LFM2_MAX_BATCH;

        /* Only compute output.weight logits on the very last chunk! */
        const bool is_last_chunk = (i + chunk == n_prompt);

        logits = lfm2_forward_batch(m, s, prompt_tokens + i, chunk,
                                    *inout_pos, is_last_chunk);
        *inout_pos += chunk;
        i += chunk;
    }
    return logits;
}

#ifndef LFM2_PREFILL_BATCH_NO_MAIN
/* ============================================================================
 * 8. Standalone Verification & Prefill Speedup Benchmark
 * ============================================================================ */
int main(int argc, char **argv) {
    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 6144;
    const uint32_t B      = 8;
    const uint32_t nb     = n_cols / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0);

    float        *x_batch   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_cols * sizeof(float));
    float        *out_seq   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_batch = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    block_q8_K_t *xq_batch  = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4      = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)n_rows * nb * sizeof(block_q4_K_t));

    for (uint32_t i = 0; i < B * n_cols; ++i) x_batch[i] = sinf((float)i * 0.011f);
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3000;
        w_q4[i].dmin = 0x2800;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i + j * 5) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 19 + j * 23) & 0xFF);
    }

    GGUFTensor fake_W = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_q4 };

    /* 1. Verify Exact Bit-for-Bit Parity between 8 Sequential GEMVs and 1 Batched GEMM */
    for (uint32_t t = 0; t < B; ++t) {
        lfm2_matmul_batch_avx2_mt(&pool, out_seq + (size_t)t * n_rows, &fake_W,
                                  x_batch + (size_t)t * n_cols, 1, xq_batch);
    }
    lfm2_matmul_batch_avx2_mt(&pool, out_batch, &fake_W, x_batch, B, xq_batch);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < B * n_rows; ++i) {
        float diff = fabsf(out_seq[i] - out_batch[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("=== LFM2 Batched Prefill GEMM (B=%u) vs Sequential GEMV Benchmark ===\n", B);
    printf("Physical Cores Active : %u\n", pool.n_threads);
    printf("Max Numerical Diff    : %.8f (Bit-Exact AVX2 Accumulator Parity)\n\n", max_diff);

    /* 2. Benchmark 8 Sequential M=1 GEMVs vs 1 Batched M=8 GEMM */
    const int iters = 50;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        for (uint32_t t = 0; t < B; ++t) {
            lfm2_matmul_batch_avx2_mt(&pool, out_seq + (size_t)t * n_rows, &fake_W,
                                      x_batch + (size_t)t * n_cols, 1, xq_batch);
        }
    }
    double seq_ms = (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_matmul_batch_avx2_mt(&pool, out_batch, &fake_W, x_batch, B, xq_batch);
    }
    double batch_ms = (now_ms() - t0) / iters;

    printf("8 Tokens via Sequential GEMV (M=1 x 8) : %.3f ms (%.2f ms/tok)\n", seq_ms, seq_ms / B);
    printf("8 Tokens via Batched GEMM    (M=8 x 1) : %.3f ms (%.2f ms/tok)\n", batch_ms, batch_ms / B);
    printf("MatMul Prefill Speedup                 : %.2fx faster\n", seq_ms / batch_ms);

    free(x_batch); free(out_seq); free(out_batch); free(xq_batch); free(w_q4);
    lfm2_threadpool_free(&pool);
    return EXIT_SUCCESS;
}
#endif

#endif /* LFM2_PREFILL_BATCH_INCLUDED */
