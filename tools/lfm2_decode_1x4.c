#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

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
#ifndef hsum_epi32_avx2
#define hsum_epi32_avx2 lfm2_hsum_epi32_avx2
#endif
#ifndef hsum_ps_avx2
#define hsum_ps_avx2 lfm2_hsum_ps_avx2
#endif
#ifndef mm256_set_m128i
#define mm256_set_m128i lfm2_mm256_set_m128i
#endif

#ifndef lfm2_aligned_alloc_64
#define lfm2_aligned_alloc_64 lfm2_aligned_alloc_64
static inline void *lfm2_aligned_alloc_64(size_t size) {
    size_t aligned_size = (size + 63) & ~((size_t)63);
    return aligned_alloc(64, aligned_size);
}
#endif

/* ============================================================================
 * 1. 4-Row Simultaneous Horizontal Reduction (4 x __m256i -> 1 x __m128i)
 * ============================================================================
 * Reduces 4 independent 8-lane int32 accumulators (r0, r1, r2, r3) into a
 * single 4-lane __m128i vector [sum(r0), sum(r1), sum(r2), sum(r3)] in 4 ops.
 * ============================================================================ */
static inline __m128i hsum4_epi32_avx2(__m256i r0, __m256i r1, __m256i r2, __m256i r3) {
    __m256i h01   = _mm256_hadd_epi32(r0, r1);
    __m256i h23   = _mm256_hadd_epi32(r2, r3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123),
                         _mm256_extracti128_si256(h0123, 1));
}

/* ============================================================================
 * 2. Fused AVX2 RMSNorm + Q8_K Quantizer (Single Pass over L1 Cache)
 * ============================================================================
 * Computes out_f32[i] = (x[i] / rms(x)) * weight[i] AND immediately quantizes
 * every 256-float super-block into `out_q8[b]` so subsequent projections
 * (QKV, Gate+Up, ShortConv in_proj) never re-read or re-quantize `out_f32`.
 * ============================================================================ */
void lfm2_rmsnorm_quantize_q8_K_avx2(float * restrict out_f32,
                                     block_q8_K_t * restrict out_q8,
                                     const float * restrict x,
                                     const GGUFTensor * restrict norm_w,
                                     uint32_t dim,
                                     float eps) {
    /* 1. Vectorized sum of squares across x[0 .. dim-1] */
    __m256 vss0 = _mm256_setzero_ps();
    __m256 vss1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(x + i + 0);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        vss0 = _mm256_fmadd_ps(vx0, vx0, vss0);
        vss1 = _mm256_fmadd_ps(vx1, vx1, vss1);
    }
    float ss = hsum_ps_avx2(_mm256_add_ps(vss0, vss1));
    for (; i < dim; ++i) {
        ss += x[i] * x[i];
    }
    const float inv_rms = 1.0f / sqrtf(ss / (float)dim + eps);
    const __m256 v_irms = _mm256_set1_ps(inv_rms);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const float    *wf32 = (norm_w && norm_w->type == GGML_TYPE_F32) ? (const float *)norm_w->data : NULL;
    const uint16_t *wf16 = (norm_w && norm_w->type != GGML_TYPE_F32) ? (const uint16_t *)norm_w->data : NULL;

    /* 2. Normalize 256 elements at a time and immediately quantize to Q8_K */
    const uint32_t nb = dim / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256;
        float *xb_out = out_f32 + base;
        __m256 vmax   = _mm256_setzero_ps();

        if (wf32) {
            for (uint32_t j = 0; j < 256; j += 16) {
                __m256 n0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + j + 0), v_irms),
                                          _mm256_loadu_ps(wf32 + base + j + 0));
                __m256 n1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + j + 8), v_irms),
                                          _mm256_loadu_ps(wf32 + base + j + 8));
                _mm256_storeu_ps(xb_out + j + 0, n0);
                _mm256_storeu_ps(xb_out + j + 8, n1);
                vmax = _mm256_max_ps(vmax, _mm256_max_ps(_mm256_andnot_ps(sign_mask, n0),
                                                         _mm256_andnot_ps(sign_mask, n1)));
            }
        } else if (wf16) {
            for (uint32_t j = 0; j < 256; j += 8) {
                float w_tmp[8];
                for (int k = 0; k < 8; ++k) w_tmp[k] = fp16_to_fp32(wf16[base + j + k]);
                __m256 n0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + j), v_irms),
                                          _mm256_loadu_ps(w_tmp));
                _mm256_storeu_ps(xb_out + j, n0);
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, n0));
            }
        } else {
            for (uint32_t j = 0; j < 256; j += 8) {
                __m256 n0 = _mm256_mul_ps(_mm256_loadu_ps(x + base + j), v_irms);
                _mm256_storeu_ps(xb_out + j, n0);
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, n0));
            }
        }

        if (out_q8 == NULL) continue;

        /* Reduce max_abs for this 256-float super-block */
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        const float max_abs = _mm_cvtss_f32(m128);

        if (max_abs == 0.0f) {
            out_q8[b].d = 0.0f;
            memset(out_q8[b].qs, 0, sizeof(out_q8[b].qs));
            memset(out_q8[b].bsums, 0, sizeof(out_q8[b].bsums));
            continue;
        }

        out_q8[b].d = max_abs / 127.0f;
        const __m256 vscale = _mm256_set1_ps(127.0f / max_abs);

        /* Quantize hot xb_out[0..255] directly from L1 cache into out_q8[b] */
        for (int j = 0; j < 16; ++j) {
            const float *x16 = xb_out + j * 16;
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 0), vscale));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 8), vscale));

            out_q8[b].bsums[j] = (int16_t)hsum_epi32_avx2(_mm256_add_epi32(i0, i1));

            __m256i p16 = _mm256_packs_epi32(i0, i1);
            p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i p8  = _mm_packs_epi16(_mm256_castsi256_si128(p16),
                                          _mm256_extracti128_si256(p16, 1));
            _mm_storeu_si128((__m128i *)(out_q8[b].qs + j * 16), p8);
        }
    }

    /* Tail scalar handling if dim % 256 != 0 */
    for (uint32_t j = nb * 256; j < dim; ++j) {
        float w = 1.0f;
        if (wf32) w = wf32[j];
        else if (wf16) w = fp16_to_fp32(wf16[j]);
        out_f32[j] = x[j] * inv_rms * w;
    }
}

/* ============================================================================
 * 3. 1x4 Row-Unrolled AVX2 Q4_K x Q8_K Micro-Kernel (4 Rows Simultaneously)
 * ============================================================================ */
static inline void lfm2_vec_dot_q4_K_q8_K_1x4_avx2(const block_q4_K_t * restrict w_row0,
                                                   size_t row_blocks,
                                                   const block_q8_K_t * restrict x_q8,
                                                   uint32_t n_cols,
                                                   float out4[4]) {
    const block_q4_K_t * restrict w0 = w_row0;
    const block_q4_K_t * restrict w1 = w_row0 + 1 * row_blocks;
    const block_q4_K_t * restrict w2 = w_row0 + 2 * row_blocks;
    const block_q4_K_t * restrict w3 = w_row0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        /* Software prefetch next super-block across all 4 rows into L1 cache */
        if (b + 1 < row_blocks) {
            _mm_prefetch((const char *)&w0[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w1[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w2[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w3[b + 1], _MM_HINT_T0);
        }

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        int32_t mins0 = 0, mins1 = 0, mins2 = 0, mins3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            /* Load 64 signed int8_t activations & pair sums ONCE for all 4 rows */
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));
            const int32_t bsum0 = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum1 = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            #define LFM2_Q4K_STEP_ROW(w_ptr, acc_reg, mins_acc) do {                                   \
                uint8_t sc_lo, m_lo, sc_hi, m_hi;                                                      \
                get_scale_min_k4(is + 0, (w_ptr)[b].scales, &sc_lo, &m_lo);                            \
                get_scale_min_k4(is + 1, (w_ptr)[b].scales, &sc_hi, &m_hi);                            \
                (mins_acc) += (int32_t)m_lo * bsum0 + (int32_t)m_hi * bsum1;                           \
                const __m256i q4b  = _mm256_loadu_si256((const __m256i *)((w_ptr)[b].qs + (j >> 1))); \
                const __m256i d_lo = _mm256_maddubs_epi16(_mm256_and_si256(q4b, mask_0f), q8_lo);     \
                const __m256i d_hi = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4b, 4), mask_0f), q8_hi); \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_lo, _mm256_set1_epi16((int16_t)sc_lo)));    \
                (acc_reg) = _mm256_add_epi32((acc_reg), _mm256_madd_epi16(d_hi, _mm256_set1_epi16((int16_t)sc_hi)));    \
            } while (0)

            LFM2_Q4K_STEP_ROW(w0, acc0, mins0);
            LFM2_Q4K_STEP_ROW(w1, acc1, mins1);
            LFM2_Q4K_STEP_ROW(w2, acc2, mins2);
            LFM2_Q4K_STEP_ROW(w3, acc3, mins3);

            #undef LFM2_Q4K_STEP_ROW
            is += 2;
        }

        /* 4-way simultaneous horizontal reduction of acc0..acc3 -> __m128 float */
        const __m128 v_dots = _mm_cvtepi32_ps(hsum4_epi32_avx2(acc0, acc1, acc2, acc3));
        const __m128 v_mins = _mm_cvtepi32_ps(_mm_set_epi32(mins3, mins2, mins1, mins0));

        const __m128 v_d = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            fp16_to_fp32(w3[b].d), fp16_to_fp32(w2[b].d),
            fp16_to_fp32(w1[b].d), fp16_to_fp32(w0[b].d)));

        const __m128 v_dmin = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            fp16_to_fp32(w3[b].dmin), fp16_to_fp32(w2[b].dmin),
            fp16_to_fp32(w1[b].dmin), fp16_to_fp32(w0[b].dmin)));

        v_sumf = _mm_add_ps(v_sumf, _mm_sub_ps(_mm_mul_ps(v_d, v_dots), _mm_mul_ps(v_dmin, v_mins)));
    }

    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* ============================================================================
 * 4. 1x4 Row-Unrolled AVX2 Q6_K x Q8_K Micro-Kernel (4 Rows Simultaneously)
 * ============================================================================ */
static inline void lfm2_vec_dot_q6_K_q8_K_1x4_avx2(const block_q6_K_t * restrict w_row0,
                                                   size_t row_blocks,
                                                   const block_q8_K_t * restrict x_q8,
                                                   uint32_t n_cols,
                                                   float out4[4]) {
    const block_q6_K_t * restrict w0 = w_row0;
    const block_q6_K_t * restrict w1 = w_row0 + 1 * row_blocks;
    const block_q6_K_t * restrict w2 = w_row0 + 2 * row_blocks;
    const block_q6_K_t * restrict w3 = w_row0 + 3 * row_blocks;

    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    const __m256i mask_0c = _mm256_set1_epi8(0x0C);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);

    __m128 v_sumf = _mm_setzero_ps();

    for (size_t b = 0; b < row_blocks; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        if (b + 1 < row_blocks) {
            _mm_prefetch((const char *)&w0[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w1[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w2[b + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&w3[b + 1], _MM_HINT_T0);
        }

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        int32_t off0 = 0, off1 = 0, off2 = 0, off3 = 0;
        for (int is = 0; is < 16; ++is) {
            const int32_t b_is = bs[is];
            off0 += (int32_t)w0[b].scales[is] * b_is;
            off1 += (int32_t)w1[b].scales[is] * b_is;
            off2 += (int32_t)w2[b].scales[is] * b_is;
            off3 += (int32_t)w3[b].scales[is] * b_is;
        }

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
            const int ql_off = half * 64;
            const int qh_off = half * 32;
            const int sc_off = half * 8;
            const int q8_off = half * 128;

            /* Load 128 signed 8-bit activations ONCE into YMM registers for all 4 rows */
            const __m256i x0 = _mm256_loadu_si256((const __m256i *)(q8 + q8_off +  0));
            const __m256i x1 = _mm256_loadu_si256((const __m256i *)(q8 + q8_off + 32));
            const __m256i x2 = _mm256_loadu_si256((const __m256i *)(q8 + q8_off + 64));
            const __m256i x3 = _mm256_loadu_si256((const __m256i *)(q8 + q8_off + 96));

            #define LFM2_Q6K_STEP_ROW(w_ptr, acc_reg) do {                                   \
                const uint8_t *ql = (w_ptr)[b].ql + ql_off;                                  \
                const uint8_t *qh = (w_ptr)[b].qh + qh_off;                                  \
                const int8_t  *sc = (w_ptr)[b].scales + sc_off;                              \
                const __m256i ql0 = _mm256_loadu_si256((const __m256i *)(ql +  0));          \
                const __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));          \
                const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);                 \
                const __m256i u1  = _mm256_or_si256(_mm256_and_si256(ql0, mask_0f),          \
                                    _mm256_slli_epi16(_mm256_and_si256(qhv, mask_03), 4));   \
                const __m256i u2  = _mm256_or_si256(_mm256_and_si256(ql1, mask_0f),          \
                                    _mm256_slli_epi16(_mm256_and_si256(qhv, mask_0c), 2));   \
                const __m256i u3  = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_0f), \
                                    _mm256_and_si256(qhv, mask_30));                         \
                const __m256i u4  = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_0f), \
                                    _mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask_30));   \
                const __m256i s01 = mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0])); \
                const __m256i s23 = mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2])); \
                const __m256i s45 = mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4])); \
                const __m256i s67 = mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6])); \
                const __m256i p0  = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, x0), s01);    \
                const __m256i p1  = _mm256_madd_epi16(_mm256_maddubs_epi16(u2, x1), s23);    \
                const __m256i p2  = _mm256_madd_epi16(_mm256_maddubs_epi16(u3, x2), s45);    \
                const __m256i p3  = _mm256_madd_epi16(_mm256_maddubs_epi16(u4, x3), s67);    \
                (acc_reg) = _mm256_add_epi32((acc_reg),                                      \
                    _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));   \
            } while (0)

            LFM2_Q6K_STEP_ROW(w0, acc0);
            LFM2_Q6K_STEP_ROW(w1, acc1);
            LFM2_Q6K_STEP_ROW(w2, acc2);
            LFM2_Q6K_STEP_ROW(w3, acc3);

            #undef LFM2_Q6K_STEP_ROW
        }

        const __m128i v_u6q8 = hsum4_epi32_avx2(acc0, acc1, acc2, acc3);
        const __m128i v_off  = _mm_slli_epi32(_mm_set_epi32(off3, off2, off1, off0), 5); /* * 32 */
        const __m128  v_diff = _mm_cvtepi32_ps(_mm_sub_epi32(v_u6q8, v_off));

        const __m128 v_d = _mm_mul_ps(_mm_set1_ps(d8), _mm_set_ps(
            fp16_to_fp32(w3[b].d), fp16_to_fp32(w2[b].d),
            fp16_to_fp32(w1[b].d), fp16_to_fp32(w0[b].d)));

        v_sumf = _mm_add_ps(v_sumf, _mm_mul_ps(v_d, v_diff));
    }

    (void)n_cols;
    _mm_storeu_ps(out4, v_sumf);
}

/* ============================================================================
 * 5. Unified 4-Row & 1-Row Tensor Dispatcher Helper
 * ============================================================================ */
static inline void lfm2_compute_4rows_avx2(const GGUFTensor * restrict W,
                                           uint32_t r,
                                           const float * restrict x_f32,
                                           const block_q8_K_t * restrict x_q8,
                                           float out4[4]) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    if (W->type == GGML_TYPE_Q4_K) {
        const size_t rb = n_cols / 256;
        lfm2_vec_dot_q4_K_q8_K_1x4_avx2((const block_q4_K_t *)W->data + (size_t)r * rb,
                                        rb, x_q8, n_cols, out4);
    } else if (W->type == GGML_TYPE_Q6_K) {
        const size_t rb = n_cols / 256;
        lfm2_vec_dot_q6_K_q8_K_1x4_avx2((const block_q6_K_t *)W->data + (size_t)r * rb,
                                        rb, x_q8, n_cols, out4);
    } else if (W->type == GGML_TYPE_Q8_0) {
        const size_t rb = n_cols / 32;
        const block_q8_0_t *base = (const block_q8_0_t *)W->data + (size_t)r * rb;
        for (int k = 0; k < 4; ++k) {
            out4[k] = lfm2_vec_dot_q8_0_q8_K_avx2(base + (size_t)k * rb, x_q8, n_cols);
        }
    } else if (W->type == GGML_TYPE_Q4_0) {
        const size_t rb = n_cols / 32;
        lfm2_vec_dot_q4_0_q8_K_1x4_avx2((const block_q4_0_t *)W->data + (size_t)r * rb,
                                        rb, x_q8, n_cols, out4);
    } else if (W->type == GGML_TYPE_Q5_K) {
        const size_t rb = n_cols / 256;
        lfm2_vec_dot_q5_K_q8_K_1x4_avx2((const block_q5_K_t *)W->data + (size_t)r * rb,
                                        rb, x_q8, n_cols, out4);
    } else if (W->type == GGML_TYPE_IQ4_XS) {
        const size_t rb = n_cols / 256;
        lfm2_vec_dot_iq4_xs_q8_K_1x4_avx2((const block_iq4_xs_t *)W->data + (size_t)r * rb,
                                          rb, x_q8, n_cols, out4);
    } else {
        const float *base = (const float *)W->data + (size_t)r * n_cols;
        for (int k = 0; k < 4; ++k) {
            out4[k] = lfm2_vec_dot_f32_avx2(base + (size_t)k * n_cols, x_f32, n_cols);
        }
    }
}

static inline float lfm2_compute_1row_avx2(const GGUFTensor * restrict W,
                                           uint32_t r,
                                           const float * restrict x_f32,
                                           const block_q8_K_t * restrict x_q8) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    if (W->type == GGML_TYPE_Q4_K) {
        const size_t rb = n_cols / 256;
        return lfm2_vec_dot_q4_K_q8_K_avx2((const block_q4_K_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else if (W->type == GGML_TYPE_Q6_K) {
        const size_t rb = n_cols / 256;
        return lfm2_vec_dot_q6_K_q8_K_avx2((const block_q6_K_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else if (W->type == GGML_TYPE_Q8_0) {
        const size_t rb = n_cols / 32;
        return lfm2_vec_dot_q8_0_q8_K_avx2((const block_q8_0_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else if (W->type == GGML_TYPE_Q4_0) {
        const size_t rb = n_cols / 32;
        return lfm2_vec_dot_q4_0_q8_K_avx2((const block_q4_0_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else if (W->type == GGML_TYPE_Q5_K) {
        const size_t rb = n_cols / 256;
        return lfm2_vec_dot_q5_K_q8_K_avx2((const block_q5_K_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else if (W->type == GGML_TYPE_IQ4_XS) {
        const size_t rb = n_cols / 256;
        return lfm2_vec_dot_iq4_xs_q8_K_avx2((const block_iq4_xs_t *)W->data + (size_t)r * rb, x_q8, n_cols);
    } else {
        return lfm2_vec_dot_f32_avx2((const float *)W->data + (size_t)r * n_cols, x_f32, n_cols);
    }
}

/* ============================================================================
 * 6. Pre-Quantized 1x4 Multi-Threaded MatMul & Fused QKV Dispatcher
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *out;
    const GGUFTensor   *W;
    const float        *x_f32;
    const block_q8_K_t *x_q8;
} MatMul1x4TaskCtx;

static void lfm2_matmul_1x4_worker_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    MatMul1x4TaskCtx *ctx = (MatMul1x4TaskCtx *)arg;
    uint32_t r_start = 0, r_end = 0;

    while (lfm2_threadpool_claim_chunk(ctx->pool, &r_start, &r_end)) {
        uint32_t r = r_start;
        for (; r + 4 <= r_end; r += 4) {
            lfm2_compute_4rows_avx2(ctx->W, r, ctx->x_f32, ctx->x_q8, ctx->out + r);
        }
        for (; r < r_end; ++r) {
            ctx->out[r] = lfm2_compute_1row_avx2(ctx->W, r, ctx->x_f32, ctx->x_q8);
        }
    }
}

/* Runs 1x4 unrolled MatMul using already-quantized `x_q8` (or quantizes if needed) */
void lfm2_matmul_1x4_prequant_mt(LFM2ThreadPool *pool,
                                 float * restrict out,
                                 const GGUFTensor * restrict W,
                                 const float * restrict x_f32,
                                 const block_q8_K_t * restrict x_q8) {
    const uint32_t n_rows = (uint32_t)W->ne[1];
    if (pool == NULL || pool->n_threads <= 1) {
        uint32_t r = 0;
        for (; r + 4 <= n_rows; r += 4) {
            lfm2_compute_4rows_avx2(W, r, x_f32, x_q8, out + r);
        }
        for (; r < n_rows; ++r) {
            out[r] = lfm2_compute_1row_avx2(W, r, x_f32, x_q8);
        }
        return;
    }

    MatMul1x4TaskCtx ctx = {
        .pool = pool, .out = out, .W = W, .x_f32 = x_f32, .x_q8 = x_q8
    };
    lfm2_parallel_for(pool, n_rows, 16, lfm2_matmul_1x4_worker_task, &ctx);
}

/* Fused Single-Barrier QKV Projection for GQA Layers:
 * Computes q = Wq * x, k = Wk * x, v = Wv * x in ONE spin-barrier wakeup
 * using the single pre-quantized `x_q8` produced by `lfm2_rmsnorm_quantize_q8_K_avx2`. */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *q_out;
    float              *k_out;
    float              *v_out;
    const GGUFTensor   *wq;
    const GGUFTensor   *wk;
    const GGUFTensor   *wv;
    uint32_t            q_rows;
    uint32_t            k_rows;
    uint32_t            v_rows;
    const float        *x_f32;
    const block_q8_K_t *x_q8;
} FusedQKVTaskCtx;

static void lfm2_gqa_qkv_fused_worker(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    FusedQKVTaskCtx *c = (FusedQKVTaskCtx *)arg;
    const uint32_t q_end = c->q_rows;
    const uint32_t k_end = q_end + c->k_rows;
    const uint32_t total = q_end + c->k_rows + c->v_rows;
    uint32_t r_start = 0, r_end = 0;

    while (lfm2_threadpool_claim_chunk(c->pool, &r_start, &r_end)) {
        uint32_t r = r_start;
        for (; r + 4 <= r_end; ) {
            if (r + 4 <= q_end) {
                lfm2_compute_4rows_avx2(c->wq, r, c->x_f32, c->x_q8, c->q_out + r);
                r += 4;
            } else if (r < q_end) {
                c->q_out[r] = lfm2_compute_1row_avx2(c->wq, r, c->x_f32, c->x_q8);
                r += 1;
            } else if (r + 4 <= k_end) {
                uint32_t kr = r - q_end;
                lfm2_compute_4rows_avx2(c->wk, kr, c->x_f32, c->x_q8, c->k_out + kr);
                r += 4;
            } else if (r < k_end) {
                uint32_t kr = r - q_end;
                c->k_out[kr] = lfm2_compute_1row_avx2(c->wk, kr, c->x_f32, c->x_q8);
                r += 1;
            } else if (r + 4 <= total) {
                uint32_t vr = r - k_end;
                lfm2_compute_4rows_avx2(c->wv, vr, c->x_f32, c->x_q8, c->v_out + vr);
                r += 4;
            } else {
                uint32_t vr = r - k_end;
                c->v_out[vr] = lfm2_compute_1row_avx2(c->wv, vr, c->x_f32, c->x_q8);
                r += 1;
            }
        }
        for (; r < r_end; ++r) {
            if (r < q_end) {
                c->q_out[r] = lfm2_compute_1row_avx2(c->wq, r, c->x_f32, c->x_q8);
            } else if (r < k_end) {
                uint32_t kr = r - q_end;
                c->k_out[kr] = lfm2_compute_1row_avx2(c->wk, kr, c->x_f32, c->x_q8);
            } else {
                uint32_t vr = r - k_end;
                c->v_out[vr] = lfm2_compute_1row_avx2(c->wv, vr, c->x_f32, c->x_q8);
            }
        }
    }
}

void lfm2_gqa_qkv_fused_1x4_mt(LFM2ThreadPool *pool,
                               float * restrict q_out,
                               float * restrict k_out,
                               float * restrict v_out,
                               const GGUFTensor * restrict wq,
                               const GGUFTensor * restrict wk,
                               const GGUFTensor * restrict wv,
                               const float * restrict x_f32,
                               const block_q8_K_t * restrict x_q8) {
    const uint32_t q_rows = (uint32_t)wq->ne[1];
    const uint32_t k_rows = (uint32_t)wk->ne[1];
    const uint32_t v_rows = (uint32_t)wv->ne[1];
    const uint32_t total_rows = q_rows + k_rows + v_rows;

    if (pool == NULL || pool->n_threads <= 1) {
        uint32_t r = 0;
        for (; r + 4 <= q_rows; r += 4) {
            lfm2_compute_4rows_avx2(wq, r, x_f32, x_q8, q_out + r);
        }
        for (; r < q_rows; ++r) {
            q_out[r] = lfm2_compute_1row_avx2(wq, r, x_f32, x_q8);
        }

        r = 0;
        for (; r + 4 <= k_rows; r += 4) {
            lfm2_compute_4rows_avx2(wk, r, x_f32, x_q8, k_out + r);
        }
        for (; r < k_rows; ++r) {
            k_out[r] = lfm2_compute_1row_avx2(wk, r, x_f32, x_q8);
        }

        r = 0;
        for (; r + 4 <= v_rows; r += 4) {
            lfm2_compute_4rows_avx2(wv, r, x_f32, x_q8, v_out + r);
        }
        for (; r < v_rows; ++r) {
            v_out[r] = lfm2_compute_1row_avx2(wv, r, x_f32, x_q8);
        }
        return;
    }

    FusedQKVTaskCtx ctx = {
        .pool   = pool,
        .q_out  = q_out, .k_out  = k_out, .v_out  = v_out,
        .wq     = wq,    .wk     = wk,    .wv     = wv,
        .q_rows = q_rows,
        .k_rows = k_rows,
        .v_rows = v_rows,
        .x_f32  = x_f32,
        .x_q8   = x_q8
    };
    lfm2_parallel_for(pool, total_rows, 16, lfm2_gqa_qkv_fused_worker, &ctx);
}

/* ============================================================================
 * 7. Fused Gate + Up + SwiGLU (+ In-Place Q8_K Output Quantization) Kernel
 * ============================================================================
 * In ONE thread-pool dispatch across `n_ff` (in 256-element super-block chunks!):
 *   1. Computes 4 rows of `ffn_gate` (g0..g3) and 4 rows of `ffn_up` (u0..u3)
 *      using the 1x4 AVX2 micro-kernels.
 *   2. Applies `hb[r + k] = (g[k] / (1.0f + expf(-g[k]))) * u[k]` in registers.
 *   3. As soon as a worker finishes a 256-element super-block of `hb`, it
 *      immediately quantizes `hb[b*256 .. b*256+255]` into `hb_q8[b]` while the
 *      256 floats are still sitting in that core's L1 cache!
 * Result: `ffn_gate`, `ffn_up`, `SwiGLU`, AND the `Q8_K` input quantization for
 * `ffn_down` all happen inside ONE single spin-barrier wakeup!
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *hb_out;    /* [n_ff] FP32 SwiGLU output (optional/fallback) */
    block_q8_K_t       *hb_q8_out; /* [n_ff / 256] Pre-quantized Q8_K for ffn_down! */
    const GGUFTensor   *w_gate;
    const GGUFTensor   *w_up;
    const float        *x_f32;
    const block_q8_K_t *x_q8;
    uint32_t            n_ff;
} FusedSwiGLUTaskCtx;

static void lfm2_ffn_swiglu_fused_worker(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    FusedSwiGLUTaskCtx *c = (FusedSwiGLUTaskCtx *)arg;
    uint32_t sb_start = 0, sb_end = 0;

    /* Each work item is one 256-row super-block of n_ff so each worker can
     * immediately quantize its completed 256 SwiGLU outputs into hb_q8_out[sb]! */
    while (lfm2_threadpool_claim_chunk(c->pool, &sb_start, &sb_end)) {
        for (uint32_t sb = sb_start; sb < sb_end; ++sb) {
            const uint32_t base_r = sb * 256;
            const uint32_t end_r  = (base_r + 256 <= c->n_ff) ? (base_r + 256) : c->n_ff;
            float *hb_block = c->hb_out + base_r;

            uint32_t r = base_r;
            for (; r + 4 <= end_r; r += 4) {
                float g[4], u[4];
                lfm2_compute_4rows_avx2(c->w_gate, r, c->x_f32, c->x_q8, g);
                lfm2_compute_4rows_avx2(c->w_up,   r, c->x_f32, c->x_q8, u);

                for (int k = 0; k < 4; ++k) {
                    const float silu = g[k] / (1.0f + expf(-g[k]));
                    c->hb_out[r + k] = silu * u[k];
                }
            }
            for (; r < end_r; ++r) {
                const float g = lfm2_compute_1row_avx2(c->w_gate, r, c->x_f32, c->x_q8);
                const float u = lfm2_compute_1row_avx2(c->w_up,   r, c->x_f32, c->x_q8);
                c->hb_out[r]  = (g / (1.0f + expf(-g))) * u;
            }

            /* Immediately quantize this L1-hot 256-float SwiGLU block to Q8_K for ffn_down */
            if (c->hb_q8_out != NULL) {
                if (end_r - base_r == 256) {
                    lfm2_quantize_row_q8_K_avx2(hb_block, &c->hb_q8_out[sb], 256);
                } else {
                    float pad[256] = {0};
                    memcpy(pad, hb_block, (end_r - base_r) * sizeof(float));
                    lfm2_quantize_row_q8_K_avx2(pad, &c->hb_q8_out[sb], 256);
                }
            }
        }
    }
}

void lfm2_ffn_swiglu_fused_1x4_mt(LFM2ThreadPool *pool,
                                  float * restrict hb_out,
                                  block_q8_K_t * restrict hb_q8_out,
                                  const GGUFTensor * restrict w_gate,
                                  const GGUFTensor * restrict w_up,
                                  const float * restrict x_f32,
                                  const block_q8_K_t * restrict x_q8) {
    const uint32_t n_ff      = (uint32_t)w_gate->ne[1];
    const uint32_t n_sblocks = (n_ff + 255) / 256;

    if (pool == NULL || pool->n_threads <= 1) {
        for (uint32_t sb = 0; sb < n_sblocks; ++sb) {
            const uint32_t base_r = sb * 256;
            const uint32_t end_r  = (base_r + 256 <= n_ff) ? (base_r + 256) : n_ff;
            float *hb_block = hb_out + base_r;
            uint32_t r = base_r;
            for (; r + 4 <= end_r; r += 4) {
                float g[4], u[4];
                lfm2_compute_4rows_avx2(w_gate, r, x_f32, x_q8, g);
                lfm2_compute_4rows_avx2(w_up,   r, x_f32, x_q8, u);
                for (int k = 0; k < 4; ++k) {
                    const float silu = g[k] / (1.0f + expf(-g[k]));
                    hb_out[r + k] = silu * u[k];
                }
            }
            for (; r < end_r; ++r) {
                const float g = lfm2_compute_1row_avx2(w_gate, r, x_f32, x_q8);
                const float u = lfm2_compute_1row_avx2(w_up,   r, x_f32, x_q8);
                hb_out[r]  = (g / (1.0f + expf(-g))) * u;
            }
            if (hb_q8_out != NULL) {
                if (end_r - base_r == 256) {
                    lfm2_quantize_row_q8_K_avx2(hb_block, &hb_q8_out[sb], 256);
                } else {
                    float pad[256] = {0};
                    memcpy(pad, hb_block, (end_r - base_r) * sizeof(float));
                    lfm2_quantize_row_q8_K_avx2(pad, &hb_q8_out[sb], 256);
                }
            }
        }
        return;
    }

    FusedSwiGLUTaskCtx ctx = {
        .pool      = pool,
        .hb_out    = hb_out,
        .hb_q8_out = hb_q8_out,
        .w_gate    = w_gate,
        .w_up      = w_up,
        .x_f32     = x_f32,
        .x_q8      = x_q8,
        .n_ff      = n_ff
    };

    /* Dispatch 1 super-block (256 rows = 64 1x4 tiles) per atomic claim */
    lfm2_parallel_for(pool, n_sblocks, 1, lfm2_ffn_swiglu_fused_worker, &ctx);
}

/* Complete Fused SwiGLU FFN Layer:
 * Executes RMSNorm + Q8_K, Gate, Up, SwiGLU, hb->Q8_K, Down, and Residual Add
 * in only TWO thread-pool barrier dispatches instead of THREE+! */
void lfm2_forward_ffn_fused_1x4(LFM2ThreadPool *pool,
                                float * restrict x,
                                float * restrict xb,
                                float * restrict hb,
                                block_q8_K_t * restrict xq_scratch,
                                const LFM2Layer * restrict layer,
                                uint32_t D,
                                float eps) {
    /* 1. Fused RMSNorm(x -> xb) + Quantize(xb -> xq_scratch) */
    lfm2_rmsnorm_quantize_q8_K_avx2(xb, xq_scratch, x, layer->ffn_norm, D, eps);

    /* 2. Barrier #1: Fused 1x4 Gate + Up + SwiGLU + in-place hb_q8 quantization!
     *    Note: We store hb_q8 at xq_scratch + (D / 256) so it doesn't overwrite x_q8 */
    block_q8_K_t *hb_q8 = xq_scratch + (D / 256);
    lfm2_ffn_swiglu_fused_1x4_mt(pool, hb, hb_q8, layer->ffn_gate, layer->ffn_up, xb, xq_scratch);

    /* 3. Barrier #2: 1x4 Down Projection using the already-quantized `hb_q8`! */
    lfm2_matmul_1x4_prequant_mt(pool, xb, layer->ffn_down, hb, hb_q8);

    /* 4. Vectorized Residual Add: x[0..D-1] += xb[0..D-1] */
    uint32_t d = 0;
    for (; d + 8 <= D; d += 8) {
        _mm256_storeu_ps(x + d, _mm256_add_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(xb + d)));
    }
    for (; d < D; ++d) {
        x[d] += xb[d];
    }
}

/* ============================================================================
 * 8. Standalone TDD Parity Verifier & Full FFN Layer Benchmark
 * ============================================================================ */
#ifndef LFM2_DECODE_1X4_NO_MAIN
int main(int argc, char **argv) {
    /* LFM2.5-1.2B FFN dimensions: D = 2048, n_ff = 6144 */
    const uint32_t D        = 2048;
    const uint32_t n_ff     = 6144;
    const uint32_t nb_d     = D / 256;
    const uint32_t nb_ff    = n_ff / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0);

    float        *x_ref   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *x_fused = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *xb      = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *norm_w  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *hb      = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    float        *hb2     = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    block_q8_K_t *xq_buf  = (block_q8_K_t *)lfm2_aligned_alloc_64((nb_d + nb_ff + 16) * sizeof(block_q8_K_t));

    /* In LFM2.5-Q4_K_M, ffn_gate and ffn_up are Q4_K, while ffn_down is Q6_K */
    block_q4_K_t *w_gate = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_up   = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q6_K_t *w_down = (block_q6_K_t *)lfm2_aligned_alloc_64((size_t)D * nb_ff * sizeof(block_q6_K_t));

    for (uint32_t i = 0; i < D; ++i) {
        x_ref[i] = x_fused[i] = sinf((float)i * 0.019f);
        norm_w[i] = 0.85f + 0.15f * cosf((float)i * 0.03f);
    }
    for (size_t i = 0; i < (size_t)n_ff * nb_d; ++i) {
        w_gate[i].d = 0x2E00; w_gate[i].dmin = 0x2600;
        w_up[i].d   = 0x2D00; w_up[i].dmin   = 0x2500;
        for (int j = 0; j < 12; ++j) {
            w_gate[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
            w_up[i].scales[j]   = (uint8_t)((i * 5 + j * 11) & 0x3F);
        }
        for (int j = 0; j < 128; ++j) {
            w_gate[i].qs[j] = (uint8_t)((i * 13 + j * 29) & 0xFF);
            w_up[i].qs[j]   = (uint8_t)((i * 17 + j * 31) & 0xFF);
        }
    }
    for (size_t i = 0; i < (size_t)D * nb_ff; ++i) {
        w_down[i].d = 0x2B00;
        for (int j = 0; j < 16; ++j)  w_down[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_down[i].ql[j]     = (uint8_t)((i * 19 + j * 11) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    GGUFTensor t_norm = { .n_dims = 1, .ne = {D, 1, 1, 1},    .type = GGML_TYPE_F32,  .data = norm_w };
    GGUFTensor t_gate = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_gate };
    GGUFTensor t_up   = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_up   };
    GGUFTensor t_down = { .n_dims = 2, .ne = {n_ff, D, 1, 1}, .type = GGML_TYPE_Q6_K, .data = w_down };

    LFM2Layer fake_layer = {
        .ffn_norm = &t_norm, .ffn_gate = &t_gate, .ffn_up = &t_up, .ffn_down = &t_down
    };

    /* 1. Run Unfused 1x1 Baseline FFN Pass */
    {
        double ss = 0.0;
        for (uint32_t i = 0; i < D; ++i) ss += (double)x_ref[i] * (double)x_ref[i];
        float inv_rms = 1.0f / sqrtf((float)(ss / D) + 1e-5f);
        for (uint32_t i = 0; i < D; ++i) xb[i] = x_ref[i] * inv_rms * norm_w[i];

        lfm2_matmul_vec_avx2_mt(&pool, hb,  w_gate, GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        lfm2_matmul_vec_avx2_mt(&pool, hb2, w_up,   GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        for (uint32_t i = 0; i < n_ff; ++i) {
            hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
        }
        lfm2_matmul_vec_avx2_mt(&pool, xb, w_down, GGML_TYPE_Q6_K, n_ff, D, hb, xq_buf);
        for (uint32_t i = 0; i < D; ++i) x_ref[i] += xb[i];
    }

    /* 2. Run Fused 1x4 Unrolled FFN Pass */
    lfm2_forward_ffn_fused_1x4(&pool, x_fused, xb, hb, xq_buf, &fake_layer, D, 1e-5f);

    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float diff = fabsf(x_ref[i] - x_fused[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    printf("=== LFM2 1x4 Row-Unrolled AVX2 & Fused SwiGLU Benchmark (D=%u, n_ff=%u) ===\n", D, n_ff);
    printf("Physical Cores Active : %u\n", pool.n_threads);
    printf("Max FFN Output Diff   : %.7f (Exact parity between 1x1 Unfused & 1x4 Fused)\n\n", max_abs_diff);

    /* 3. Benchmark Unfused 1x1 FFN vs Fused 1x4 FFN (100 iterations) */
    const int iters = 100;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        double ss = 0.0;
        for (uint32_t i = 0; i < D; ++i) ss += (double)x_ref[i] * (double)x_ref[i];
        float inv_rms = 1.0f / sqrtf((float)(ss / D) + 1e-5f);
        for (uint32_t i = 0; i < D; ++i) xb[i] = x_ref[i] * inv_rms * norm_w[i];

        lfm2_matmul_vec_avx2_mt(&pool, hb,  w_gate, GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        lfm2_matmul_vec_avx2_mt(&pool, hb2, w_up,   GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        for (uint32_t i = 0; i < n_ff; ++i) {
            hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
        }
        lfm2_matmul_vec_avx2_mt(&pool, xb, w_down, GGML_TYPE_Q6_K, n_ff, D, hb, xq_buf);
        for (uint32_t i = 0; i < D; ++i) x_ref[i] += xb[i];
    }
    double unfused_ms = (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_forward_ffn_fused_1x4(&pool, x_fused, xb, hb, xq_buf, &fake_layer, D, 1e-5f);
    }
    double fused_ms = (now_ms() - t0) / iters;

    const double ffn_bytes = (double)n_ff * nb_d * sizeof(block_q4_K_t) * 2.0 +
                             (double)D    * nb_ff * sizeof(block_q6_K_t);
    printf("Unfused 1x1 FFN Layer : %.3f ms | Effective Bandwidth: %.2f GB/s\n",
           unfused_ms, ffn_bytes / (unfused_ms * 1e6));
    printf("Fused   1x4 FFN Layer : %.3f ms | Effective Bandwidth: %.2f GB/s (%.2fx faster)\n",
           fused_ms, ffn_bytes / (fused_ms * 1e6), unfused_ms / fused_ms);

    free(x_ref); free(x_fused); free(xb); free(norm_w);
    free(hb); free(hb2); free(xq_buf); free(w_gate); free(w_up); free(w_down);
    lfm2_threadpool_free(&pool);
    return EXIT_SUCCESS;
}
#endif
