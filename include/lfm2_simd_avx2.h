#ifndef LFM2_SIMD_AVX2_H
#define LFM2_SIMD_AVX2_H

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define LFM2_HAS_AVX2 1
#else
#define LFM2_HAS_AVX2 0
#endif

#ifndef LFM2_TARGET_AVX2
#define LFM2_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif
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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#include <pmmintrin.h>

static inline void lfm2_set_denormals_zero_mode(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

static inline bool lfm2_get_denormals_zero_mode(void) {
    return (_MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON) &&
           (_MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON);
}
#elif defined(__aarch64__)
static inline void lfm2_set_denormals_zero_mode(void) {
    uint64_t fpcr = 0;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24) | (1ULL << 19); /* Bit 24 = FZ (FP32/64), Bit 19 = FZ16 (FP16 FZ) */
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
}

static inline bool lfm2_get_denormals_zero_mode(void) {
    uint64_t fpcr = 0;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    return ((fpcr & (1ULL << 24)) != 0) && ((fpcr & (1ULL << 19)) != 0);
}
#else
static inline void lfm2_set_denormals_zero_mode(void) {
    /* No-op on other architectures */
}

static inline bool lfm2_get_denormals_zero_mode(void) {
    return true;
}
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * 1. Quantized Block Definitions (GGML Binary-Compatible Layout)
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
typedef struct {
    uint16_t d;           /* FP16 delta scale */
    int8_t   qs[32];      /* 32 signed 8-bit weights */
} block_q8_0_t;
#endif

#ifndef LFM2_BLOCK_Q4_K_DEFINED
#define LFM2_BLOCK_Q4_K_DEFINED
typedef struct {
    uint16_t d;           /* FP16 super-block scale for quantized scales */
    uint16_t dmin;        /* FP16 super-block scale for quantized mins   */
    uint8_t  scales[12];  /* 6-bit packed scales and mins for 8 sub-blocks */
    uint8_t  qs[128];     /* 256 4-bit unsigned weights (low/high nibbles) */
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
typedef struct {
    uint8_t  ql[128];     /* Lower 4 bits of 256 unsigned 6-bit weights  */
    uint8_t  qh[64];      /* Upper 2 bits of 256 unsigned 6-bit weights  */
    int8_t   scales[16];  /* 8-bit signed sub-block scales (16 weights/ea) */
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
/* Activation super-block: 256 elements quantized once per matmul */
typedef struct {
    float   d;            /* FP32 scale: max_abs / 127.0f                */
    int8_t  qs[256];      /* 256 signed 8-bit quantized activations      */
    int16_t bsums[16];    /* Precomputed sum of each 16-element sub-block */
} block_q8_K_t;
#endif

#pragma pack(pop)

/* IEEE 754 FP16 to FP32 conversion */
static inline float lfm2_fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
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

/* Helper to unpack 6-bit scale and min for sub-block j (0..7) in Q4_K */
static inline void lfm2_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

#ifndef LFM2_KVALUES_IQ4NL_DEFINED
#define LFM2_KVALUES_IQ4NL_DEFINED
/* Official GGML non-linear 4-bit importance grid for IQ4_NL and IQ4_XS */
static const _Alignas(16) int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};
#endif

#ifndef LFM2_GET_SCALE_IQ4_XS_DEFINED
#define LFM2_GET_SCALE_IQ4_XS_DEFINED
#define get_scale_iq4_xs get_scale_iq4_xs
static inline int get_scale_iq4_xs(int ib, const uint8_t scales_l[4], uint16_t scales_h) {
    const int low4  = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int high2 = ((scales_h >> (2 * ib)) & 0x03) << 4;
    return (low4 | high2) - 32;
}
#endif

#if LFM2_HAS_AVX2

/* ============================================================================
 * 2. AVX2 Intrinsics Helpers
 * ============================================================================ */

static inline __m256i lfm2_mm256_set_m128i(__m128i hi, __m128i lo) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* Horizontally sums 8 x int32_t lanes in a __m256i register to a scalar int32_t */
static inline int32_t lfm2_hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum128);
}

/* Horizontally sums 8 x float lanes in a __m256 register to a scalar float */
static inline float lfm2_hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

/* ============================================================================
 * 2b. Portable Reference Activation Quantizer: float x[n_cols] -> block_q8_K_t y[(n_cols+255)/256]
 * ============================================================================ */
static inline void lfm2_quantize_block_q8_K_fallback(const float * restrict xb, block_q8_K_t * restrict yb) {
    float max_abs = 0.0f;
    for (int i = 0; i < 256; ++i) {
        float v = fabsf(xb[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0f) {
        yb->d = 0.0f;
        memset(yb->qs, 0, sizeof(yb->qs));
        memset(yb->bsums, 0, sizeof(yb->bsums));
        return;
    }
    const float d = max_abs / 127.0f;
    const float iscale = 127.0f / max_abs;
    yb->d = d;

    for (int j = 0; j < 16; ++j) {
        int16_t bsum = 0;
        for (int l = 0; l < 16; ++l) {
            float v = xb[j * 16 + l] * iscale;
            int8_t q = (int8_t)roundf(v);
            yb->qs[j * 16 + l] = q;
            bsum += q;
        }
        yb->bsums[j] = bsum;
    }
}

static inline void lfm2_quantize_row_q8_K_fallback(const float * restrict x, block_q8_K_t * restrict y, uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        lfm2_quantize_block_q8_K_fallback(x + (size_t)b * 256, &y[b]);
    }
    const uint32_t rem = n_cols % 256;
    if (rem > 0) {
        float pad[256];
        memset(pad, 0, sizeof(pad));
        memcpy(pad, x + (size_t)nb * 256, (size_t)rem * sizeof(float));
        lfm2_quantize_block_q8_K_fallback(pad, &y[nb]);
    }
}

/* ============================================================================
 * 3. Fast AVX2 Activation Quantizer: float x[n_cols] -> block_q8_K_t y[(n_cols+255)/256]
 * ============================================================================ */
static inline void lfm2_quantize_block_q8_K_avx2(const float * restrict xb, block_q8_K_t * restrict yb) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    /* 1. Find max absolute value across 256 floats using 8x unrolled AVX2 max */
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
        yb->d = 0.0f;
        memset(yb->qs, 0, sizeof(yb->qs));
        memset(yb->bsums, 0, sizeof(yb->bsums));
        return;
    }

    const float d      = max_abs / 127.0f;
    const float iscale = 127.0f / max_abs;
    yb->d = d;

    const __m256 vscale = _mm256_set1_ps(iscale);

    /* 2. Quantize 16 floats at a time -> int8_t and compute 16-element bsums[j] */
    for (int j = 0; j < 16; ++j) {
        const float *x16 = xb + j * 16;

        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 0), vscale));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x16 + 8), vscale));

        yb->bsums[j] = (int16_t)lfm2_hsum_epi32_avx2(_mm256_add_epi32(i0, i1));

        __m256i p16 = _mm256_packs_epi32(i0, i1);
        p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i lo16 = _mm256_castsi256_si128(p16);
        __m128i hi16 = _mm256_extracti128_si256(p16, 1);
        __m128i p8   = _mm_packs_epi16(lo16, hi16);
        _mm_storeu_si128((__m128i *)(yb->qs + j * 16), p8);
    }
}

static inline void lfm2_quantize_row_q8_K_avx2(const float * restrict x, block_q8_K_t * restrict y, uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        lfm2_quantize_block_q8_K_avx2(x + (size_t)b * 256, &y[b]);
    }
    const uint32_t rem = n_cols % 256;
    if (rem > 0) {
        _Alignas(32) float pad[256];
        memset(pad, 0, sizeof(pad));
        memcpy(pad, x + (size_t)nb * 256, (size_t)rem * sizeof(float));
        lfm2_quantize_block_q8_K_avx2(pad, &y[nb]);
    }
}

/* ============================================================================
 * 4. AVX2 Q4_K x Q8_K Dot Product Kernel (_mm256_maddubs_epi16)
 * ============================================================================ */
static inline float lfm2_vec_dot_q4_K_q8_K_avx2(const block_q4_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        _mm_prefetch((const char *)(w_row + b + 2), _MM_HINT_T0);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d    = lfm2_fp16_to_fp32(w_row[b].d)    * d8;
        const float dmin = lfm2_fp16_to_fp32(w_row[b].dmin) * d8;

        const uint8_t * restrict q4 = w_row[b].qs;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc_i32   = _mm256_setzero_si256();
        int32_t sum_mins  = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            lfm2_get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            lfm2_get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);

            sum_mins += (int32_t)m1 * ((int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1])
                      + (int32_t)m2 * ((int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3]);

            const __m256i q4_bytes = _mm256_loadu_si256((const __m256i *)q4);
            const __m256i q4_lo    = _mm256_and_si256(q4_bytes, mask_0f);
            const __m256i q4_hi    = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), mask_0f);

            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));

            const __m256i dot_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
            const __m256i dot_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);

            acc_i32 = _mm256_add_epi32(acc_i32,
                _mm256_madd_epi16(dot_lo_i16, _mm256_set1_epi16((int16_t)sc1)));
            acc_i32 = _mm256_add_epi32(acc_i32,
                _mm256_madd_epi16(dot_hi_i16, _mm256_set1_epi16((int16_t)sc2)));

            q4 += 32;
            is += 2;
        }

        const int32_t sum_q4q8 = lfm2_hsum_epi32_avx2(acc_i32);
        sumf += d * (float)sum_q4q8 - dmin * (float)sum_mins;
    }

    return sumf;
}

/* 1x4 Row-Unrolled AVX2 Q4_K x Q8_K micro-kernel:
 * Reuses activation vector (q8_lo, q8_hi) in registers across 4 rows simultaneously,
 * using 4 independent accumulators to break latency dependency chains. */
static inline void lfm2_vec_dot_q4_K_q8_K_4rows_avx2(
    const block_q4_K_t * restrict w0,
    const block_q4_K_t * restrict w1,
    const block_q4_K_t * restrict w2,
    const block_q4_K_t * restrict w3,
    const block_q8_K_t * restrict x_q8,
    uint32_t n_cols,
    float * restrict out0,
    float * restrict out1,
    float * restrict out2,
    float * restrict out3
) {
    const uint32_t nb = (n_cols + 255) / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        _mm_prefetch((const char *)(w0 + b + 2), _MM_HINT_T0);
        _mm_prefetch((const char *)(w1 + b + 2), _MM_HINT_T0);
        _mm_prefetch((const char *)(w2 + b + 2), _MM_HINT_T0);
        _mm_prefetch((const char *)(w3 + b + 2), _MM_HINT_T0);

        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d0    = lfm2_fp16_to_fp32(w0[b].d)    * d8;
        const float dmin0 = lfm2_fp16_to_fp32(w0[b].dmin) * d8;
        const float d1    = lfm2_fp16_to_fp32(w1[b].d)    * d8;
        const float dmin1 = lfm2_fp16_to_fp32(w1[b].dmin) * d8;
        const float d2    = lfm2_fp16_to_fp32(w2[b].d)    * d8;
        const float dmin2 = lfm2_fp16_to_fp32(w2[b].dmin) * d8;
        const float d3    = lfm2_fp16_to_fp32(w3[b].d)    * d8;
        const float dmin3 = lfm2_fp16_to_fp32(w3[b].dmin) * d8;

        const uint8_t * restrict q4_0 = w0[b].qs;
        const uint8_t * restrict q4_1 = w1[b].qs;
        const uint8_t * restrict q4_2 = w2[b].qs;
        const uint8_t * restrict q4_3 = w3[b].qs;

        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        int32_t min0 = 0, min1 = 0, min2 = 0, min3 = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1_0, m1_0, sc2_0, m2_0;
            uint8_t sc1_1, m1_1, sc2_1, m2_1;
            uint8_t sc1_2, m1_2, sc2_2, m2_2;
            uint8_t sc1_3, m1_3, sc2_3, m2_3;

            lfm2_get_scale_min_k4(is + 0, w0[b].scales, &sc1_0, &m1_0);
            lfm2_get_scale_min_k4(is + 1, w0[b].scales, &sc2_0, &m2_0);
            lfm2_get_scale_min_k4(is + 0, w1[b].scales, &sc1_1, &m1_1);
            lfm2_get_scale_min_k4(is + 1, w1[b].scales, &sc2_1, &m2_1);
            lfm2_get_scale_min_k4(is + 0, w2[b].scales, &sc1_2, &m1_2);
            lfm2_get_scale_min_k4(is + 1, w2[b].scales, &sc2_2, &m2_2);
            lfm2_get_scale_min_k4(is + 0, w3[b].scales, &sc1_3, &m1_3);
            lfm2_get_scale_min_k4(is + 1, w3[b].scales, &sc2_3, &m2_3);

            const int32_t bsum_lo = (int32_t)bs[2 * is + 0] + (int32_t)bs[2 * is + 1];
            const int32_t bsum_hi = (int32_t)bs[2 * is + 2] + (int32_t)bs[2 * is + 3];

            min0 += (int32_t)m1_0 * bsum_lo + (int32_t)m2_0 * bsum_hi;
            min1 += (int32_t)m1_1 * bsum_lo + (int32_t)m2_1 * bsum_hi;
            min2 += (int32_t)m1_2 * bsum_lo + (int32_t)m2_2 * bsum_hi;
            min3 += (int32_t)m1_3 * bsum_lo + (int32_t)m2_3 * bsum_hi;

            /* Load activation registers ONCE for all 4 rows */
            const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)(q8 + j +  0));
            const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)(q8 + j + 32));

            /* Row 0 */
            const __m256i q4_bytes0 = _mm256_loadu_si256((const __m256i *)q4_0);
            const __m256i dot_lo0   = _mm256_maddubs_epi16(_mm256_and_si256(q4_bytes0, mask_0f), q8_lo);
            const __m256i dot_hi0   = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4_bytes0, 4), mask_0f), q8_hi);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(dot_lo0, _mm256_set1_epi16((int16_t)sc1_0)));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(dot_hi0, _mm256_set1_epi16((int16_t)sc2_0)));
            q4_0 += 32;

            /* Row 1 */
            const __m256i q4_bytes1 = _mm256_loadu_si256((const __m256i *)q4_1);
            const __m256i dot_lo1   = _mm256_maddubs_epi16(_mm256_and_si256(q4_bytes1, mask_0f), q8_lo);
            const __m256i dot_hi1   = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4_bytes1, 4), mask_0f), q8_hi);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(dot_lo1, _mm256_set1_epi16((int16_t)sc1_1)));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(dot_hi1, _mm256_set1_epi16((int16_t)sc2_1)));
            q4_1 += 32;

            /* Row 2 */
            const __m256i q4_bytes2 = _mm256_loadu_si256((const __m256i *)q4_2);
            const __m256i dot_lo2   = _mm256_maddubs_epi16(_mm256_and_si256(q4_bytes2, mask_0f), q8_lo);
            const __m256i dot_hi2   = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4_bytes2, 4), mask_0f), q8_hi);
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(dot_lo2, _mm256_set1_epi16((int16_t)sc1_2)));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(dot_hi2, _mm256_set1_epi16((int16_t)sc2_2)));
            q4_2 += 32;

            /* Row 3 */
            const __m256i q4_bytes3 = _mm256_loadu_si256((const __m256i *)q4_3);
            const __m256i dot_lo3   = _mm256_maddubs_epi16(_mm256_and_si256(q4_bytes3, mask_0f), q8_lo);
            const __m256i dot_hi3   = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4_bytes3, 4), mask_0f), q8_hi);
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(dot_lo3, _mm256_set1_epi16((int16_t)sc1_3)));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(dot_hi3, _mm256_set1_epi16((int16_t)sc2_3)));
            q4_3 += 32;

            is += 2;
        }

        sum0 += d0 * (float)lfm2_hsum_epi32_avx2(acc0) - dmin0 * (float)min0;
        sum1 += d1 * (float)lfm2_hsum_epi32_avx2(acc1) - dmin1 * (float)min1;
        sum2 += d2 * (float)lfm2_hsum_epi32_avx2(acc2) - dmin2 * (float)min2;
        sum3 += d3 * (float)lfm2_hsum_epi32_avx2(acc3) - dmin3 * (float)min3;
    }

    *out0 = sum0;
    *out1 = sum1;
    *out2 = sum2;
    *out3 = sum3;
}

/* ============================================================================
 * 5. AVX2 Q6_K x Q8_K Dot Product Kernel (_mm256_maddubs_epi16)
 * ============================================================================ */
static inline float lfm2_vec_dot_q6_K_q8_K_avx2(const block_q6_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    const __m256i mask_0c = _mm256_set1_epi8(0x0C);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);

    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb; ++b) {
        _mm_prefetch((const char *)(w_row + b + 2), _MM_HINT_T0);
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const float d = lfm2_fp16_to_fp32(w_row[b].d) * d8;

        const uint8_t * restrict ql = w_row[b].ql;
        const uint8_t * restrict qh = w_row[b].qh;
        const int8_t  * restrict sc = w_row[b].scales;
        const int8_t  * restrict q8 = x_q8[b].qs;
        const int16_t * restrict bs = x_q8[b].bsums;

        int32_t sum_offset = 0;
        for (int is = 0; is < 16; ++is) {
            sum_offset += (int32_t)sc[is] * (int32_t)bs[is];
        }

        __m256i acc_i32 = _mm256_setzero_si256();

        for (int half = 0; half < 2; ++half) {
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

            const __m256i x0 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
            const __m256i x1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
            const __m256i x2 = _mm256_loadu_si256((const __m256i *)(q8 + 64));
            const __m256i x3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));

            const __m256i dot0 = _mm256_maddubs_epi16(u1, x0);
            const __m256i dot1 = _mm256_maddubs_epi16(u2, x1);
            const __m256i dot2 = _mm256_maddubs_epi16(u3, x2);
            const __m256i dot3 = _mm256_maddubs_epi16(u4, x3);

            const __m256i sc01 = lfm2_mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            const __m256i sc23 = lfm2_mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            const __m256i sc45 = lfm2_mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            const __m256i sc67 = lfm2_mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));

            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot0, sc01));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot1, sc23));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot2, sc45));
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(dot3, sc67));

            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }

        const int32_t sum_u6q8 = lfm2_hsum_epi32_avx2(acc_i32);
        sumf += d * (float)(sum_u6q8 - 32 * sum_offset);
    }

    return sumf;
}

/* ============================================================================
 * 6. AVX2 Q8_0 x Q8_K and F32 Dot Product Kernels
 * ============================================================================ */
static inline float lfm2_vec_dot_q8_0_q8_K_avx2(const block_q8_0_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb_k = (n_cols + 255) / 256;
    const __m256i ones_16 = _mm256_set1_epi16(1);
    float sumf = 0.0f;

    for (uint32_t b = 0; b < nb_k; ++b) {
        const float d8 = x_q8[b].d;
        if (d8 == 0.0f) continue;

        const block_q8_0_t *w_sub = w_row + (size_t)b * 8;
        const int8_t       *q8    = x_q8[b].qs;
        float block_sum = 0.0f;

        uint32_t n_sub = 8;
        if ((size_t)(b + 1) * 256 > n_cols) {
            uint32_t rem_cols = n_cols - b * 256;
            n_sub = (rem_cols + 31) / 32;
        }

        for (uint32_t j = 0; j < n_sub; ++j) {
            const float dw = lfm2_fp16_to_fp32(w_sub[j].d);
            const __m256i vw = _mm256_loadu_si256((const __m256i *)w_sub[j].qs);
            const __m256i vx = _mm256_loadu_si256((const __m256i *)(q8 + j * 32));

            const __m256i ax  = _mm256_sign_epi8(vw, vw);
            const __m256i sy  = _mm256_sign_epi8(vx, vw);
            const __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), ones_16);

            block_sum += dw * (float)lfm2_hsum_epi32_avx2(dot);
        }
        sumf += d8 * block_sum;
    }
    return sumf;
}

static inline float lfm2_vec_dot_f32_avx2(const float * restrict w_row, const float * restrict x, uint32_t n_cols) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n_cols; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + i + 0), _mm256_loadu_ps(x + i + 0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + i + 8), _mm256_loadu_ps(x + i + 8), acc1);
    }
    float sum = lfm2_hsum_ps_avx2(_mm256_add_ps(acc0, acc1));
    for (; i < n_cols; ++i) sum += w_row[i] * x[i];
    return sum;
}

/* ============================================================================
 * 7. Unified Matrix-Vector Multiplier (Pre-Quantizes x to Q8_K Once)
 * ============================================================================ */
static inline void lfm2_matmul_vec_avx2(float * restrict out,
                                        const void * restrict w_data,
                                        uint32_t w_type,
                                        uint32_t n_cols,
                                        uint32_t n_rows,
                                        const float * restrict x,
                                        block_q8_K_t * restrict xq_scratch) {
    if (w_type == 12 /* GGML_TYPE_Q4_K */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 255) / 256;
        const block_q4_K_t *W = (const block_q4_K_t *)w_data;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q4_K_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 14 /* GGML_TYPE_Q6_K */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 255) / 256;
        const block_q6_K_t *W = (const block_q6_K_t *)w_data;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q6_K_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 8 /* GGML_TYPE_Q8_0 */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 31) / 32;
        const block_q8_0_t *W = (const block_q8_0_t *)w_data;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q8_0_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 0 /* GGML_TYPE_F32 */) {
        const float *W = (const float *)w_data;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_f32_avx2(W + (size_t)r * n_cols, x, n_cols);
        }
    }
}

#else /* !LFM2_HAS_AVX2 (Portable Fallback) */

static inline float lfm2_vec_dot_q4_K_fallback(const block_q4_K_t *w_row, const float *x, uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d   = lfm2_fp16_to_fp32(w_row[b].d);
        const float min = lfm2_fp16_to_fp32(w_row[b].dmin);
        const uint8_t *q = w_row[b].qs;
        const float *xb  = x + (size_t)b * 256;
        const uint32_t rem = (b * 256 < n_cols) ? (n_cols - b * 256) : 0;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            lfm2_get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            lfm2_get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            float s1 = 0.0f, s1_x = 0.0f, s2 = 0.0f, s2_x = 0.0f;
            for (int l = 0; l < 32; ++l) {
                float v1 = (j + l < (int)rem) ? xb[j + l] : 0.0f;
                float v2 = (j + 32 + l < (int)rem) ? xb[j + 32 + l] : 0.0f;
                s1 += (float)(q[l] & 0x0F) * v1;      s1_x += v1;
                s2 += (float)(q[l] >> 4)   * v2;      s2_x += v2;
            }
            sumf += (d * sc1 * s1 - min * m1 * s1_x) + (d * sc2 * s2 - min * m2 * s2_x);
            q += 32;
            is += 2;
        }
    }
    return sumf;
}

static inline float lfm2_vec_dot_q6_K_fallback(const block_q6_K_t *w_row, const float *x, uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d = lfm2_fp16_to_fp32(w_row[b].d);
        const uint8_t *ql = w_row[b].ql;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *sc = w_row[b].scales;
        const float   *xb = x + (size_t)b * 256;
        const uint32_t rem = (b * 256 < n_cols) ? (n_cols - b * 256) : 0;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                float v1 = (n + l +  0 < (int)rem) ? xb[n + l +  0] : 0.0f;
                float v2 = (n + l + 32 < (int)rem) ? xb[n + l + 32] : 0.0f;
                float v3 = (n + l + 64 < (int)rem) ? xb[n + l + 64] : 0.0f;
                float v4 = (n + l + 96 < (int)rem) ? xb[n + l + 96] : 0.0f;
                sumf += d * (sc[is + 0] * q1 * v1 + sc[is + 2] * q2 * v2 +
                             sc[is + 4] * q3 * v3 + sc[is + 6] * q4 * v4);
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    return sumf;
}

static inline float lfm2_vec_dot_f32_avx2(const float *a, const float *b, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline void lfm2_quantize_row_q8_K_avx2(const float * restrict x,
                                               block_q8_K_t * restrict y,
                                               uint32_t n_cols) {
    lfm2_quantize_row_q8_K_fallback(x, y, n_cols);
}

static inline float lfm2_vec_dot_q4_K_q8_K_avx2(const block_q4_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d    = lfm2_fp16_to_fp32(w_row[b].d)    * x_q8[b].d;
        const float dmin = lfm2_fp16_to_fp32(w_row[b].dmin) * x_q8[b].d;
        const uint8_t *q4 = w_row[b].qs;
        const int8_t  *q8 = x_q8[b].qs;
        const int16_t *bs = x_q8[b].bsums;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            lfm2_get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            lfm2_get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            int32_t dot1 = 0, dot2 = 0;
            for (int l = 0; l < 32; ++l) {
                dot1 += (int32_t)(q4[l] & 0x0F) * (int32_t)q8[j + l];
                dot2 += (int32_t)(q4[l] >> 4)   * (int32_t)q8[j + 32 + l];
            }
            sumf += d * (float)(sc1 * dot1 + sc2 * dot2) -
                    dmin * (float)(m1 * (bs[j/16] + bs[j/16 + 1]) + m2 * (bs[j/16 + 2] + bs[j/16 + 3]));
            q4 += 32;
            is += 2;
        }
    }
    return sumf;
}

static inline float lfm2_vec_dot_q6_K_q8_K_avx2(const block_q6_K_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb = (n_cols + 255) / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d = lfm2_fp16_to_fp32(w_row[b].d) * x_q8[b].d;
        const uint8_t *ql = w_row[b].ql;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *sc = w_row[b].scales;
        const int8_t  *q8 = x_q8[b].qs;
        for (int n = 0; n < 256; n += 128) {
            int32_t dot_sub[8] = {0};
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                dot_sub[is + 0] += (int32_t)q1 * (int32_t)q8[n + l +  0];
                dot_sub[is + 2] += (int32_t)q2 * (int32_t)q8[n + l + 32];
                dot_sub[is + 4] += (int32_t)q3 * (int32_t)q8[n + l + 64];
                dot_sub[is + 6] += (int32_t)q4 * (int32_t)q8[n + l + 96];
            }
            float block_sum = 0.0f;
            for (int s = 0; s < 8; ++s) {
                block_sum += (float)(sc[s] * dot_sub[s]);
            }
            sumf += d * block_sum;
            ql += 64; qh += 32; sc += 8;
        }
    }
    return sumf;
}

static inline float lfm2_vec_dot_q8_0_q8_K_avx2(const block_q8_0_t * restrict w_row,
                                                const block_q8_K_t * restrict x_q8,
                                                uint32_t n_cols) {
    const uint32_t nb = (n_cols + 31) / 32;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d8 = x_q8[b / 8].d;
        const float d  = lfm2_fp16_to_fp32(w_row[b].d) * d8;
        const int8_t *q8_w = w_row[b].qs;
        const int8_t *q8_x = x_q8[b / 8].qs + (b % 8) * 32;
        int32_t idot = 0;
        for (int i = 0; i < 32; ++i) {
            idot += (int32_t)q8_w[i] * (int32_t)q8_x[i];
        }
        sumf += d * (float)idot;
    }
    return sumf;
}

static inline void lfm2_matmul_vec_avx2(float * restrict out,
                                        const void * restrict w_data,
                                        uint32_t w_type,
                                        uint32_t n_cols,
                                        uint32_t n_rows,
                                        const float * restrict x,
                                        block_q8_K_t * restrict xq_scratch) {
    if (w_type == 12 /* GGML_TYPE_Q4_K */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 255) / 256;
        const block_q4_K_t *W = (const block_q4_K_t *)w_data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q4_K_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 14 /* GGML_TYPE_Q6_K */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 255) / 256;
        const block_q6_K_t *W = (const block_q6_K_t *)w_data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q6_K_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 8 /* GGML_TYPE_Q8_0 */) {
        lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        const size_t row_blocks = (n_cols + 31) / 32;
        const block_q8_0_t *W = (const block_q8_0_t *)w_data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_q8_0_q8_K_avx2(W + (size_t)r * row_blocks, xq_scratch, n_cols);
        }
    } else if (w_type == 0 /* GGML_TYPE_F32 */) {
        const float *W = (const float *)w_data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            out[r] = lfm2_vec_dot_f32_avx2(W + (size_t)r * n_cols, x, n_cols);
        }
    }
}

#endif /* LFM2_HAS_AVX2 */

#endif /* LFM2_SIMD_AVX2_H */
