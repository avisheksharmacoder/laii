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

#if defined(__aarch64__)
/* ARM64 NEON activation quantizer prototype */
void lfm2_quantize_row_q8_K_neon(const float * restrict x, block_q8_K_t * restrict y, uint32_t n_cols);
#endif

#ifndef fp32_to_fp16_scalar
#define fp32_to_fp16_scalar fp32_to_fp16_scalar
static inline uint16_t fp32_to_fp16_scalar(float f) {
#if defined(__F16C__)
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x00001000) >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    return (uint16_t)(sign | (exp << 10) | ((mant + 0x00001000) >> 13));
#endif
}
#define fp32_to_fp16 fp32_to_fp16_scalar
#endif

#ifndef LFM2_TARGET_AVX2
#if defined(__x86_64__) || defined(_M_X64)
#define LFM2_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define LFM2_TARGET_AVX2
#endif
#endif

#ifndef LFM2_TARGET_ARM_FP16
#if defined(__aarch64__)
#define LFM2_TARGET_ARM_FP16 __attribute__((target("arch=armv8.2-a+dotprod+fp16")))
#else
#define LFM2_TARGET_ARM_FP16
#endif
#endif

#ifndef LFM2_KERNELS_FAST_INCLUDED
#define LFM2_KERNELS_FAST_INCLUDED 1

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_DECODE_1X4_NO_MAIN 1

#if __has_include("tools/lfm2_decode_1x4.c")
#include "tools/lfm2_decode_1x4.c"
#elif __has_include("lfm2_decode_1x4.c")
#include "lfm2_decode_1x4.c"
#else
#include "../tools/lfm2_decode_1x4.c"
#endif

#ifndef lfm2_aligned_alloc_64
#define lfm2_aligned_alloc_64 lfm2_aligned_alloc_64
static inline void *lfm2_aligned_alloc_64(size_t size) {
    size_t aligned_size = (size + 63) & ~((size_t)63);
    return aligned_alloc(64, aligned_size);
}
#endif

/* ============================================================================
 * 1. Planar ShortConv 3-Tap Weight Cache & Fused AVX2 ShortConv Kernel
 * ============================================================================ */
typedef struct {
    float *w0; /* [D] 64-byte aligned tap 0 weights (multiplies state[t-2]) */
    float *w1; /* [D] 64-byte aligned tap 1 weights (multiplies state[t-1]) */
    float *w2; /* [D] 64-byte aligned tap 2 weights (multiplies u[t])       */
} LFM2PlanarConvWeights;

/* Transposes GGUF shortconv.conv.weight [3, D] into planar w0[D], w1[D], w2[D]
 * once at startup so runtime ShortConv executes 8-wide contiguous AVX2 loads. */
static inline bool lfm2_planar_conv_init(LFM2PlanarConvWeights *pw, const GGUFTensor *conv_t, uint32_t D) {
    if (!pw || !conv_t || !conv_t->data) return false;
    pw->w0 = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    pw->w1 = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    pw->w2 = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    if (!pw->w0 || !pw->w1 || !pw->w2) return false;

    const bool stride_is_3 = (conv_t->ne[0] == 3);

    if (conv_t->type == GGML_TYPE_F32) {
        const float *src = (const float *)conv_t->data;
        for (uint32_t d = 0; d < D; ++d) {
            pw->w0[d] = stride_is_3 ? src[d * 3 + 0] : src[0 * D + d];
            pw->w1[d] = stride_is_3 ? src[d * 3 + 1] : src[1 * D + d];
            pw->w2[d] = stride_is_3 ? src[d * 3 + 2] : src[2 * D + d];
        }
    } else {
        const uint16_t *src = (const uint16_t *)conv_t->data;
        for (uint32_t d = 0; d < D; ++d) {
            pw->w0[d] = stride_is_3 ? fp16_to_fp32(src[d * 3 + 0]) : fp16_to_fp32(src[0 * D + d]);
            pw->w1[d] = stride_is_3 ? fp16_to_fp32(src[d * 3 + 1]) : fp16_to_fp32(src[1 * D + d]);
            pw->w2[d] = stride_is_3 ? fp16_to_fp32(src[d * 3 + 2]) : fp16_to_fp32(src[2 * D + d]);
        }
    }
    return true;
}

static inline void lfm2_planar_conv_free(LFM2PlanarConvWeights *pw) {
    if (!pw) return;
    free(pw->w0);
    free(pw->w1);
    free(pw->w2);
    memset(pw, 0, sizeof(*pw));
}

#if LFM2_HAS_AVX2
/* Fused AVX2 ShortConv Operator:
 *   1. Gate 1: u_t = B * x_tilde
 *   2. 3-Tap Depthwise Conv1D: acc = w0 * state[t-2] + w1 * state[t-1] + w2 * u_t
 *   3. Rolling State Shift: state[t-2] <- state[t-1], state[t-1] <- u_t
 *   4. Gate 2: conv_out = C * acc
 *   5. Immediate L1-hot Q8_K quantization of `conv_out` into `conv_out_q8`! */
static inline void lfm2_shortconv_gate_conv_quant_avx2(float * restrict conv_out,
                                                       block_q8_K_t * restrict conv_out_q8,
                                                       const float * restrict bcx,
                                                       float * restrict conv_state,
                                                       const LFM2PlanarConvWeights * restrict pw,
                                                       uint32_t D) {
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2 * D;

    float * restrict state_tm2 = conv_state;     /* [D] token t - 2 */
    float * restrict state_tm1 = conv_state + D; /* [D] token t - 1 */

    const float * restrict w0 = pw->w0;
    const float * restrict w1 = pw->w1;
    const float * restrict w2 = pw->w2;

    const uint32_t nb = D / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256;

        for (uint32_t i = 0; i < 256; i += 8) {
            const uint32_t d = base + i;

            const __m256 vB  = _mm256_loadu_ps(B + d);
            const __m256 vC  = _mm256_loadu_ps(C + d);
            const __m256 vx  = _mm256_loadu_ps(x_tilde + d);

            /* 1. First Multiplicative Gate: u_t = B * x_tilde */
            const __m256 u_t = _mm256_mul_ps(vB, vx);

            /* 2. Load 2-step rolling state and update in-place */
            const __m256 s0  = _mm256_loadu_ps(state_tm2 + d);
            const __m256 s1  = _mm256_loadu_ps(state_tm1 + d);
            _mm256_storeu_ps(state_tm2 + d, s1);
            _mm256_storeu_ps(state_tm1 + d, u_t);

            /* 3. 3-Tap Depthwise Convolution via FMA */
            __m256 acc = _mm256_mul_ps(_mm256_load_ps(w0 + d), s0);
            acc = _mm256_fmadd_ps(_mm256_load_ps(w1 + d), s1, acc);
            acc = _mm256_fmadd_ps(_mm256_load_ps(w2 + d), u_t, acc);

            /* 4. Second Multiplicative Gate: conv_out = C * acc */
            _mm256_storeu_ps(conv_out + d, _mm256_mul_ps(vC, acc));
        }

        /* 5. Quantize this L1-hot 256-float block directly to Q8_K for out_proj */
        if (conv_out_q8 != NULL) {
            lfm2_quantize_row_q8_K_avx2(conv_out + base, &conv_out_q8[b], 256);
        }
    }
}
#endif

/* Portable C11 Scalar Fallback */
static inline void lfm2_shortconv_gate_conv_quant_scalar(float * restrict conv_out,
                                                         block_q8_K_t * restrict conv_out_q8,
                                                         const float * restrict bcx,
                                                         float * restrict conv_state,
                                                         const LFM2PlanarConvWeights * restrict pw,
                                                         uint32_t D) {
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2 * D;

    float * restrict state_tm2 = conv_state;
    float * restrict state_tm1 = conv_state + D;

    const float * restrict w0 = pw->w0;
    const float * restrict w1 = pw->w1;
    const float * restrict w2 = pw->w2;

    const uint32_t nb = D / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256;
        for (uint32_t i = 0; i < 256; ++i) {
            const uint32_t d = base + i;
            float u_t = B[d] * x_tilde[d];
            float s0 = state_tm2[d];
            float s1 = state_tm1[d];
            state_tm2[d] = s1;
            state_tm1[d] = u_t;
            float acc = w0[d] * s0 + w1[d] * s1 + w2[d] * u_t;
            conv_out[d] = C[d] * acc;
        }
        if (conv_out_q8 != NULL) {
            lfm2_quantize_row_q8_K_scalar(conv_out + base, &conv_out_q8[b], 256);
        }
    }
}

#if defined(__aarch64__)
/* ARM64 NEON Planar ShortConv 3-Tap + Gate + Optional In-Place Q8_K Quantizer */
LFM2_TARGET_ARM_FP16
static inline void lfm2_shortconv_gate_conv_quant_neon(float * restrict conv_out,
                                                       block_q8_K_t * restrict conv_out_q8,
                                                       const float * restrict bcx,
                                                       float * restrict conv_state,
                                                       const LFM2PlanarConvWeights * restrict pw,
                                                       uint32_t D) {
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2 * D;

    float * restrict state_tm2 = conv_state;     /* [D] token t - 2 */
    float * restrict state_tm1 = conv_state + D; /* [D] token t - 1 */

    const float * restrict w0 = pw->w0;
    const float * restrict w1 = pw->w1;
    const float * restrict w2 = pw->w2;

    const uint32_t nb = D / 256;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256;

        /* 2x4 unrolled (8 channels per iteration) to hide 4-cycle FMA pipeline latency */
        for (uint32_t i = 0; i < 256; i += 8) {
            const uint32_t d0 = base + i;
            const uint32_t d1 = d0 + 4;

            const float32x4_t vB0 = vld1q_f32(B + d0),       vB1 = vld1q_f32(B + d1);
            const float32x4_t vC0 = vld1q_f32(C + d0),       vC1 = vld1q_f32(C + d1);
            const float32x4_t vx0 = vld1q_f32(x_tilde + d0), vx1 = vld1q_f32(x_tilde + d1);

            /* 1. First Multiplicative Gate: u_t = B * x_tilde */
            const float32x4_t u0 = vmulq_f32(vB0, vx0);
            const float32x4_t u1 = vmulq_f32(vB1, vx1);

            /* 2. Load 2-step rolling state and shift in-place: tm2 <- tm1, tm1 <- u_t */
            const float32x4_t s0_0 = vld1q_f32(state_tm2 + d0), s0_1 = vld1q_f32(state_tm2 + d1);
            const float32x4_t s1_0 = vld1q_f32(state_tm1 + d0), s1_1 = vld1q_f32(state_tm1 + d1);
            vst1q_f32(state_tm2 + d0, s1_0);
            vst1q_f32(state_tm2 + d1, s1_1);
            vst1q_f32(state_tm1 + d0, u0);
            vst1q_f32(state_tm1 + d1, u1);

            /* 3. 3-Tap Depthwise Convolution via NEON FMA: w0*s0 + w1*s1 + w2*u_t */
            float32x4_t acc0 = vmulq_f32(vld1q_f32(w0 + d0), s0_0);
            float32x4_t acc1 = vmulq_f32(vld1q_f32(w0 + d1), s0_1);
            acc0 = vfmaq_f32(acc0, vld1q_f32(w1 + d0), s1_0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(w1 + d1), s1_1);
            acc0 = vfmaq_f32(acc0, vld1q_f32(w2 + d0), u0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(w2 + d1), u1);

            /* 4. Second Multiplicative Gate: conv_out = C * acc */
            vst1q_f32(conv_out + d0, vmulq_f32(vC0, acc0));
            vst1q_f32(conv_out + d1, vmulq_f32(vC1, acc1));
        }

        /* 5. Quantize L1-hot 256-float block directly to Q8_K for out_proj if requested */
        if (conv_out_q8 != NULL) {
            lfm2_quantize_row_q8_K_neon(conv_out + base, &conv_out_q8[b], 256);
        }
    }
}
#endif

/* Cross-Platform Vectorized In-Place Addition */
static inline void lfm2_vec_add_inplace(float * restrict dst, const float * restrict src, uint32_t n) {
#if LFM2_HAS_AVX2
    for (uint32_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
#elif defined(__aarch64__)
    for (uint32_t i = 0; i < n; i += 8) {
        vst1q_f32(dst + i + 0, vaddq_f32(vld1q_f32(dst + i + 0), vld1q_f32(src + i + 0)));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4)));
    }
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] += src[i];
#endif
}

#if LFM2_HAS_AVX2
    #define lfm2_shortconv_gate_conv_quant_simd lfm2_shortconv_gate_conv_quant_avx2
#elif defined(__aarch64__)
    #define lfm2_shortconv_gate_conv_quant_simd lfm2_shortconv_gate_conv_quant_neon
#else
    #define lfm2_shortconv_gate_conv_quant_simd lfm2_shortconv_gate_conv_quant_scalar
#endif

/* Complete Fused Single-Token ShortConv Layer:
 * Executes in only TWO thread-pool barrier dispatches with L1-hot Q8_K quantization. */
static inline void lfm2_forward_shortconv_fast(LFM2ThreadPool *pool,
                                               float * restrict x,
                                               float * restrict xb,
                                               float * restrict bcx,
                                               float * restrict conv_out,
                                               float * restrict conv_state,
                                               block_q8_K_t * restrict xq_scratch,
                                               const LFM2Layer * restrict layer,
                                               const LFM2PlanarConvWeights * restrict pw,
                                               uint32_t D,
                                               float eps) {
    /* 1. Fused RMSNorm(x -> xb) + Quantize(xb -> xq_scratch) */
    lfm2_rmsnorm_quantize_q8_K_simd(xb, xq_scratch, x, layer->attn_norm, D, eps);

    /* 2. 1x4 Unrolled Input Projection: xb [D] -> bcx [3 * D] */
    lfm2_matmul_1x4_prequant_mt(pool, bcx, layer->shortconv.in_proj, xb, xq_scratch);

    /* 3. Fused SIMD Gate1 + 3-Tap Conv1D + State Shift + Gate2 + Q8_K Quantize */
    lfm2_shortconv_gate_conv_quant_simd(conv_out, xq_scratch, bcx, conv_state, pw, D);

    /* 4. 1x4 Unrolled Output Projection: conv_out [D] -> xb [D] */
    lfm2_matmul_1x4_prequant_mt(pool, xb, layer->shortconv.out_proj, conv_out, xq_scratch);

    /* 5. Vectorized Residual Add */
    lfm2_vec_add_inplace(x, xb, D);
}

/* ============================================================================
 * 2. Precomputed RoPE Lookup Table & Fused AVX2 Per-Head QK-Norm + F16C Store
 * ============================================================================ */
typedef struct {
    uint32_t max_seq_len;
    uint32_t half_dim; /* 32 for LFM2 (head_dim = 64) */
    float   *cos_lut;  /* [max_seq_len, 32] 64-byte aligned */
    float   *sin_lut;  /* [max_seq_len, 32] 64-byte aligned */
} LFM2RoPELUT;

static inline bool lfm2_rope_lut_init(LFM2RoPELUT *lut, uint32_t max_seq_len, uint32_t head_dim, float freq_base) {
    const uint32_t half_dim = head_dim / 2;
    lut->max_seq_len = max_seq_len;
    lut->half_dim    = half_dim;

    lut->cos_lut = (float *)lfm2_aligned_alloc_64((size_t)max_seq_len * half_dim * sizeof(float));
    lut->sin_lut = (float *)lfm2_aligned_alloc_64((size_t)max_seq_len * half_dim * sizeof(float));
    if (!lut->cos_lut || !lut->sin_lut) return false;

    for (uint32_t i = 0; i < half_dim; ++i) {
        const double freq = pow((double)freq_base, -((double)(2 * i) / (double)head_dim));
        for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
            const double angle = (double)pos * freq;
            lut->cos_lut[(size_t)pos * half_dim + i] = (float)cos(angle);
            lut->sin_lut[(size_t)pos * half_dim + i] = (float)sin(angle);
        }
    }
    return true;
}

static inline void lfm2_rope_lut_free(LFM2RoPELUT *lut) {
    if (!lut) return;
    free(lut->cos_lut);
    free(lut->sin_lut);
    memset(lut, 0, sizeof(*lut));
}

#if LFM2_HAS_AVX2
/* Vectorized Per-Head QK-Norm + NeoX Half-Split RoPE (head_dim = 64):
 * - Normalizes each 64-dim head in AVX2 registers
 * - Rotates (head[0..31], head[32..63]) using precomputed `cos_lut[pos]` / `sin_lut[pos]`
 * - If `dst_f16 != NULL`, converts rotated head directly to FP16 via F16C
 *   (`_mm256_cvtps_ph`) and writes to the FP16 KV-cache! */
static inline void lfm2_qk_norm_rope_avx2(float * restrict vec_f32,
                                          uint16_t * restrict dst_f16,
                                          uint32_t n_heads,
                                          uint32_t head_dim,
                                          const GGUFTensor * restrict norm_w,
                                          float eps,
                                          const LFM2RoPELUT * restrict lut,
                                          uint32_t pos) {
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * 32;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * 32;

    /* Load the 32 cos and 32 sin values for `pos` into 8 YMM registers */
    const __m256 c0 = _mm256_load_ps(cos_p +  0), s0 = _mm256_load_ps(sin_p +  0);
    const __m256 c1 = _mm256_load_ps(cos_p +  8), s1 = _mm256_load_ps(sin_p +  8);
    const __m256 c2 = _mm256_load_ps(cos_p + 16), s2 = _mm256_load_ps(sin_p + 16);
    const __m256 c3 = _mm256_load_ps(cos_p + 24), s3 = _mm256_load_ps(sin_p + 24);

    /* Load the 64-float per-head RMSNorm weight vector once */
    const float * restrict nw = (norm_w != NULL) ? (const float *)norm_w->data : NULL;
    const __m256 nw_lo0 = nw ? _mm256_loadu_ps(nw +  0) : _mm256_set1_ps(1.0f);
    const __m256 nw_lo1 = nw ? _mm256_loadu_ps(nw +  8) : _mm256_set1_ps(1.0f);
    const __m256 nw_lo2 = nw ? _mm256_loadu_ps(nw + 16) : _mm256_set1_ps(1.0f);
    const __m256 nw_lo3 = nw ? _mm256_loadu_ps(nw + 24) : _mm256_set1_ps(1.0f);
    const __m256 nw_hi0 = nw ? _mm256_loadu_ps(nw + 32) : _mm256_set1_ps(1.0f);
    const __m256 nw_hi1 = nw ? _mm256_loadu_ps(nw + 40) : _mm256_set1_ps(1.0f);
    const __m256 nw_hi2 = nw ? _mm256_loadu_ps(nw + 48) : _mm256_set1_ps(1.0f);
    const __m256 nw_hi3 = nw ? _mm256_loadu_ps(nw + 56) : _mm256_set1_ps(1.0f);

    (void)head_dim; /* LFM2 head_dim is 64 */

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp = vec_f32 + (size_t)h * 64;

        __m256 x_lo0 = _mm256_loadu_ps(hp +  0), x_hi0 = _mm256_loadu_ps(hp + 32);
        __m256 x_lo1 = _mm256_loadu_ps(hp +  8), x_hi1 = _mm256_loadu_ps(hp + 40);
        __m256 x_lo2 = _mm256_loadu_ps(hp + 16), x_hi2 = _mm256_loadu_ps(hp + 48);
        __m256 x_lo3 = _mm256_loadu_ps(hp + 24), x_hi3 = _mm256_loadu_ps(hp + 56);

        /* 1. Per-Head RMSNorm across all 64 lanes in AVX2 registers */
        if (nw != NULL) {
            __m256 ss0 = _mm256_add_ps(_mm256_mul_ps(x_lo0, x_lo0), _mm256_mul_ps(x_hi0, x_hi0));
            __m256 ss1 = _mm256_add_ps(_mm256_mul_ps(x_lo1, x_lo1), _mm256_mul_ps(x_hi1, x_hi1));
            ss0 = _mm256_fmadd_ps(x_lo2, x_lo2, _mm256_fmadd_ps(x_hi2, x_hi2, ss0));
            ss1 = _mm256_fmadd_ps(x_lo3, x_lo3, _mm256_fmadd_ps(x_hi3, x_hi3, ss1));

            const float ss = hsum_ps_avx2(_mm256_add_ps(ss0, ss1));
            const __m256 virms = _mm256_set1_ps(1.0f / sqrtf(ss * (1.0f / 64.0f) + eps));

            x_lo0 = _mm256_mul_ps(_mm256_mul_ps(x_lo0, virms), nw_lo0);
            x_lo1 = _mm256_mul_ps(_mm256_mul_ps(x_lo1, virms), nw_lo1);
            x_lo2 = _mm256_mul_ps(_mm256_mul_ps(x_lo2, virms), nw_lo2);
            x_lo3 = _mm256_mul_ps(_mm256_mul_ps(x_lo3, virms), nw_lo3);

            x_hi0 = _mm256_mul_ps(_mm256_mul_ps(x_hi0, virms), nw_hi0);
            x_hi1 = _mm256_mul_ps(_mm256_mul_ps(x_hi1, virms), nw_hi1);
            x_hi2 = _mm256_mul_ps(_mm256_mul_ps(x_hi2, virms), nw_hi2);
            x_hi3 = _mm256_mul_ps(_mm256_mul_ps(x_hi3, virms), nw_hi3);
        }

        /* 2. NeoX Half-Split RoPE:
         *    r_lo = x_lo * cos - x_hi * sin
         *    r_hi = x_hi * cos + x_lo * sin */
        const __m256 r_lo0 = _mm256_fmsub_ps(x_lo0, c0, _mm256_mul_ps(x_hi0, s0));
        const __m256 r_hi0 = _mm256_fmadd_ps(x_hi0, c0, _mm256_mul_ps(x_lo0, s0));
        const __m256 r_lo1 = _mm256_fmsub_ps(x_lo1, c1, _mm256_mul_ps(x_hi1, s1));
        const __m256 r_hi1 = _mm256_fmadd_ps(x_hi1, c1, _mm256_mul_ps(x_lo1, s1));
        const __m256 r_lo2 = _mm256_fmsub_ps(x_lo2, c2, _mm256_mul_ps(x_hi2, s2));
        const __m256 r_hi2 = _mm256_fmadd_ps(x_hi2, c2, _mm256_mul_ps(x_lo2, s2));
        const __m256 r_lo3 = _mm256_fmsub_ps(x_lo3, c3, _mm256_mul_ps(x_hi3, s3));
        const __m256 r_hi3 = _mm256_fmadd_ps(x_hi3, c3, _mm256_mul_ps(x_lo3, s3));

        if (dst_f16 != NULL) {
            /* Convert rotated K head directly to FP16 via hardware F16C */
            uint16_t * restrict kp = dst_f16 + (size_t)h * 64;
            _mm_storeu_si128((__m128i *)(kp +  0), _mm256_cvtps_ph(r_lo0, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp +  8), _mm256_cvtps_ph(r_lo1, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 16), _mm256_cvtps_ph(r_lo2, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 24), _mm256_cvtps_ph(r_lo3, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 32), _mm256_cvtps_ph(r_hi0, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 40), _mm256_cvtps_ph(r_hi1, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 48), _mm256_cvtps_ph(r_hi2, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(kp + 56), _mm256_cvtps_ph(r_hi3, _MM_FROUND_TO_NEAREST_INT));
        } else {
            /* Store rotated Q head in FP32 */
            _mm256_storeu_ps(hp +  0, r_lo0);
            _mm256_storeu_ps(hp +  8, r_lo1);
            _mm256_storeu_ps(hp + 16, r_lo2);
            _mm256_storeu_ps(hp + 24, r_lo3);
            _mm256_storeu_ps(hp + 32, r_hi0);
            _mm256_storeu_ps(hp + 40, r_hi1);
            _mm256_storeu_ps(hp + 48, r_hi2);
            _mm256_storeu_ps(hp + 56, r_hi3);
        }
    }
}

/* Converts a float vector `v_f32[kv_dim]` directly into `v_cache_f16[pos * kv_dim]` */
static inline void lfm2_store_fp32_to_fp16_avx2(uint16_t * restrict dst_f16,
                                                const float * restrict src_f32,
                                                uint32_t n_elems) {
    for (uint32_t i = 0; i < n_elems; i += 8) {
        __m256 v = _mm256_loadu_ps(src_f32 + i);
        _mm_storeu_si128((__m128i *)(dst_f16 + i), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }
}
#endif

/* Portable Scalar Fallbacks for QK-Norm & FP16 Cache Write */
static inline void lfm2_qk_norm_rope_scalar(float * restrict vec_f32,
                                            uint16_t * restrict dst_f16,
                                            uint32_t n_heads,
                                            uint32_t head_dim,
                                            const GGUFTensor * restrict norm_w,
                                            float eps,
                                            const LFM2RoPELUT * restrict lut,
                                            uint32_t pos) {
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * 32;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * 32;
    const float * restrict nw    = (norm_w != NULL) ? (const float *)norm_w->data : NULL;
    (void)head_dim;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp = vec_f32 + (size_t)h * 64;
        if (nw != NULL) {
            double ss = 0.0;
            for (uint32_t i = 0; i < 64; ++i) ss += (double)hp[i] * (double)hp[i];
            float irms = 1.0f / sqrtf((float)(ss * (1.0 / 64.0)) + eps);
            for (uint32_t i = 0; i < 64; ++i) hp[i] = hp[i] * irms * nw[i];
        }
        for (uint32_t i = 0; i < 32; ++i) {
            float x0 = hp[i], x1 = hp[i + 32];
            float c = cos_p[i], s = sin_p[i];
            float r_lo = x0 * c - x1 * s;
            float r_hi = x1 * c + x0 * s;
            if (dst_f16 != NULL) {
                uint16_t * restrict kp = dst_f16 + (size_t)h * 64;
                kp[i]      = fp32_to_fp16(r_lo);
                kp[i + 32] = fp32_to_fp16(r_hi);
            } else {
                hp[i]      = r_lo;
                hp[i + 32] = r_hi;
            }
        }
    }
}

static inline void lfm2_store_fp32_to_fp16_scalar(uint16_t * restrict dst_f16,
                                                  const float * restrict src_f32,
                                                  uint32_t n_elems) {
    for (uint32_t i = 0; i < n_elems; ++i) {
        dst_f16[i] = fp32_to_fp16(src_f32[i]);
    }
}

#if defined(__aarch64__)
/* Helper: Unpacks 8 contiguous IEEE-754 FP16 values into two FP32 NEON registers */
LFM2_TARGET_ARM_FP16
static inline void neon_load_fp16x8_to_fp32(const uint16_t * restrict src,
                                            float32x4_t * restrict out_lo,
                                            float32x4_t * restrict out_hi) {
    const float16x8_t h8 = vreinterpretq_f16_u16(vld1q_u16(src));
    *out_lo = vcvt_f32_f16(vget_low_f16(h8));
    *out_hi = vcvt_f32_f16(vget_high_f16(h8));
}

/* Helper: Converts two FP32 NEON registers (8 floats) and stores 8 FP16 values */
LFM2_TARGET_ARM_FP16
static inline void neon_store_fp32_to_fp16x8(uint16_t * restrict dst,
                                             float32x4_t v_lo,
                                             float32x4_t v_hi) {
    const float16x8_t h8 = vcombine_f16(vcvt_f16_f32(v_lo), vcvt_f16_f32(v_hi));
    vst1q_u16(dst, vreinterpretq_u16_f16(h8));
}

/* ARM64 NEON FP32 -> FP16 conversion vector store */
#ifndef LFM2_SIMD_ARM64_INCLUDED
LFM2_TARGET_ARM_FP16
static inline void lfm2_store_fp32_to_fp16_neon(uint16_t * restrict dst_f16,
                                                const float * restrict src_f32,
                                                uint32_t n_elems) {
    uint32_t i = 0;
    for (; i + 8 <= n_elems; i += 8) {
        float32x4_t lo = vld1q_f32(src_f32 + i + 0);
        float32x4_t hi = vld1q_f32(src_f32 + i + 4);
        neon_store_fp32_to_fp16x8(dst_f16 + i, lo, hi);
    }
    for (; i < n_elems; ++i) {
        float16x4_t h = vcvt_f16_f32(vdupq_n_f32(src_f32[i]));
        dst_f16[i] = vget_lane_u16(vreinterpret_u16_f16(h), 0);
    }
}
#endif

/* ARM64 NEON + FP16 Per-Head QK-Norm & Precomputed RoPE LUT */
LFM2_TARGET_ARM_FP16
static inline void lfm2_qk_norm_rope_neon(float * restrict vec_f32,
                                          uint16_t * restrict dst_f16,
                                          uint32_t n_heads,
                                          uint32_t head_dim,
                                          const GGUFTensor * restrict norm_w,
                                          float eps,
                                          const LFM2RoPELUT * restrict lut,
                                          uint32_t pos) {
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * 32;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * 32;
    const float * restrict nw    = (norm_w != NULL) ? (const float *)norm_w->data : NULL;
    (void)head_dim; /* LFM2 head_dim = 64 */

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp    = vec_f32 + (size_t)h * 64;
        uint16_t * restrict kp = dst_f16 ? (dst_f16 + (size_t)h * 64) : NULL;

        /* 1. Compute Per-Head RMSNorm scale factor across all 64 lanes */
        float irms = 1.0f;
        if (nw != NULL) {
            float32x4_t ss0 = vdupq_n_f32(0.0f);
            float32x4_t ss1 = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < 64; i += 8) {
                const float32x4_t v0 = vld1q_f32(hp + i + 0);
                const float32x4_t v1 = vld1q_f32(hp + i + 4);
                ss0 = vfmaq_f32(ss0, v0, v0);
                ss1 = vfmaq_f32(ss1, v1, v1);
            }
            const float ss = vaddvq_f32(vaddq_f32(ss0, ss1));
            irms = 1.0f / sqrtf(ss * (1.0f / 64.0f) + eps);
        }
        const float32x4_t virms = vdupq_n_f32(irms);

        /* 2. Fused RMSNorm + NeoX Half-Split RoPE (8 lanes per iteration):
         *    r_lo = (x_lo * c) - (x_hi * s)  -> vfmsq_f32(vmulq_f32(x_lo, c), x_hi, s)
         *    r_hi = (x_hi * c) + (x_lo * s)  -> vfmaq_f32(vmulq_f32(x_hi, c), x_lo, s) */
        for (uint32_t i = 0; i < 32; i += 8) {
            float32x4_t x_lo0 = vld1q_f32(hp + i +  0), x_lo1 = vld1q_f32(hp + i +  4);
            float32x4_t x_hi0 = vld1q_f32(hp + i + 32), x_hi1 = vld1q_f32(hp + i + 36);

            if (nw != NULL) {
                x_lo0 = vmulq_f32(vmulq_f32(x_lo0, virms), vld1q_f32(nw + i +  0));
                x_lo1 = vmulq_f32(vmulq_f32(x_lo1, virms), vld1q_f32(nw + i +  4));
                x_hi0 = vmulq_f32(vmulq_f32(x_hi0, virms), vld1q_f32(nw + i + 32));
                x_hi1 = vmulq_f32(vmulq_f32(x_hi1, virms), vld1q_f32(nw + i + 36));
            }

            const float32x4_t c0 = vld1q_f32(cos_p + i + 0), c1 = vld1q_f32(cos_p + i + 4);
            const float32x4_t s0 = vld1q_f32(sin_p + i + 0), s1 = vld1q_f32(sin_p + i + 4);

            const float32x4_t r_lo0 = vfmsq_f32(vmulq_f32(x_lo0, c0), x_hi0, s0);
            const float32x4_t r_lo1 = vfmsq_f32(vmulq_f32(x_lo1, c1), x_hi1, s1);
            const float32x4_t r_hi0 = vfmaq_f32(vmulq_f32(x_hi0, c0), x_lo0, s0);
            const float32x4_t r_hi1 = vfmaq_f32(vmulq_f32(x_hi1, c1), x_lo1, s1);

            if (kp != NULL) {
                neon_store_fp32_to_fp16x8(kp + i +  0, r_lo0, r_lo1);
                neon_store_fp32_to_fp16x8(kp + i + 32, r_hi0, r_hi1);
            } else {
                vst1q_f32(hp + i +  0, r_lo0);
                vst1q_f32(hp + i +  4, r_lo1);
                vst1q_f32(hp + i + 32, r_hi0);
                vst1q_f32(hp + i + 36, r_hi1);
            }
        }
    }
}
#endif

#if LFM2_HAS_AVX2
    #define lfm2_qk_norm_rope_simd       lfm2_qk_norm_rope_avx2
    #define lfm2_store_fp32_to_fp16_simd lfm2_store_fp32_to_fp16_avx2
#elif defined(__aarch64__)
    #define lfm2_qk_norm_rope_simd       lfm2_qk_norm_rope_neon
    #define lfm2_store_fp32_to_fp16_simd lfm2_store_fp32_to_fp16_neon
#else
    #define lfm2_qk_norm_rope_simd       lfm2_qk_norm_rope_scalar
    #define lfm2_store_fp32_to_fp16_simd lfm2_store_fp32_to_fp16_scalar
#endif

/* ============================================================================
 * 3. Single-Pass Online Softmax (FlashAttention-CPU) over FP16 KV-Cache
 * ============================================================================
 * Streams `k_cache_f16` and `v_cache_f16` from t = 0 .. pos in ONE pass.
 * Keeps the 64-dim head output accumulator in 8 YMM registers (vo0..vo7),
 * requiring ZERO attention score scratch buffers (`s->att` is eliminated).
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool *pool;
    const float    *q;           /* [n_head_q * 64] FP32 rotated query heads   */
    const uint16_t *k_cache_f16; /* [max_seq_len * kv_dim] FP16 key cache      */
    const uint16_t *v_cache_f16; /* [max_seq_len * kv_dim] FP16 value cache    */
    float          *xb_out;      /* [n_head_q * 64] FP32 attention output      */
    block_q8_K_t   *xb_q8_out;   /* [n_head_q * 64 / 256] In-place Q8_K output */
    uint32_t        n_head_q;
    uint32_t        kv_mul;
    uint32_t        kv_dim;
    uint32_t        pos;
    float           scale;
} OnlineGQATaskCtx;

#if LFM2_HAS_AVX2
static inline void lfm2_online_gqa_f16_worker(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    OnlineGQATaskCtx *c = (OnlineGQATaskCtx *)arg;
    uint32_t group_start = 0, group_end = 0;

    /* Each work item is a group of 4 query heads (4 * 64 = 256 floats = 1 Q8_K block)
     * so each worker can quantize its 256-float output block into `xb_q8_out` in L1! */
    while (lfm2_threadpool_claim_chunk(c->pool, &group_start, &group_end)) {
        for (uint32_t g = group_start; g < group_end; ++g) {
            const uint32_t h_base = g * 4;
            const uint32_t h_end  = (h_base + 4 <= c->n_head_q) ? (h_base + 4) : c->n_head_q;

            for (uint32_t h = h_base; h < h_end; ++h) {
                const float * restrict q_h = c->q + (size_t)h * 64;
                const uint32_t kv_h        = h / c->kv_mul;

                /* Pin the 64-float Query head in 8 YMM registers */
                const __m256 vq0 = _mm256_loadu_ps(q_h +  0);
                const __m256 vq1 = _mm256_loadu_ps(q_h +  8);
                const __m256 vq2 = _mm256_loadu_ps(q_h + 16);
                const __m256 vq3 = _mm256_loadu_ps(q_h + 24);
                const __m256 vq4 = _mm256_loadu_ps(q_h + 32);
                const __m256 vq5 = _mm256_loadu_ps(q_h + 40);
                const __m256 vq6 = _mm256_loadu_ps(q_h + 48);
                const __m256 vq7 = _mm256_loadu_ps(q_h + 56);

                /* Running 64-float unnormalized attention output in 8 YMM registers */
                __m256 vo0 = _mm256_setzero_ps(), vo1 = _mm256_setzero_ps();
                __m256 vo2 = _mm256_setzero_ps(), vo3 = _mm256_setzero_ps();
                __m256 vo4 = _mm256_setzero_ps(), vo5 = _mm256_setzero_ps();
                __m256 vo6 = _mm256_setzero_ps(), vo7 = _mm256_setzero_ps();

                float m_prev = -1e30f; /* Running max logit */
                float l_prev = 0.0f;   /* Running softmax denominator */

                const uint16_t *k_ptr = c->k_cache_f16 + (size_t)kv_h * 64;
                const uint16_t *v_ptr = c->v_cache_f16 + (size_t)kv_h * 64;
                const size_t stride   = c->kv_dim;

                for (uint32_t t = 0; t <= c->pos; ++t) {
                    if (t + 2 <= c->pos) {
                        _mm_prefetch((const char *)(k_ptr + 2 * stride), _MM_HINT_T0);
                        _mm_prefetch((const char *)(v_ptr + 2 * stride), _MM_HINT_T0);
                    }

                    /* 1. F16C load of K_t[0..63] + 64-dim dot product with Q_h */
                    __m256 d0 = _mm256_mul_ps(vq0, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr +  0))));
                    __m256 d1 = _mm256_mul_ps(vq1, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr +  8))));
                    d0 = _mm256_fmadd_ps(vq2, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 16))), d0);
                    d1 = _mm256_fmadd_ps(vq3, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 24))), d1);
                    d0 = _mm256_fmadd_ps(vq4, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 32))), d0);
                    d1 = _mm256_fmadd_ps(vq5, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 40))), d1);
                    d0 = _mm256_fmadd_ps(vq6, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 48))), d0);
                    d1 = _mm256_fmadd_ps(vq7, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + 56))), d1);

                    const float score = hsum_ps_avx2(_mm256_add_ps(d0, d1)) * c->scale;

                    /* 2. Online Softmax Rescaling & F16C Value Accumulation */
                    if (score > m_prev) {
                        const float alpha   = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
                        const __m256 valpha = _mm256_set1_ps(alpha);
                        l_prev = l_prev * alpha + 1.0f;
                        m_prev = score;

                        vo0 = _mm256_fmadd_ps(vo0, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr +  0))));
                        vo1 = _mm256_fmadd_ps(vo1, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr +  8))));
                        vo2 = _mm256_fmadd_ps(vo2, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 16))));
                        vo3 = _mm256_fmadd_ps(vo3, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 24))));
                        vo4 = _mm256_fmadd_ps(vo4, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 32))));
                        vo5 = _mm256_fmadd_ps(vo5, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 40))));
                        vo6 = _mm256_fmadd_ps(vo6, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 48))));
                        vo7 = _mm256_fmadd_ps(vo7, valpha, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 56))));
                    } else {
                        const float beta   = expf(score - m_prev);
                        const __m256 vbeta = _mm256_set1_ps(beta);
                        l_prev += beta;

                        vo0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr +  0))), vbeta, vo0);
                        vo1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr +  8))), vbeta, vo1);
                        vo2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 16))), vbeta, vo2);
                        vo3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 24))), vbeta, vo3);
                        vo4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 32))), vbeta, vo4);
                        vo5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 40))), vbeta, vo5);
                        vo6 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 48))), vbeta, vo6);
                        vo7 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + 56))), vbeta, vo7);
                    }

                    k_ptr += stride;
                    v_ptr += stride;
                }

                /* 3. Normalize by 1.0f / l_prev and store to xb_out[h * 64] */
                const __m256 v_inv_l = _mm256_set1_ps(1.0f / l_prev);
                float * restrict out_h = c->xb_out + (size_t)h * 64;
                _mm256_storeu_ps(out_h +  0, _mm256_mul_ps(vo0, v_inv_l));
                _mm256_storeu_ps(out_h +  8, _mm256_mul_ps(vo1, v_inv_l));
                _mm256_storeu_ps(out_h + 16, _mm256_mul_ps(vo2, v_inv_l));
                _mm256_storeu_ps(out_h + 24, _mm256_mul_ps(vo3, v_inv_l));
                _mm256_storeu_ps(out_h + 32, _mm256_mul_ps(vo4, v_inv_l));
                _mm256_storeu_ps(out_h + 40, _mm256_mul_ps(vo5, v_inv_l));
                _mm256_storeu_ps(out_h + 48, _mm256_mul_ps(vo6, v_inv_l));
                _mm256_storeu_ps(out_h + 56, _mm256_mul_ps(vo7, v_inv_l));
            }

            /* Immediately quantize this L1-hot 256-float (4-head) slice into xb_q8_out[g] */
            if (h_end - h_base == 4 && c->xb_q8_out != NULL) {
                lfm2_quantize_row_q8_K_avx2(c->xb_out + (size_t)h_base * 64, &c->xb_q8_out[g], 256);
            }
        }
    }
}
#endif

/* Portable C11 Scalar Fallback */
static inline void lfm2_online_gqa_f16_worker_scalar(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    OnlineGQATaskCtx *c = (OnlineGQATaskCtx *)arg;
    uint32_t group_start = 0, group_end = 0;

    while (lfm2_threadpool_claim_chunk(c->pool, &group_start, &group_end)) {
        for (uint32_t g = group_start; g < group_end; ++g) {
            const uint32_t h_base = g * 4;
            const uint32_t h_end  = (h_base + 4 <= c->n_head_q) ? (h_base + 4) : c->n_head_q;

            for (uint32_t h = h_base; h < h_end; ++h) {
                const float * restrict q_h = c->q + (size_t)h * 64;
                const uint32_t kv_h        = h / c->kv_mul;
                float vo[64] = {0.0f};
                float m_prev = -1e30f;
                float l_prev = 0.0f;

                const uint16_t *k_ptr = c->k_cache_f16 + (size_t)kv_h * 64;
                const uint16_t *v_ptr = c->v_cache_f16 + (size_t)kv_h * 64;
                const size_t stride   = c->kv_dim;

                for (uint32_t t = 0; t <= c->pos; ++t) {
                    float dot = 0.0f;
                    for (uint32_t i = 0; i < 64; ++i) {
                        dot += q_h[i] * fp16_to_fp32(k_ptr[i]);
                    }
                    float score = dot * c->scale;

                    if (score > m_prev) {
                        float alpha = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
                        l_prev = l_prev * alpha + 1.0f;
                        m_prev = score;
                        for (uint32_t i = 0; i < 64; ++i) {
                            vo[i] = vo[i] * alpha + fp16_to_fp32(v_ptr[i]);
                        }
                    } else {
                        float beta = expf(score - m_prev);
                        l_prev += beta;
                        for (uint32_t i = 0; i < 64; ++i) {
                            vo[i] += beta * fp16_to_fp32(v_ptr[i]);
                        }
                    }
                    k_ptr += stride;
                    v_ptr += stride;
                }

                float inv_l = 1.0f / l_prev;
                float * restrict out_h = c->xb_out + (size_t)h * 64;
                for (uint32_t i = 0; i < 64; ++i) {
                    out_h[i] = vo[i] * inv_l;
                }
            }

            if (h_end - h_base == 4 && c->xb_q8_out != NULL) {
                lfm2_quantize_row_q8_K_scalar(c->xb_out + (size_t)h_base * 64, &c->xb_q8_out[g], 256);
            }
        }
    }
}

#if defined(__aarch64__)
/* ARM64 NEON + FP16 Single-Pass Online Softmax GQA Worker */
LFM2_TARGET_ARM_FP16
static inline void lfm2_online_gqa_f16_worker_neon(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    OnlineGQATaskCtx *c = (OnlineGQATaskCtx *)arg;
    uint32_t group_start = 0, group_end = 0;

    while (lfm2_threadpool_claim_chunk(c->pool, &group_start, &group_end)) {
        for (uint32_t g = group_start; g < group_end; ++g) {
            const uint32_t h_base = g * 4;
            const uint32_t h_end  = (h_base + 4 <= c->n_head_q) ? (h_base + 4) : c->n_head_q;

            for (uint32_t h = h_base; h < h_end; ++h) {
                const float * restrict q_h = c->q + (size_t)h * 64;
                const uint32_t kv_h        = h / c->kv_mul;

                /* Output accumulator in 16 NEON registers (vo0..vo15) */
                float32x4_t vo0  = vdupq_n_f32(0.0f), vo1  = vdupq_n_f32(0.0f);
                float32x4_t vo2  = vdupq_n_f32(0.0f), vo3  = vdupq_n_f32(0.0f);
                float32x4_t vo4  = vdupq_n_f32(0.0f), vo5  = vdupq_n_f32(0.0f);
                float32x4_t vo6  = vdupq_n_f32(0.0f), vo7  = vdupq_n_f32(0.0f);
                float32x4_t vo8  = vdupq_n_f32(0.0f), vo9  = vdupq_n_f32(0.0f);
                float32x4_t vo10 = vdupq_n_f32(0.0f), vo11 = vdupq_n_f32(0.0f);
                float32x4_t vo12 = vdupq_n_f32(0.0f), vo13 = vdupq_n_f32(0.0f);
                float32x4_t vo14 = vdupq_n_f32(0.0f), vo15 = vdupq_n_f32(0.0f);

                float m_prev = -1e30f;
                float l_prev = 0.0f;

                const uint16_t *k_ptr = c->k_cache_f16 + (size_t)kv_h * 64;
                const uint16_t *v_ptr = c->v_cache_f16 + (size_t)kv_h * 64;
                const size_t stride   = c->kv_dim;

                for (uint32_t t = 0; t <= c->pos; ++t) {
                    if (t + 2 <= c->pos) {
                        __builtin_prefetch(k_ptr + 2 * stride, 0, 0);
                        __builtin_prefetch(v_ptr + 2 * stride, 0, 0);
                    }

                    /* 1. Hardware FP16 -> FP32 load of K_t[0..63] + dot product with Q_h
                     * We stream Q_h in 8-lane chunks, using 4 scratch registers for loads/products.
                     * This leaves vo0..vo15 completely unspilled in registers! */
                    float32x4_t d0 = vdupq_n_f32(0.0f), d1 = vdupq_n_f32(0.0f);
                    for (uint32_t i = 0; i < 64; i += 8) {
                        float32x4_t k_lo, k_hi;
                        neon_load_fp16x8_to_fp32(k_ptr + i, &k_lo, &k_hi);
                        const float32x4_t q_lo = vld1q_f32(q_h + i + 0);
                        const float32x4_t q_hi = vld1q_f32(q_h + i + 4);
                        d0 = vfmaq_f32(d0, q_lo, k_lo);
                        d1 = vfmaq_f32(d1, q_hi, k_hi);
                    }

                    const float score = vaddvq_f32(vaddq_f32(d0, d1)) * c->scale;

                    /* 2. Single-Pass Online Softmax Rescaling & FP16 Value Accumulation */
                    float32x4_t v_lo, v_hi;
                    if (score > m_prev) {
                        const float alpha = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
                        l_prev = l_prev * alpha + 1.0f;
                        m_prev = score;

                        /* vfmaq_n_f32(v_t, vo, alpha) -> v_t + (vo * alpha) */
                        neon_load_fp16x8_to_fp32(v_ptr +  0, &v_lo, &v_hi);
                        vo0  = vfmaq_n_f32(v_lo, vo0,  alpha); vo1  = vfmaq_n_f32(v_hi, vo1,  alpha);
                        neon_load_fp16x8_to_fp32(v_ptr +  8, &v_lo, &v_hi);
                        vo2  = vfmaq_n_f32(v_lo, vo2,  alpha); vo3  = vfmaq_n_f32(v_hi, vo3,  alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 16, &v_lo, &v_hi);
                        vo4  = vfmaq_n_f32(v_lo, vo4,  alpha); vo5  = vfmaq_n_f32(v_hi, vo5,  alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 24, &v_lo, &v_hi);
                        vo6  = vfmaq_n_f32(v_lo, vo6,  alpha); vo7  = vfmaq_n_f32(v_hi, vo7,  alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 32, &v_lo, &v_hi);
                        vo8  = vfmaq_n_f32(v_lo, vo8,  alpha); vo9  = vfmaq_n_f32(v_hi, vo9,  alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 40, &v_lo, &v_hi);
                        vo10 = vfmaq_n_f32(v_lo, vo10, alpha); vo11 = vfmaq_n_f32(v_hi, vo11, alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 48, &v_lo, &v_hi);
                        vo12 = vfmaq_n_f32(v_lo, vo12, alpha); vo13 = vfmaq_n_f32(v_hi, vo13, alpha);
                        neon_load_fp16x8_to_fp32(v_ptr + 56, &v_lo, &v_hi);
                        vo14 = vfmaq_n_f32(v_lo, vo14, alpha); vo15 = vfmaq_n_f32(v_hi, vo15, alpha);
                    } else {
                        const float beta = expf(score - m_prev);
                        l_prev += beta;

                        /* vfmaq_n_f32(vo, v_t, beta) -> vo + (v_t * beta) */
                        neon_load_fp16x8_to_fp32(v_ptr +  0, &v_lo, &v_hi);
                        vo0  = vfmaq_n_f32(vo0,  v_lo, beta); vo1  = vfmaq_n_f32(vo1,  v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr +  8, &v_lo, &v_hi);
                        vo2  = vfmaq_n_f32(vo2,  v_lo, beta); vo3  = vfmaq_n_f32(vo3,  v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 16, &v_lo, &v_hi);
                        vo4  = vfmaq_n_f32(vo4,  v_lo, beta); vo5  = vfmaq_n_f32(vo5,  v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 24, &v_lo, &v_hi);
                        vo6  = vfmaq_n_f32(vo6,  v_lo, beta); vo7  = vfmaq_n_f32(vo7,  v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 32, &v_lo, &v_hi);
                        vo8  = vfmaq_n_f32(vo8,  v_lo, beta); vo9  = vfmaq_n_f32(vo9,  v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 40, &v_lo, &v_hi);
                        vo10 = vfmaq_n_f32(vo10, v_lo, beta); vo11 = vfmaq_n_f32(vo11, v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 48, &v_lo, &v_hi);
                        vo12 = vfmaq_n_f32(vo12, v_lo, beta); vo13 = vfmaq_n_f32(vo13, v_hi, beta);
                        neon_load_fp16x8_to_fp32(v_ptr + 56, &v_lo, &v_hi);
                        vo14 = vfmaq_n_f32(vo14, v_lo, beta); vo15 = vfmaq_n_f32(vo15, v_hi, beta);
                    }

                    k_ptr += stride;
                    v_ptr += stride;
                }

                /* 3. Normalize by 1.0f / l_prev and store to xb_out[h * 64] */
                const float inv_l = 1.0f / l_prev;
                float * restrict out_h = c->xb_out + (size_t)h * 64;
                vst1q_f32(out_h +  0, vmulq_n_f32(vo0,  inv_l));
                vst1q_f32(out_h +  4, vmulq_n_f32(vo1,  inv_l));
                vst1q_f32(out_h +  8, vmulq_n_f32(vo2,  inv_l));
                vst1q_f32(out_h + 12, vmulq_n_f32(vo3,  inv_l));
                vst1q_f32(out_h + 16, vmulq_n_f32(vo4,  inv_l));
                vst1q_f32(out_h + 20, vmulq_n_f32(vo5,  inv_l));
                vst1q_f32(out_h + 24, vmulq_n_f32(vo6,  inv_l));
                vst1q_f32(out_h + 28, vmulq_n_f32(vo7,  inv_l));
                vst1q_f32(out_h + 32, vmulq_n_f32(vo8,  inv_l));
                vst1q_f32(out_h + 36, vmulq_n_f32(vo9,  inv_l));
                vst1q_f32(out_h + 40, vmulq_n_f32(vo10, inv_l));
                vst1q_f32(out_h + 44, vmulq_n_f32(vo11, inv_l));
                vst1q_f32(out_h + 48, vmulq_n_f32(vo12, inv_l));
                vst1q_f32(out_h + 52, vmulq_n_f32(vo13, inv_l));
                vst1q_f32(out_h + 56, vmulq_n_f32(vo14, inv_l));
                vst1q_f32(out_h + 60, vmulq_n_f32(vo15, inv_l));
            }

            /* Immediately quantize L1-hot 256-float (4-head) slice into xb_q8_out[g] */
            if (h_end - h_base == 4 && c->xb_q8_out != NULL) {
                lfm2_quantize_row_q8_K_neon(c->xb_out + (size_t)h_base * 64, &c->xb_q8_out[g], 256);
            }
        }
    }
}
#endif

#if LFM2_HAS_AVX2
    #define lfm2_online_gqa_f16_worker_simd lfm2_online_gqa_f16_worker
#elif defined(__aarch64__)
    #define lfm2_online_gqa_f16_worker_simd lfm2_online_gqa_f16_worker_neon
#else
    #define lfm2_online_gqa_f16_worker_simd lfm2_online_gqa_f16_worker_scalar
#endif

/* Complete Fused Single-Token GQA Attention Layer (with FP16 KV-Cache & Online Softmax) */
static inline void lfm2_forward_gqa_fast(LFM2ThreadPool *pool,
                                         float * restrict x,
                                         float * restrict xb,
                                         float * restrict q,
                                         float * restrict k,
                                         float * restrict v,
                                         uint16_t * restrict k_cache_f16,
                                         uint16_t * restrict v_cache_f16,
                                         block_q8_K_t * restrict xq_scratch,
                                         const LFM2Layer * restrict layer,
                                         const LFM2RoPELUT * restrict rope_lut,
                                         uint32_t D,
                                         uint32_t n_head_q,
                                         uint32_t n_head_kv,
                                         uint32_t pos,
                                         float eps) {
    const uint32_t head_dim = D / n_head_q; /* 64 */
    const uint32_t kv_dim   = n_head_kv * head_dim;

    /* 1. Fused RMSNorm(x -> xb) + Quantize(xb -> xq_scratch) */
    lfm2_rmsnorm_quantize_q8_K_simd(xb, xq_scratch, x, layer->attn_norm, D, eps);

    /* 2. Fused Single-Barrier 1x4 Q, K, V Projection */
    lfm2_gqa_qkv_fused_1x4_mt(pool, q, k, v, layer->attn.wq, layer->attn.wk, layer->attn.wv,
                              xb, xq_scratch);

    /* 3. Vectorized Per-Head QK-Norm + RoPE LUT + Direct F16C Cache Write */
    lfm2_qk_norm_rope_simd(q, NULL, n_head_q, head_dim, layer->attn.q_norm, eps, rope_lut, pos);
    lfm2_qk_norm_rope_simd(k, k_cache_f16 + (size_t)pos * kv_dim, n_head_kv, head_dim,
                           layer->attn.k_norm, eps, rope_lut, pos);
    lfm2_store_fp32_to_fp16_simd(v_cache_f16 + (size_t)pos * kv_dim, v, kv_dim);

    /* 4. Single-Pass Online Softmax GQA over FP16 KV-Cache + In-Place Q8_K Quantize */
    OnlineGQATaskCtx gqa_ctx = {
        .pool        = pool,
        .q           = q,
        .k_cache_f16 = k_cache_f16,
        .v_cache_f16 = v_cache_f16,
        .xb_out      = xb,
        .xb_q8_out   = xq_scratch,
        .n_head_q    = n_head_q,
        .kv_mul      = n_head_q / n_head_kv,
        .kv_dim      = kv_dim,
        .pos         = pos,
        .scale       = 1.0f / sqrtf((float)head_dim)
    };
    const uint32_t n_head_groups = (n_head_q + 3) / 4;
    lfm2_parallel_for(pool, n_head_groups, 1, lfm2_online_gqa_f16_worker_simd, &gqa_ctx);

    /* 5. 1x4 Unrolled Output Projection (attn_output.weight) + Residual Add */
    lfm2_matmul_1x4_prequant_mt(pool, q, layer->attn.wo, xb, xq_scratch);
    lfm2_vec_add_inplace(x, q, D);
}

/* ============================================================================
 * 4. High-Level Fused Fast-Path Context & Token Forward Pass
 * ============================================================================ */
typedef struct {
    LFM2PlanarConvWeights planar_conv[LFM2_MAX_LAYERS];
    LFM2RoPELUT           rope_lut;
    bool                  initialized;
} LFM2FastContext;

static inline bool lfm2_fast_context_init(LFM2FastContext *fc, const LFM2Model *m, uint32_t max_seq_len) {
    if (!fc || !m) return false;
    memset(fc, 0, sizeof(*fc));

    const uint32_t D        = m->cfg.n_embd;
    const uint32_t head_dim = m->cfg.n_head_q > 0 ? (D / m->cfg.n_head_q) : 64;
    const float freq_base   = (m->cfg.rope_freq_base > 0.0f) ? m->cfg.rope_freq_base : 1000000.0f;

    /* Fast-path requires standard Liquid AI architecture specs: head_dim == 64 and D % 256 == 0 */
    if (head_dim != 64 || (D % 256) != 0) {
        return false;
    }

    /* Initialize precomputed RoPE LUT */
    if (!lfm2_rope_lut_init(&fc->rope_lut, max_seq_len, head_dim, freq_base)) {
        return false;
    }

    /* Initialize planar convolution weights for all ShortConv layers */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            const GGUFTensor *conv_t = m->layers[l].shortconv.conv;
            if (conv_t && conv_t->data) {
                if (!lfm2_planar_conv_init(&fc->planar_conv[l], conv_t, D)) {
                    lfm2_rope_lut_free(&fc->rope_lut);
                    return false;
                }
            }
        }
    }

    fc->initialized = true;
    return true;
}

static inline void lfm2_fast_context_free(LFM2FastContext *fc) {
    if (!fc || !fc->initialized) return;
    for (uint32_t l = 0; l < LFM2_MAX_LAYERS; ++l) {
        lfm2_planar_conv_free(&fc->planar_conv[l]);
    }
    lfm2_rope_lut_free(&fc->rope_lut);
    fc->initialized = false;
}

/* Single-token autoregressive decoding via 100% Fused Fast-Path Micro-Kernels */
static inline float *lfm2_forward_token_fast(const LFM2Model *m,
                                             LFM2State *s,
                                             LFM2FastContext *fc,
                                             uint32_t token_id,
                                             uint32_t pos,
                                             bool compute_logits) {
    const uint32_t D    = m->cfg.n_embd;
    const float    eps  = m->cfg.rms_norm_eps;

    if (token_id >= m->cfg.n_vocab || pos >= s->max_seq_len) return NULL;

    /* 1. Embedding Lookup */
    lfm2_dequantize_row(m->token_embd, token_id, s->x);

    /* 2. Hybrid Transformer / ShortConv Layers */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv_fast(&s->pool, s->x, s->xb, s->bcx, s->conv_out,
                                        s->conv_state[l], s->xq_scratch, layer,
                                        &fc->planar_conv[l], D, eps);
        } else {
            const uint32_t n_head_q  = m->cfg.n_head_q;
            const uint32_t n_head_kv = m->cfg.n_head_kv[l];
            lfm2_forward_gqa_fast(&s->pool, s->x, s->xb, s->q, s->k, s->v,
                                  s->k_cache_f16[l], s->v_cache_f16[l], s->xq_scratch,
                                  layer, &fc->rope_lut, D, n_head_q, n_head_kv, pos, eps);
        }

        /* SwiGLU FFN Layer via Fused 1x4 Micro-Kernels */
        lfm2_forward_ffn_fused_1x4(&s->pool, s->x, s->xb, s->hb, s->xq_scratch, layer, D, eps);
    }

    if (!compute_logits) return NULL;

    /* 3. Output RMSNorm + LM Head Projection */
    if (m->output->type == GGML_TYPE_Q4_K || m->output->type == GGML_TYPE_Q6_K ||
        m->output->type == GGML_TYPE_Q8_0 || m->output->type == GGML_TYPE_Q4_0 ||
        m->output->type == GGML_TYPE_Q5_K || m->output->type == GGML_TYPE_IQ4_XS ||
        m->output->type == GGML_TYPE_F32) {
        lfm2_rmsnorm_quantize_q8_K_simd(s->xb, s->xq_scratch, s->x, m->output_norm, D, eps);
        lfm2_matmul_1x4_prequant_mt(&s->pool, s->logits, m->output, s->xb, s->xq_scratch);
    } else {
        lfm2_rmsnorm(s->x, s->x, m->output_norm, D, eps);
        lfm2_matmul_vec(s->logits, m->output, s->x, s->xq_scratch, &s->pool);
    }
    return s->logits;
}

#endif /* LFM2_KERNELS_FAST_INCLUDED */

/* ============================================================================
 * 5. Standalone TDD Numerical Parity & Throughput Benchmark
 * ============================================================================ */
#ifndef LFM2_KERNELS_FAST_NO_MAIN
int main(int argc, char **argv) {
    const uint32_t D         = 2048;
    const uint32_t n_head_q  = 32;
    const uint32_t n_head_kv = 8;
    const uint32_t head_dim  = 64;
    const uint32_t kv_dim    = n_head_kv * head_dim;
    const uint32_t seq_len   = 1024;
    const uint32_t pos       = seq_len - 1;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0);

    printf("=== LFM2 Fast Kernels: Planar ShortConv, RoPE LUT & FP16 Online Softmax ===\n");
    printf("Physical Cores Active : %u\n\n", pool.n_threads);

    /* ------------------------------------------------------------------------
     * Test 1: AVX2 Planar 3-Tap ShortConv vs Scalar Stride-3 Reference
     * ------------------------------------------------------------------------ */
    float *bcx         = (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *conv_weights= (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *state_ref   = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    float *state_avx   = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    float *out_ref     = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *out_avx     = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    block_q8_K_t *out_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256) * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < 3 * D; ++i) {
        bcx[i]          = sinf((float)i * 0.013f);
        conv_weights[i] = cosf((float)i * 0.029f) * 0.5f;
    }
    for (uint32_t i = 0; i < 2 * D; ++i) {
        state_ref[i] = state_avx[i] = sinf((float)i * 0.007f);
    }

    GGUFTensor t_conv = { .n_dims = 2, .ne = {3, D, 1, 1}, .type = GGML_TYPE_F32, .data = conv_weights };
    LFM2PlanarConvWeights pw;
    lfm2_planar_conv_init(&pw, &t_conv, D);

    /* Scalar Reference */
    for (uint32_t d = 0; d < D; ++d) {
        float u_t = bcx[d] * bcx[2 * D + d];
        float acc = conv_weights[d * 3 + 0] * state_ref[d]
                  + conv_weights[d * 3 + 1] * state_ref[D + d]
                  + conv_weights[d * 3 + 2] * u_t;
        out_ref[d]       = bcx[D + d] * acc;
        state_ref[d]     = state_ref[D + d];
        state_ref[D + d] = u_t;
    }

    /* Fused SIMD Planar ShortConv + Q8_K Quantize */
    lfm2_shortconv_gate_conv_quant_simd(out_avx, out_q8, bcx, state_avx, &pw, D);

    float max_conv_err = 0.0f;
    for (uint32_t d = 0; d < D; ++d) {
        float err = fabsf(out_ref[d] - out_avx[d]);
        if (err > max_conv_err) max_conv_err = err;
    }
    printf("1. ShortConv 3-Tap AVX2 Max Diff : %.8f (Bit-exact FMA parity)\n", max_conv_err);

    /* ------------------------------------------------------------------------
     * Test 2: Precomputed RoPE LUT + AVX2 QK-Norm vs Scalar powf/cosf/sinf
     * ------------------------------------------------------------------------ */
    LFM2RoPELUT rope_lut;
    lfm2_rope_lut_init(&rope_lut, seq_len, head_dim, 1000000.0f);

    float *q_ref  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *q_avx  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *q_norm = (float *)lfm2_aligned_alloc_64(head_dim * sizeof(float));
    for (uint32_t i = 0; i < D; ++i) q_ref[i] = q_avx[i] = cosf((float)i * 0.017f);
    for (uint32_t i = 0; i < head_dim; ++i) q_norm[i] = 0.9f + 0.1f * sinf((float)i);

    GGUFTensor t_qnorm = { .n_dims = 1, .ne = {head_dim, 1, 1, 1}, .type = GGML_TYPE_F32, .data = q_norm };

    /* Scalar QK-Norm + RoPE */
    for (uint32_t h = 0; h < n_head_q; ++h) {
        float *hp = q_ref + (size_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) ss += (double)hp[i] * (double)hp[i];
        float irms = 1.0f / sqrtf((float)(ss / head_dim) + 1e-5f);
        for (uint32_t i = 0; i < head_dim; ++i) hp[i] = hp[i] * irms * q_norm[i];
        for (uint32_t i = 0; i < 32; ++i) {
            float freq = powf(1000000.0f, -((float)(2 * i) / 64.0f));
            float c = cosf((float)pos * freq), s = sinf((float)pos * freq);
            float x0 = hp[i], x1 = hp[i + 32];
            hp[i] = x0 * c - x1 * s;
            hp[i + 32] = x1 * c + x0 * s;
        }
    }
    lfm2_qk_norm_rope_simd(q_avx, NULL, n_head_q, head_dim, &t_qnorm, 1e-5f, &rope_lut, pos);

    float max_rope_err = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float err = fabsf(q_ref[i] - q_avx[i]);
        if (err > max_rope_err) max_rope_err = err;
    }
    printf("2. QK-Norm + RoPE LUT Max Diff   : %.8f\n", max_rope_err);

    /* ------------------------------------------------------------------------
     * Test 3: Single-Pass FP16 Online Softmax GQA vs Two-Pass FP32 Reference
     * ------------------------------------------------------------------------ */
    float    *k_cache_f32 = (float *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(float));
    float    *v_cache_f32 = (float *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(float));
    uint16_t *k_cache_f16 = (uint16_t *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *v_cache_f16 = (uint16_t *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(uint16_t));
    float    *att_scratch = (float *)lfm2_aligned_alloc_64((size_t)n_head_q * seq_len * sizeof(float));
    float    *gqa_ref     = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float    *gqa_online  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));

    for (size_t i = 0; i < (size_t)seq_len * kv_dim; ++i) {
        k_cache_f32[i] = sinf((float)i * 0.0031f);
        v_cache_f32[i] = cosf((float)i * 0.0023f);
    }
    lfm2_store_fp32_to_fp16_simd(k_cache_f16, k_cache_f32, seq_len * kv_dim);
    lfm2_store_fp32_to_fp16_simd(v_cache_f16, v_cache_f32, seq_len * kv_dim);

    GQAAttnTaskCtx ref_ctx = {
        .pool = &pool, .q = q_avx, .k_cache = k_cache_f32, .v_cache = v_cache_f32,
        .att = att_scratch, .xb = gqa_ref, .n_head_q = n_head_q, .kv_mul = 4,
        .head_dim = 64, .kv_dim = kv_dim, .max_seq_len = seq_len, .pos = pos, .scale = 0.125f
    };
    lfm2_parallel_for(&pool, n_head_q, 1, lfm2_gqa_heads_worker_task, &ref_ctx);

    OnlineGQATaskCtx online_ctx = {
        .pool = &pool, .q = q_avx, .k_cache_f16 = k_cache_f16, .v_cache_f16 = v_cache_f16,
        .xb_out = gqa_online, .xb_q8_out = out_q8, .n_head_q = n_head_q, .kv_mul = 4,
        .kv_dim = kv_dim, .pos = pos, .scale = 0.125f
    };
    lfm2_parallel_for(&pool, n_head_q / 4, 1, lfm2_online_gqa_f16_worker_simd, &online_ctx);

    float max_gqa_err = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float err = fabsf(gqa_ref[i] - gqa_online[i]);
        if (err > max_gqa_err) max_gqa_err = err;
    }
    printf("3. FP16 Online Softmax Max Diff  : %.6f (at seq_len = %u)\n\n", max_gqa_err, seq_len);

    /* Benchmark 300 calls of Two-Pass FP32 GQA vs Single-Pass FP16 Online Softmax GQA */
    const int iters = 300;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_parallel_for(&pool, n_head_q, 1, lfm2_gqa_heads_worker_task, &ref_ctx);
    }
    double f32_ms = (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_parallel_for(&pool, n_head_q / 4, 1, lfm2_online_gqa_f16_worker_simd, &online_ctx);
    }
    double f16_ms = (now_ms() - t0) / iters;

    printf("Two-Pass FP32 GQA (seq=%u)          : %.3f ms/layer\n", seq_len, f32_ms);
    printf("Single-Pass FP16 Online Softmax GQA : %.3f ms/layer (%.2fx faster, 50%% KV RAM)\n",
            f16_ms, f32_ms / f16_ms);

    lfm2_planar_conv_free(&pw);
    lfm2_rope_lut_free(&rope_lut);
    free(bcx); free(conv_weights); free(state_ref); free(state_avx);
    free(out_ref); free(out_avx); free(out_q8); free(q_ref); free(q_avx); free(q_norm);
    free(k_cache_f32); free(v_cache_f32); free(k_cache_f16); free(v_cache_f16);
    free(att_scratch); free(gqa_ref); free(gqa_online);
    lfm2_threadpool_free(&pool);
    return EXIT_SUCCESS;
}
#endif
