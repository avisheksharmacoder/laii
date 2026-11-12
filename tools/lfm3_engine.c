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
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#ifndef LFM3_ENGINE_INCLUDED
#define LFM3_ENGINE_INCLUDED 1

#ifndef LFM2_SPECULATIVE_PLD_INCLUDED
#define LFM2_SPECULATIVE_PLD_NO_MAIN 1
#if __has_include("tools/lfm2_speculative_pld.c")
#include "tools/lfm2_speculative_pld.c"
#elif __has_include("lfm2_speculative_pld.c")
#include "lfm2_speculative_pld.c"
#else
#include "../tools/lfm2_speculative_pld.c"
#endif
#endif

#ifndef LFM2_KERNELS_FAST_INCLUDED
#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_DECODE_1X4_NO_MAIN 1
#define LFM2_KERNELS_FAST_NO_MAIN 1
#if __has_include("tools/lfm2_kernels_fast.c")
#include "tools/lfm2_kernels_fast.c"
#elif __has_include("lfm2_kernels_fast.c")
#include "lfm2_kernels_fast.c"
#else
#include "../tools/lfm2_kernels_fast.c"
#endif
#endif

#ifndef LFM3_SHORTCONV_TAPS
#define LFM3_SHORTCONV_TAPS 7U
#endif
#ifndef LFM3_SHORTCONV_HIST
#define LFM3_SHORTCONV_HIST 6U
#endif

/* ============================================================================
 * 0. Runtime ISA Helpers & Portable Conversion Utilities
 * ============================================================================ */
static inline void *lfm3_aligned_calloc_64(size_t size) {
    size_t aligned = (size + 63U) & ~((size_t)63U);
    if (aligned == 0U) aligned = 64U;
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64U, aligned) != 0) return NULL;
    memset(ptr, 0, aligned);
    return ptr;
}

static inline bool lfm3_is_forced_scalar(void) {
    const char *force_isa = getenv("LFM2_FORCE_ISA");
    return (force_isa && strcmp(force_isa, "scalar") == 0);
}

static inline bool lfm3_use_avx2(void) {
#if LFM2_HAS_AVX2
    if (lfm3_is_forced_scalar()) return false;
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    return (kt->has_avx2 && kt->has_fma && kt->has_f16c);
#else
    return false;
#endif
}

static inline bool lfm3_use_neon(void) {
#if defined(__aarch64__)
    return !lfm3_is_forced_scalar();
#else
    return false;
#endif
}

static inline float lfm3_read_float_elem(const GGUFTensor *t, size_t idx) {
    if (!t || !t->data) return 1.0f;
    if (t->type == GGML_TYPE_F32)  return ((const float *)t->data)[idx];
    if (t->type == GGML_TYPE_F16)  return ggml_fp16_to_fp32(((const uint16_t *)t->data)[idx]);
    if (t->type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(((const uint16_t *)t->data)[idx]);
    return 1.0f;
}

static inline void lfm3_quantize_q8_K(const float * restrict x, block_q8_K_t * restrict y, uint32_t n_cols) {
    if (lfm3_is_forced_scalar()) {
        lfm2_quantize_row_q8_K_scalar(x, y, n_cols);
        return;
    }
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    kt->quantize_q8_K(x, y, n_cols);
}

static inline bool lfm3_type_supports_1x4(uint32_t type) {
    return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ||
            type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 ||
            type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS ||
            type == GGML_TYPE_F32);
}

static inline void lfm3_store_f32_to_f16(uint16_t * restrict dst, const float * restrict src, uint32_t n) {
    uint32_t i = 0;
#if LFM2_HAS_AVX2
    if (lfm3_use_avx2()) {
        for (; i + 8 <= n; i += 8) {
            const __m256 v = _mm256_loadu_ps(src + i);
            const __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i *)(dst + i), h);
        }
    }
#elif defined(__aarch64__)
    if (lfm3_use_neon()) {
        for (; i + 8 <= n; i += 8) {
            neon_store_fp32_to_fp16x8(dst + i, vld1q_f32(src + i), vld1q_f32(src + i + 4));
        }
    }
#endif
    for (; i < n; ++i) {
        dst[i] = lfm2_fp32_to_fp16(src[i]);
    }
}

/* ============================================================================
 * 1. Planar 7-Tap ShortConv Weight Cache & Circular-Ring Kernel
 * ============================================================================ */
typedef struct {
    float *w[LFM3_SHORTCONV_TAPS]; /* w[0..6] each [D] 64-byte aligned (w[0]=t-6 .. w[6]=t) */
} LFM3PlanarConv7;

static inline bool lfm3_planar_conv7_init(LFM3PlanarConv7 *pw, const GGUFTensor *conv_t, uint32_t D, uint32_t l_cache) {
    if (!pw || !conv_t || !conv_t->data) return false;
    memset(pw, 0, sizeof(*pw));
    const uint32_t eff_taps = (l_cache > 0 && l_cache <= LFM3_SHORTCONV_TAPS) ? l_cache : LFM3_SHORTCONV_TAPS;
    const uint32_t tap_off  = LFM3_SHORTCONV_TAPS - eff_taps;
    const bool stride_is_lcache = (conv_t->ne[0] == eff_taps);

    for (uint32_t k = 0; k < LFM3_SHORTCONV_TAPS; ++k) {
        pw->w[k] = (float *)lfm3_aligned_calloc_64((size_t)D * sizeof(float));
        if (!pw->w[k]) {
            for (uint32_t j = 0; j < k; ++j) free(pw->w[j]);
            memset(pw, 0, sizeof(*pw));
            return false;
        }
    }

    for (uint32_t k = 0; k < eff_taps; ++k) {
        float *dst = pw->w[tap_off + k];
        for (uint32_t d = 0; d < D; ++d) {
            const size_t src_idx = stride_is_lcache ? ((size_t)d * eff_taps + k) : ((size_t)k * D + d);
            dst[d] = lfm3_read_float_elem(conv_t, src_idx);
        }
    }
    return true;
}

static inline void lfm3_planar_conv7_free(LFM3PlanarConv7 *pw) {
    if (!pw) return;
    for (uint32_t k = 0; k < LFM3_SHORTCONV_TAPS; ++k) {
        free(pw->w[k]);
        pw->w[k] = NULL;
    }
}

/* Reorder 6 circular-ring slots chronologically [t-6..t-1] -> dst[6*D] for disk prompt cache */
static inline void lfm3_shortconv7_linearize_ring(const float * restrict ring,
                                                  uint32_t head,
                                                  uint32_t hist_len,
                                                  uint32_t D,
                                                  float * restrict dst_linear) {
    for (uint32_t s = 0; s < hist_len; ++s) {
        const uint32_t slot = (head + s) % hist_len;
        memcpy(dst_linear + (size_t)s * D, ring + (size_t)slot * D, (size_t)D * sizeof(float));
    }
}

static inline void lfm3_shortconv7_load_linear(float * restrict ring,
                                               uint32_t * restrict out_head,
                                               uint32_t hist_len,
                                               uint32_t D,
                                               const float * restrict src_linear) {
    memcpy(ring, src_linear, (size_t)hist_len * D * sizeof(float));
    *out_head = 0;
}

#if LFM2_HAS_AVX2
LFM2_TARGET_AVX2
static inline void lfm3_shortconv7_gate_conv_quant_avx2(float * restrict conv_out,
                                                        block_q8_K_t * restrict conv_out_q8,
                                                        const float * restrict bcx,
                                                        float * restrict conv_ring,
                                                        uint32_t * restrict inout_head,
                                                        const LFM3PlanarConv7 * restrict pw,
                                                        uint32_t D) {
    const uint32_t head = *inout_head;
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2U * D;

    float * restrict r0 = conv_ring + (size_t)((head + 0U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r1 = conv_ring + (size_t)((head + 1U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r2 = conv_ring + (size_t)((head + 2U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r3 = conv_ring + (size_t)((head + 3U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r4 = conv_ring + (size_t)((head + 4U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r5 = conv_ring + (size_t)((head + 5U) % LFM3_SHORTCONV_HIST) * D;

    const float * restrict w0 = pw->w[0], * restrict w1 = pw->w[1], * restrict w2 = pw->w[2];
    const float * restrict w3 = pw->w[3], * restrict w4 = pw->w[4], * restrict w5 = pw->w[5];
    const float * restrict w6 = pw->w[6];

    const uint32_t nb = D / 256U;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256U;
        for (uint32_t i = 0; i < 256U; i += 8U) {
            const uint32_t d = base + i;
            const __m256 vB  = _mm256_loadu_ps(B + d);
            const __m256 vC  = _mm256_loadu_ps(C + d);
            const __m256 vx  = _mm256_loadu_ps(x_tilde + d);
            const __m256 u_t = _mm256_mul_ps(vB, vx);

            __m256 acc0 = _mm256_mul_ps(_mm256_load_ps(w0 + d), _mm256_loadu_ps(r0 + d));
            __m256 acc1 = _mm256_mul_ps(_mm256_load_ps(w1 + d), _mm256_loadu_ps(r1 + d));
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(w2 + d), _mm256_loadu_ps(r2 + d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(w3 + d), _mm256_loadu_ps(r3 + d), acc1);
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(w4 + d), _mm256_loadu_ps(r4 + d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(w5 + d), _mm256_loadu_ps(r5 + d), acc1);
            __m256 acc = _mm256_fmadd_ps(_mm256_load_ps(w6 + d), u_t, _mm256_add_ps(acc0, acc1));

            _mm256_storeu_ps(r0 + d, u_t); /* Overwrite oldest slot t-6 with u_t */
            _mm256_storeu_ps(conv_out + d, _mm256_mul_ps(vC, acc));
        }
        if (conv_out_q8 != NULL) {
            lfm3_quantize_q8_K(conv_out + base, &conv_out_q8[b], 256U);
        }
    }

    for (uint32_t d = nb * 256U; d < D; ++d) {
        const float u_t = B[d] * x_tilde[d];
        const float acc = w0[d] * r0[d] + w1[d] * r1[d] + w2[d] * r2[d] +
                          w3[d] * r3[d] + w4[d] * r4[d] + w5[d] * r5[d] + w6[d] * u_t;
        r0[d] = u_t;
        conv_out[d] = C[d] * acc;
    }

    *inout_head = (head + 1U == LFM3_SHORTCONV_HIST) ? 0U : (head + 1U);
}
#endif

#if defined(__aarch64__)
LFM2_TARGET_ARM_FP16
static inline void lfm3_shortconv7_gate_conv_quant_neon(float * restrict conv_out,
                                                        block_q8_K_t * restrict conv_out_q8,
                                                        const float * restrict bcx,
                                                        float * restrict conv_ring,
                                                        uint32_t * restrict inout_head,
                                                        const LFM3PlanarConv7 * restrict pw,
                                                        uint32_t D) {
    const uint32_t head = *inout_head;
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2U * D;

    float * restrict r0 = conv_ring + (size_t)((head + 0U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r1 = conv_ring + (size_t)((head + 1U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r2 = conv_ring + (size_t)((head + 2U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r3 = conv_ring + (size_t)((head + 3U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r4 = conv_ring + (size_t)((head + 4U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r5 = conv_ring + (size_t)((head + 5U) % LFM3_SHORTCONV_HIST) * D;

    const float * restrict w0 = pw->w[0], * restrict w1 = pw->w[1], * restrict w2 = pw->w[2];
    const float * restrict w3 = pw->w[3], * restrict w4 = pw->w[4], * restrict w5 = pw->w[5];
    const float * restrict w6 = pw->w[6];

    const uint32_t nb = D / 256U;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t base = b * 256U;
        for (uint32_t i = 0; i < 256U; i += 4U) {
            const uint32_t d = base + i;
            const float32x4_t u_t = vmulq_f32(vld1q_f32(B + d), vld1q_f32(x_tilde + d));
            float32x4_t acc0 = vmulq_f32(vld1q_f32(w0 + d), vld1q_f32(r0 + d));
            float32x4_t acc1 = vmulq_f32(vld1q_f32(w1 + d), vld1q_f32(r1 + d));
            acc0 = vfmaq_f32(acc0, vld1q_f32(w2 + d), vld1q_f32(r2 + d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(w3 + d), vld1q_f32(r3 + d));
            acc0 = vfmaq_f32(acc0, vld1q_f32(w4 + d), vld1q_f32(r4 + d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(w5 + d), vld1q_f32(r5 + d));
            float32x4_t acc = vfmaq_f32(vaddq_f32(acc0, acc1), vld1q_f32(w6 + d), u_t);
            vst1q_f32(r0 + d, u_t);
            vst1q_f32(conv_out + d, vmulq_f32(vld1q_f32(C + d), acc));
        }
        if (conv_out_q8 != NULL) {
            lfm3_quantize_q8_K(conv_out + base, &conv_out_q8[b], 256U);
        }
    }

    for (uint32_t d = nb * 256U; d < D; ++d) {
        const float u_t = B[d] * x_tilde[d];
        const float acc = w0[d] * r0[d] + w1[d] * r1[d] + w2[d] * r2[d] +
                          w3[d] * r3[d] + w4[d] * r4[d] + w5[d] * r5[d] + w6[d] * u_t;
        r0[d] = u_t;
        conv_out[d] = C[d] * acc;
    }

    *inout_head = (head + 1U == LFM3_SHORTCONV_HIST) ? 0U : (head + 1U);
}
#endif

static inline void lfm3_shortconv7_gate_conv_quant_scalar(float * restrict conv_out,
                                                          block_q8_K_t * restrict conv_out_q8,
                                                          const float * restrict bcx,
                                                          float * restrict conv_ring,
                                                          uint32_t * restrict inout_head,
                                                          const LFM3PlanarConv7 * restrict pw,
                                                          uint32_t D) {
    const uint32_t head = *inout_head;
    const float * restrict B       = bcx;
    const float * restrict C       = bcx + D;
    const float * restrict x_tilde = bcx + 2U * D;

    float * restrict r0 = conv_ring + (size_t)((head + 0U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r1 = conv_ring + (size_t)((head + 1U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r2 = conv_ring + (size_t)((head + 2U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r3 = conv_ring + (size_t)((head + 3U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r4 = conv_ring + (size_t)((head + 4U) % LFM3_SHORTCONV_HIST) * D;
    float * restrict r5 = conv_ring + (size_t)((head + 5U) % LFM3_SHORTCONV_HIST) * D;

    const float * restrict w0 = pw->w[0], * restrict w1 = pw->w[1], * restrict w2 = pw->w[2];
    const float * restrict w3 = pw->w[3], * restrict w4 = pw->w[4], * restrict w5 = pw->w[5];
    const float * restrict w6 = pw->w[6];

    for (uint32_t d = 0; d < D; ++d) {
        const float u_t = B[d] * x_tilde[d];
        const float acc = w0[d] * r0[d] + w1[d] * r1[d] + w2[d] * r2[d] +
                          w3[d] * r3[d] + w4[d] * r4[d] + w5[d] * r5[d] + w6[d] * u_t;
        r0[d] = u_t;
        conv_out[d] = C[d] * acc;
    }

    if (conv_out_q8 != NULL && (D % 256U) == 0U) {
        const uint32_t nb = D / 256U;
        for (uint32_t b = 0; b < nb; ++b) {
            lfm2_quantize_row_q8_K_scalar(conv_out + b * 256U, &conv_out_q8[b], 256U);
        }
    }

    *inout_head = (head + 1U == LFM3_SHORTCONV_HIST) ? 0U : (head + 1U);
}

static inline void lfm3_shortconv7_gate_conv_quant_simd(float * restrict conv_out,
                                                        block_q8_K_t * restrict conv_out_q8,
                                                        const float * restrict bcx,
                                                        float * restrict conv_ring,
                                                        uint32_t * restrict inout_head,
                                                        const LFM3PlanarConv7 * restrict pw,
                                                        uint32_t D) {
#if LFM2_HAS_AVX2
    if (lfm3_use_avx2()) {
        lfm3_shortconv7_gate_conv_quant_avx2(conv_out, conv_out_q8, bcx, conv_ring, inout_head, pw, D);
        return;
    }
#elif defined(__aarch64__)
    if (lfm3_use_neon()) {
        lfm3_shortconv7_gate_conv_quant_neon(conv_out, conv_out_q8, bcx, conv_ring, inout_head, pw, D);
        return;
    }
#endif
    lfm3_shortconv7_gate_conv_quant_scalar(conv_out, conv_out_q8, bcx, conv_ring, inout_head, pw, D);
}

/* ============================================================================
 * 2. LFM3 Dual RoPE LUT Builder (Supports Partial RoPE via rope_freqs.weight)
 * ============================================================================ */
static inline bool lfm3_rope_lut_init(LFM2RoPELUT *lut,
                                      uint32_t max_seq_len,
                                      uint32_t head_dim,
                                      uint32_t rope_dim,
                                      float freq_base,
                                      const GGUFTensor *rope_freqs) {
    if (!lut || max_seq_len == 0 || head_dim == 0 || (head_dim % 2U) != 0U) return false;
    const uint32_t half_dim = head_dim / 2U;
    lut->max_seq_len = max_seq_len;
    lut->half_dim    = half_dim;

    lut->cos_lut = (float *)lfm3_aligned_calloc_64((size_t)max_seq_len * half_dim * sizeof(float));
    lut->sin_lut = (float *)lfm3_aligned_calloc_64((size_t)max_seq_len * half_dim * sizeof(float));
    if (!lut->cos_lut || !lut->sin_lut) {
        free(lut->cos_lut);
        free(lut->sin_lut);
        memset(lut, 0, sizeof(*lut));
        return false;
    }

    const uint32_t rot_half = (rope_dim > 0 && rope_dim <= head_dim) ? (rope_dim / 2U) : half_dim;

    for (uint32_t i = 0; i < half_dim; ++i) {
        double freq = 0.0;
        if (i < rot_half) {
            freq = pow((double)freq_base, -((double)(2U * i) / (double)head_dim));
            if (rope_freqs != NULL && rope_freqs->data != NULL && i < rope_freqs->ne[0]) {
                const double scale_factor = (double)lfm3_read_float_elem(rope_freqs, i);
                if (scale_factor >= 1e10 || scale_factor <= 0.0) {
                    freq = 0.0; /* Zero frequency => identity rotation (partial RoPE) */
                } else {
                    freq /= scale_factor;
                }
            }
        }
        for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
            if (freq == 0.0) {
                lut->cos_lut[(size_t)pos * half_dim + i] = 1.0f;
                lut->sin_lut[(size_t)pos * half_dim + i] = 0.0f;
            } else {
                const double angle = (double)pos * freq;
                lut->cos_lut[(size_t)pos * half_dim + i] = (float)cos(angle);
                lut->sin_lut[(size_t)pos * half_dim + i] = (float)sin(angle);
            }
        }
    }
    return true;
}

/* ============================================================================
 * 3. Per-Head QK-Norm + NeoX Split RoPE (head_dim = 128 Fast + Generic)
 * ============================================================================ */
static inline float lfm3_hsum_ps_avx2_fallback(const float *p, uint32_t n) {
    float s = 0.0f;
    for (uint32_t i = 0; i < n; ++i) s += p[i];
    return s;
}

#if LFM2_HAS_AVX2
LFM2_TARGET_AVX2
static inline void lfm3_qk_norm_rope_128_avx2(float * restrict vec_f32,
                                               uint16_t * restrict dst_f16,
                                               uint32_t n_heads,
                                               const GGUFTensor * restrict norm_w,
                                               float eps,
                                               const LFM2RoPELUT * restrict lut,
                                               uint32_t pos) {
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * 64U;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * 64U;
    const float    *wf32 = (norm_w && norm_w->type == GGML_TYPE_F32) ? (const float *)norm_w->data : NULL;
    const uint16_t *wf16 = (norm_w && norm_w->type != GGML_TYPE_F32) ? (const uint16_t *)norm_w->data : NULL;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp    = vec_f32 + (size_t)h * 128U;
        uint16_t * restrict kp = dst_f16 ? (dst_f16 + (size_t)h * 128U) : NULL;

        float irms = 1.0f;
        if (norm_w != NULL && norm_w->data != NULL) {
            __m256 ss0 = _mm256_setzero_ps(), ss1 = _mm256_setzero_ps();
            for (uint32_t i = 0; i < 128U; i += 16U) {
                const __m256 v0 = _mm256_loadu_ps(hp + i + 0U);
                const __m256 v1 = _mm256_loadu_ps(hp + i + 8U);
                ss0 = _mm256_fmadd_ps(v0, v0, ss0);
                ss1 = _mm256_fmadd_ps(v1, v1, ss1);
            }
            const __m256 ssum = _mm256_add_ps(ss0, ss1);
            const __m128 hi   = _mm256_extractf128_ps(ssum, 1);
            const __m128 lo   = _mm256_castps256_ps128(ssum);
            __m128 s4         = _mm_add_ps(lo, hi);
            s4 = _mm_hadd_ps(s4, s4);
            s4 = _mm_hadd_ps(s4, s4);
            const float ss = _mm_cvtss_f32(s4);
            irms = 1.0f / sqrtf(ss * (1.0f / 128.0f) + eps);
        }
        const __m256 virms = _mm256_set1_ps(irms);

        for (uint32_t i = 0; i < 64U; i += 8U) {
            __m256 x0 = _mm256_loadu_ps(hp + i);
            __m256 x1 = _mm256_loadu_ps(hp + 64U + i);
            if (wf32) {
                x0 = _mm256_mul_ps(_mm256_mul_ps(x0, virms), _mm256_loadu_ps(wf32 + i));
                x1 = _mm256_mul_ps(_mm256_mul_ps(x1, virms), _mm256_loadu_ps(wf32 + 64U + i));
            } else if (wf16) {
                __m256 wv0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wf16 + i)));
                __m256 wv1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wf16 + 64U + i)));
                x0 = _mm256_mul_ps(_mm256_mul_ps(x0, virms), wv0);
                x1 = _mm256_mul_ps(_mm256_mul_ps(x1, virms), wv1);
            }
            const __m256 vc = _mm256_load_ps(cos_p + i);
            const __m256 vs = _mm256_load_ps(sin_p + i);
            const __m256 r0 = _mm256_fmsub_ps(x0, vc, _mm256_mul_ps(x1, vs));
            const __m256 r1 = _mm256_fmadd_ps(x1, vc, _mm256_mul_ps(x0, vs));

            if (kp != NULL) {
                _mm_storeu_si128((__m128i *)(kp + i),       _mm256_cvtps_ph(r0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm_storeu_si128((__m128i *)(kp + 64U + i), _mm256_cvtps_ph(r1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            } else {
                _mm256_storeu_ps(hp + i,       r0);
                _mm256_storeu_ps(hp + 64U + i, r1);
            }
        }
    }
}
#endif

#if defined(__aarch64__)
LFM2_TARGET_ARM_FP16
static inline void lfm3_qk_norm_rope_128_neon(float * restrict vec_f32,
                                               uint16_t * restrict dst_f16,
                                               uint32_t n_heads,
                                               const GGUFTensor * restrict norm_w,
                                               float eps,
                                               const LFM2RoPELUT * restrict lut,
                                               uint32_t pos) {
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * 64U;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * 64U;
    const float * restrict nw    = (norm_w && norm_w->type == GGML_TYPE_F32) ? (const float *)norm_w->data : NULL;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp    = vec_f32 + (size_t)h * 128U;
        uint16_t * restrict kp = dst_f16 ? (dst_f16 + (size_t)h * 128U) : NULL;

        float irms = 1.0f;
        if (norm_w != NULL && norm_w->data != NULL) {
            float32x4_t ss0 = vdupq_n_f32(0.0f), ss1 = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < 128U; i += 8U) {
                const float32x4_t v0 = vld1q_f32(hp + i + 0U);
                const float32x4_t v1 = vld1q_f32(hp + i + 4U);
                ss0 = vfmaq_f32(ss0, v0, v0);
                ss1 = vfmaq_f32(ss1, v1, v1);
            }
            const float ss = vaddvq_f32(vaddq_f32(ss0, ss1));
            irms = 1.0f / sqrtf(ss * (1.0f / 128.0f) + eps);
        }
        const float32x4_t virms = vdupq_n_f32(irms);

        for (uint32_t i = 0; i < 64U; i += 8U) {
            float32x4_t x_lo0 = vld1q_f32(hp + i + 0U),  x_lo1 = vld1q_f32(hp + i + 4U);
            float32x4_t x_hi0 = vld1q_f32(hp + i + 64U), x_hi1 = vld1q_f32(hp + i + 68U);
            if (nw != NULL) {
                x_lo0 = vmulq_f32(vmulq_f32(x_lo0, virms), vld1q_f32(nw + i + 0U));
                x_lo1 = vmulq_f32(vmulq_f32(x_lo1, virms), vld1q_f32(nw + i + 4U));
                x_hi0 = vmulq_f32(vmulq_f32(x_hi0, virms), vld1q_f32(nw + i + 64U));
                x_hi1 = vmulq_f32(vmulq_f32(x_hi1, virms), vld1q_f32(nw + i + 68U));
            } else if (norm_w != NULL && norm_w->data != NULL) {
                float tmp[8], tmp2[8];
                for (uint32_t k = 0; k < 8U; ++k) {
                    tmp[k]  = lfm3_read_float_elem(norm_w, i + k);
                    tmp2[k] = lfm3_read_float_elem(norm_w, 64U + i + k);
                }
                x_lo0 = vmulq_f32(vmulq_f32(x_lo0, virms), vld1q_f32(tmp));
                x_lo1 = vmulq_f32(vmulq_f32(x_lo1, virms), vld1q_f32(tmp + 4));
                x_hi0 = vmulq_f32(vmulq_f32(x_hi0, virms), vld1q_f32(tmp2));
                x_hi1 = vmulq_f32(vmulq_f32(x_hi1, virms), vld1q_f32(tmp2 + 4));
            }
            const float32x4_t c0 = vld1q_f32(cos_p + i + 0U), c1 = vld1q_f32(cos_p + i + 4U);
            const float32x4_t s0 = vld1q_f32(sin_p + i + 0U), s1 = vld1q_f32(sin_p + i + 4U);
            const float32x4_t r_lo0 = vfmsq_f32(vmulq_f32(x_lo0, c0), x_hi0, s0);
            const float32x4_t r_lo1 = vfmsq_f32(vmulq_f32(x_lo1, c1), x_hi1, s1);
            const float32x4_t r_hi0 = vfmaq_f32(vmulq_f32(x_hi0, c0), x_lo0, s0);
            const float32x4_t r_hi1 = vfmaq_f32(vmulq_f32(x_hi1, c1), x_lo1, s1);

            if (kp != NULL) {
                neon_store_fp32_to_fp16x8(kp + i,       r_lo0, r_lo1);
                neon_store_fp32_to_fp16x8(kp + i + 64U, r_hi0, r_hi1);
            } else {
                vst1q_f32(hp + i + 0U,  r_lo0);
                vst1q_f32(hp + i + 4U,  r_lo1);
                vst1q_f32(hp + i + 64U, r_hi0);
                vst1q_f32(hp + i + 68U, r_hi1);
            }
        }
    }
}
#endif

static inline void lfm3_qk_norm_rope_scalar(float * restrict vec_f32,
                                            uint16_t * restrict dst_f16,
                                            uint32_t n_heads,
                                            uint32_t head_dim,
                                            const GGUFTensor * restrict norm_w,
                                            float eps,
                                            const LFM2RoPELUT * restrict lut,
                                            uint32_t pos) {
    const uint32_t half_dim = head_dim / 2U;
    const float * restrict cos_p = lut->cos_lut + (size_t)pos * half_dim;
    const float * restrict sin_p = lut->sin_lut + (size_t)pos * half_dim;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float * restrict hp    = vec_f32 + (size_t)h * head_dim;
        uint16_t * restrict kp = dst_f16 ? (dst_f16 + (size_t)h * head_dim) : NULL;

        float irms = 1.0f;
        if (norm_w != NULL && norm_w->data != NULL) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < head_dim; ++i) ss += hp[i] * hp[i];
            irms = 1.0f / sqrtf(ss / (float)head_dim + eps);
        }

        for (uint32_t i = 0; i < half_dim; ++i) {
            float x0 = hp[i];
            float x1 = hp[i + half_dim];
            if (norm_w != NULL && norm_w->data != NULL) {
                x0 = (x0 * irms) * lfm3_read_float_elem(norm_w, i);
                x1 = (x1 * irms) * lfm3_read_float_elem(norm_w, i + half_dim);
            }
            const float c = cos_p[i];
            const float s = sin_p[i];
            const float r0 = x0 * c - x1 * s;
            const float r1 = x1 * c + x0 * s;
            if (kp != NULL) {
                kp[i]            = lfm2_fp32_to_fp16(r0);
                kp[i + half_dim] = lfm2_fp32_to_fp16(r1);
            } else {
                hp[i]            = r0;
                hp[i + half_dim] = r1;
            }
        }
    }
}

static inline void lfm3_qk_norm_rope_simd(float * restrict vec_f32,
                                          uint16_t * restrict dst_f16,
                                          uint32_t n_heads,
                                          uint32_t head_dim,
                                          const GGUFTensor * restrict norm_w,
                                          float eps,
                                          const LFM2RoPELUT * restrict lut,
                                          uint32_t pos) {
    if (head_dim == 128U) {
#if LFM2_HAS_AVX2
        if (lfm3_use_avx2()) {
            lfm3_qk_norm_rope_128_avx2(vec_f32, dst_f16, n_heads, norm_w, eps, lut, pos);
            return;
        }
#elif defined(__aarch64__)
        if (lfm3_use_neon()) {
            lfm3_qk_norm_rope_128_neon(vec_f32, dst_f16, n_heads, norm_w, eps, lut, pos);
            return;
        }
#endif
    }
    lfm3_qk_norm_rope_scalar(vec_f32, dst_f16, n_heads, head_dim, norm_w, eps, lut, pos);
}

/* ============================================================================
 * 4. Online Softmax FlashAttention (128-Dim Paired KV Reuse + Gate + SWA Ring)
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool *pool;
    const float    *q_batch;     /* [B, n_head_q, head_dim] */
    const float    *gate_sig;    /* [B, n_head_q] or NULL */
    const uint16_t *k_cache_f16; /* [cache_len, kv_dim] */
    const uint16_t *v_cache_f16; /* [cache_len, kv_dim] */
    float          *xb_out;      /* [B, n_head_q * head_dim] */
    block_q8_K_t   *xb_q8_out;   /* Optional [n_head_q * head_dim / 256] when B==1 */
    uint32_t        B;
    uint32_t        pos_0;
    uint32_t        window_len;  /* 0 = global linear; >0 = sliding window size */
    uint32_t        swa_cap;     /* capacity of k_cache_f16 in tokens */
    uint32_t        n_head_q;
    uint32_t        kv_mul;
    uint32_t        kv_dim;
    uint32_t        head_dim;
    uint32_t        n_groups;
    float           scale;
} LFM3OnlineGQACtx;

#if LFM2_HAS_AVX2
LFM2_TARGET_AVX2
static inline float lfm3_dot128_f32_f16_avx2(const float * restrict q, const uint16_t * restrict k16) {
    __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
    for (uint32_t i = 0; i < 128U; i += 16U) {
        const __m256 fk0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k16 + i + 0U)));
        const __m256 fk1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k16 + i + 8U)));
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 0U), fk0, d0);
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8U), fk1, d1);
    }
    const __m256 sum = _mm256_add_ps(d0, d1);
    const __m128 hi  = _mm256_extractf128_ps(sum, 1);
    const __m128 lo  = _mm256_castps256_ps128(sum);
    __m128 s4        = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

/* Stream one contiguous KV cache segment [seg_len] into TWO query heads sharing kv_h */
LFM2_TARGET_AVX2
static inline void lfm3_gqa128_dual_head_seg_avx2(const float * restrict q0,
                                                  const float * restrict q1,
                                                  const uint16_t *k_ptr,
                                                  const uint16_t *v_ptr,
                                                  size_t stride,
                                                  uint32_t seg_len,
                                                  float scale,
                                                  float * restrict vo0,
                                                  float * restrict vo1,
                                                  float * restrict m0_p,
                                                  float * restrict l0_p,
                                                  float * restrict m1_p,
                                                  float * restrict l1_p) {
    float m0 = *m0_p, l0 = *l0_p;
    float m1 = *m1_p, l1 = *l1_p;

    for (uint32_t t = 0; t < seg_len; ++t) {
        if (t + 2U < seg_len) {
            _mm_prefetch((const char *)(k_ptr + 2U * stride), _MM_HINT_T0);
            _mm_prefetch((const char *)(v_ptr + 2U * stride), _MM_HINT_T0);
        }

        __m256 d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
        __m256 d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
        for (uint32_t i = 0; i < 128U; i += 16U) {
            const __m256 fk0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + i + 0U)));
            const __m256 fk1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_ptr + i + 8U)));
            d00 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 0U), fk0, d00);
            d01 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 8U), fk1, d01);
            d10 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 0U), fk0, d10);
            d11 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 8U), fk1, d11);
        }

        const __m256 sum0 = _mm256_add_ps(d00, d01);
        const __m128 s04  = _mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(sum0), _mm256_extractf128_ps(sum0, 1)),
                                        _mm_add_ps(_mm256_castps256_ps128(sum0), _mm256_extractf128_ps(sum0, 1)));
        const float score0 = _mm_cvtss_f32(_mm_hadd_ps(s04, s04)) * scale;

        const __m256 sum1 = _mm256_add_ps(d10, d11);
        const __m128 s14  = _mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1)),
                                        _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1)));
        const float score1 = _mm_cvtss_f32(_mm_hadd_ps(s14, s14)) * scale;

        float so0, sv0, so1, sv1;
        if (score0 > m0) {
            so0 = (m0 <= -1e29f) ? 0.0f : expf(m0 - score0);
            sv0 = 1.0f;
            l0  = l0 * so0 + 1.0f;
            m0  = score0;
        } else {
            so0 = 1.0f;
            sv0 = expf(score0 - m0);
            l0 += sv0;
        }

        if (score1 > m1) {
            so1 = (m1 <= -1e29f) ? 0.0f : expf(m1 - score1);
            sv1 = 1.0f;
            l1  = l1 * so1 + 1.0f;
            m1  = score1;
        } else {
            so1 = 1.0f;
            sv1 = expf(score1 - m1);
            l1 += sv1;
        }

        const __m256 v_so0 = _mm256_set1_ps(so0), v_sv0 = _mm256_set1_ps(sv0);
        const __m256 v_so1 = _mm256_set1_ps(so1), v_sv1 = _mm256_set1_ps(sv1);

        for (uint32_t i = 0; i < 128U; i += 16U) {
            const __m256 fv0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + i + 0U)));
            const __m256 fv1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + i + 8U)));

            __m256 o00 = _mm256_load_ps(vo0 + i + 0U);
            __m256 o01 = _mm256_load_ps(vo0 + i + 8U);
            o00 = _mm256_fmadd_ps(o00, v_so0, _mm256_mul_ps(fv0, v_sv0));
            o01 = _mm256_fmadd_ps(o01, v_so0, _mm256_mul_ps(fv1, v_sv0));
            _mm256_store_ps(vo0 + i + 0U, o00);
            _mm256_store_ps(vo0 + i + 8U, o01);

            __m256 o10 = _mm256_load_ps(vo1 + i + 0U);
            __m256 o11 = _mm256_load_ps(vo1 + i + 8U);
            o10 = _mm256_fmadd_ps(o10, v_so1, _mm256_mul_ps(fv0, v_sv1));
            o11 = _mm256_fmadd_ps(o11, v_so1, _mm256_mul_ps(fv1, v_sv1));
            _mm256_store_ps(vo1 + i + 0U, o10);
            _mm256_store_ps(vo1 + i + 8U, o11);
        }

        k_ptr += stride;
        v_ptr += stride;
    }

    *m0_p = m0; *l0_p = l0;
    *m1_p = m1; *l1_p = l1;
}

LFM2_TARGET_AVX2
static inline void lfm3_gqa128_single_head_seg_avx2(const float * restrict q_h,
                                                    const uint16_t *k_ptr,
                                                    const uint16_t *v_ptr,
                                                    size_t stride,
                                                    uint32_t seg_len,
                                                    float scale,
                                                    float * restrict vo,
                                                    float * restrict m_p,
                                                    float * restrict l_p) {
    float m_prev = *m_p, l_prev = *l_p;
    for (uint32_t t = 0; t < seg_len; ++t) {
        if (t + 2U < seg_len) {
            _mm_prefetch((const char *)(k_ptr + 2U * stride), _MM_HINT_T0);
            _mm_prefetch((const char *)(v_ptr + 2U * stride), _MM_HINT_T0);
        }
        const float score = lfm3_dot128_f32_f16_avx2(q_h, k_ptr) * scale;
        float so, sv;
        if (score > m_prev) {
            so = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
            sv = 1.0f;
            l_prev = l_prev * so + 1.0f;
            m_prev = score;
        } else {
            so = 1.0f;
            sv = expf(score - m_prev);
            l_prev += sv;
        }
        const __m256 v_so = _mm256_set1_ps(so), v_sv = _mm256_set1_ps(sv);
        for (uint32_t i = 0; i < 128U; i += 16U) {
            const __m256 fv0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + i + 0U)));
            const __m256 fv1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_ptr + i + 8U)));
            __m256 o0 = _mm256_load_ps(vo + i + 0U);
            __m256 o1 = _mm256_load_ps(vo + i + 8U);
            _mm256_store_ps(vo + i + 0U, _mm256_fmadd_ps(o0, v_so, _mm256_mul_ps(fv0, v_sv)));
            _mm256_store_ps(vo + i + 8U, _mm256_fmadd_ps(o1, v_so, _mm256_mul_ps(fv1, v_sv)));
        }
        k_ptr += stride;
        v_ptr += stride;
    }
    *m_p = m_prev; *l_p = l_prev;
}

static void lfm3_online_gqa_128_f16_worker_avx2(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    LFM3OnlineGQACtx *c = (LFM3OnlineGQACtx *)arg;
    const uint32_t total_items = c->B * c->n_groups;
    uint32_t item_start = 0, item_end = 0;
    const size_t stride = c->kv_dim;
    const size_t q_stride = (size_t)c->n_head_q * 128U;

    while (lfm2_threadpool_claim_chunk(c->pool, &item_start, &item_end)) {
        for (uint32_t idx = item_start; idx < item_end; ++idx) {
            const uint32_t b = idx / c->n_groups;
            const uint32_t g = idx % c->n_groups;
            const uint32_t abs_pos = c->pos_0 + b;
            const uint32_t h_base  = g * 2U;
            const uint32_t h_end   = (h_base + 2U <= c->n_head_q) ? (h_base + 2U) : c->n_head_q;

            const uint32_t t_start = (c->window_len > 0U && abs_pos + 1U > c->window_len)
                                   ? (abs_pos + 1U - c->window_len) : 0U;
            const uint32_t count = abs_pos - t_start + 1U;
            const uint32_t seg1_slot = (c->window_len > 0U) ? (t_start % c->swa_cap) : t_start;
            const uint32_t seg1_len  = (c->window_len > 0U && seg1_slot + count > c->swa_cap)
                                     ? (c->swa_cap - seg1_slot) : count;
            const uint32_t seg2_len  = count - seg1_len;

            const float *q_tok = c->q_batch + (size_t)b * q_stride;
            float       *o_tok = c->xb_out  + (size_t)b * q_stride;
            const float *g_tok = c->gate_sig ? (c->gate_sig + (size_t)b * c->n_head_q) : NULL;

            if (h_end - h_base == 2U && (h_base / c->kv_mul) == ((h_base + 1U) / c->kv_mul)) {
                const uint32_t kv_h = h_base / c->kv_mul;
                const size_t kv_off = (size_t)kv_h * 128U;
                float _Alignas(64) vo0[128] = {0.0f};
                float _Alignas(64) vo1[128] = {0.0f};
                float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;

                const float *q0 = q_tok + (size_t)h_base * 128U;
                const float *q1 = q_tok + (size_t)(h_base + 1U) * 128U;

                lfm3_gqa128_dual_head_seg_avx2(
                    q0, q1,
                    c->k_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    c->v_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    stride, seg1_len, c->scale, vo0, vo1, &m0, &l0, &m1, &l1);
                if (seg2_len > 0U) {
                    lfm3_gqa128_dual_head_seg_avx2(
                        q0, q1,
                        c->k_cache_f16 + kv_off,
                        c->v_cache_f16 + kv_off,
                        stride, seg2_len, c->scale, vo0, vo1, &m0, &l0, &m1, &l1);
                }

                const float g0 = g_tok ? g_tok[h_base + 0U] : 1.0f;
                const float g1 = g_tok ? g_tok[h_base + 1U] : 1.0f;
                const __m256 s0 = _mm256_set1_ps(l0 > 0.0f ? (g0 / l0) : 0.0f);
                const __m256 s1 = _mm256_set1_ps(l1 > 0.0f ? (g1 / l1) : 0.0f);
                float *dst0 = o_tok + (size_t)(h_base + 0U) * 128U;
                float *dst1 = o_tok + (size_t)(h_base + 1U) * 128U;
                for (uint32_t i = 0; i < 128U; i += 8U) {
                    _mm256_storeu_ps(dst0 + i, _mm256_mul_ps(_mm256_load_ps(vo0 + i), s0));
                    _mm256_storeu_ps(dst1 + i, _mm256_mul_ps(_mm256_load_ps(vo1 + i), s1));
                }
            } else {
                for (uint32_t h = h_base; h < h_end; ++h) {
                    const uint32_t kv_h = h / c->kv_mul;
                    const size_t kv_off = (size_t)kv_h * 128U;
                    float _Alignas(64) vo[128] = {0.0f};
                    float m_prev = -1e30f, l_prev = 0.0f;
                    const float *q_h = q_tok + (size_t)h * 128U;
                    lfm3_gqa128_single_head_seg_avx2(
                        q_h,
                        c->k_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                        c->v_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                        stride, seg1_len, c->scale, vo, &m_prev, &l_prev);
                    if (seg2_len > 0U) {
                        lfm3_gqa128_single_head_seg_avx2(
                            q_h,
                            c->k_cache_f16 + kv_off,
                            c->v_cache_f16 + kv_off,
                            stride, seg2_len, c->scale, vo, &m_prev, &l_prev);
                    }
                    const float gv = g_tok ? g_tok[h] : 1.0f;
                    const __m256 vs = _mm256_set1_ps(l_prev > 0.0f ? (gv / l_prev) : 0.0f);
                    float *dst = o_tok + (size_t)h * 128U;
                    for (uint32_t i = 0; i < 128U; i += 8U) {
                        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_load_ps(vo + i), vs));
                    }
                }
            }

            if (c->B == 1U && c->xb_q8_out != NULL && (h_end - h_base) == 2U) {
                lfm3_quantize_q8_K(o_tok + (size_t)h_base * 128U, &c->xb_q8_out[g], 256U);
            }
        }
        (void)total_items;
    }
}
#endif

#if defined(__aarch64__)
LFM2_TARGET_ARM_FP16
static inline void lfm3_gqa128_single_head_seg_neon(const float * restrict q_h,
                                                    const uint16_t *k_ptr,
                                                    const uint16_t *v_ptr,
                                                    size_t stride,
                                                    uint32_t seg_len,
                                                    float scale,
                                                    float * restrict vo,
                                                    float * restrict m_p,
                                                    float * restrict l_p) {
    float m_prev = *m_p, l_prev = *l_p;
    for (uint32_t t = 0; t < seg_len; ++t) {
        if (t + 2U < seg_len) {
            __builtin_prefetch(k_ptr + 2U * stride, 0, 0);
            __builtin_prefetch(v_ptr + 2U * stride, 0, 0);
        }
        float32x4_t d0 = vdupq_n_f32(0.0f), d1 = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < 128U; i += 8U) {
            float32x4_t klo, khi;
            neon_load_fp16x8_to_fp32(k_ptr + i, &klo, &khi);
            d0 = vfmaq_f32(d0, vld1q_f32(q_h + i + 0U), klo);
            d1 = vfmaq_f32(d1, vld1q_f32(q_h + i + 4U), khi);
        }
        const float score = vaddvq_f32(vaddq_f32(d0, d1)) * scale;
        float so, sv;
        if (score > m_prev) {
            so = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
            sv = 1.0f;
            l_prev = l_prev * so + 1.0f;
            m_prev = score;
        } else {
            so = 1.0f;
            sv = expf(score - m_prev);
            l_prev += sv;
        }
        for (uint32_t i = 0; i < 128U; i += 8U) {
            float32x4_t vlo, vhi;
            neon_load_fp16x8_to_fp32(v_ptr + i, &vlo, &vhi);
            float32x4_t o0 = vld1q_f32(vo + i + 0U);
            float32x4_t o1 = vld1q_f32(vo + i + 4U);
            vst1q_f32(vo + i + 0U, vfmaq_n_f32(vmulq_n_f32(o0, so), vlo, sv));
            vst1q_f32(vo + i + 4U, vfmaq_n_f32(vmulq_n_f32(o1, so), vhi, sv));
        }
        k_ptr += stride;
        v_ptr += stride;
    }
    *m_p = m_prev; *l_p = l_prev;
}

static void lfm3_online_gqa_128_f16_worker_neon(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    LFM3OnlineGQACtx *c = (LFM3OnlineGQACtx *)arg;
    uint32_t item_start = 0, item_end = 0;
    const size_t stride = c->kv_dim;
    const size_t q_stride = (size_t)c->n_head_q * 128U;

    while (lfm2_threadpool_claim_chunk(c->pool, &item_start, &item_end)) {
        for (uint32_t idx = item_start; idx < item_end; ++idx) {
            const uint32_t b = idx / c->n_groups;
            const uint32_t g = idx % c->n_groups;
            const uint32_t abs_pos = c->pos_0 + b;
            const uint32_t h_base  = g * 2U;
            const uint32_t h_end   = (h_base + 2U <= c->n_head_q) ? (h_base + 2U) : c->n_head_q;

            const uint32_t t_start = (c->window_len > 0U && abs_pos + 1U > c->window_len)
                                   ? (abs_pos + 1U - c->window_len) : 0U;
            const uint32_t count = abs_pos - t_start + 1U;
            const uint32_t seg1_slot = (c->window_len > 0U) ? (t_start % c->swa_cap) : t_start;
            const uint32_t seg1_len  = (c->window_len > 0U && seg1_slot + count > c->swa_cap)
                                     ? (c->swa_cap - seg1_slot) : count;
            const uint32_t seg2_len  = count - seg1_len;

            const float *q_tok = c->q_batch + (size_t)b * q_stride;
            float       *o_tok = c->xb_out  + (size_t)b * q_stride;
            const float *g_tok = c->gate_sig ? (c->gate_sig + (size_t)b * c->n_head_q) : NULL;

            for (uint32_t h = h_base; h < h_end; ++h) {
                const uint32_t kv_h = h / c->kv_mul;
                const size_t kv_off = (size_t)kv_h * 128U;
                float _Alignas(64) vo[128] = {0.0f};
                float m_prev = -1e30f, l_prev = 0.0f;
                const float *q_h = q_tok + (size_t)h * 128U;
                lfm3_gqa128_single_head_seg_neon(
                    q_h,
                    c->k_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    c->v_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    stride, seg1_len, c->scale, vo, &m_prev, &l_prev);
                if (seg2_len > 0U) {
                    lfm3_gqa128_single_head_seg_neon(
                        q_h,
                        c->k_cache_f16 + kv_off,
                        c->v_cache_f16 + kv_off,
                        stride, seg2_len, c->scale, vo, &m_prev, &l_prev);
                }
                const float gv = g_tok ? g_tok[h] : 1.0f;
                const float inv_l = (l_prev > 0.0f) ? (gv / l_prev) : 0.0f;
                float *dst = o_tok + (size_t)h * 128U;
                for (uint32_t i = 0; i < 128U; ++i) dst[i] = vo[i] * inv_l;
            }
            if (c->B == 1U && c->xb_q8_out != NULL && (h_end - h_base) == 2U) {
                lfm3_quantize_q8_K(o_tok + (size_t)h_base * 128U, &c->xb_q8_out[g], 256U);
            }
        }
    }
}
#endif

static inline void lfm3_gqa_head_seg_scalar(const float * restrict q_h,
                                            const uint16_t *k_ptr,
                                            const uint16_t *v_ptr,
                                            size_t stride,
                                            uint32_t seg_len,
                                            uint32_t head_dim,
                                            float scale,
                                            float * restrict vo,
                                            float * restrict m_p,
                                            float * restrict l_p) {
    float m_prev = *m_p, l_prev = *l_p;
    for (uint32_t t = 0; t < seg_len; ++t) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < head_dim; ++i) {
            dot += q_h[i] * lfm2_fp16_to_fp32(k_ptr[i]);
        }
        const float score = dot * scale;
        if (score > m_prev) {
            const float alpha = (m_prev <= -1e29f) ? 0.0f : expf(m_prev - score);
            l_prev = l_prev * alpha + 1.0f;
            m_prev = score;
            for (uint32_t i = 0; i < head_dim; ++i) {
                vo[i] = vo[i] * alpha + lfm2_fp16_to_fp32(v_ptr[i]);
            }
        } else {
            const float beta = expf(score - m_prev);
            l_prev += beta;
            for (uint32_t i = 0; i < head_dim; ++i) {
                vo[i] += beta * lfm2_fp16_to_fp32(v_ptr[i]);
            }
        }
        k_ptr += stride;
        v_ptr += stride;
    }
    *m_p = m_prev; *l_p = l_prev;
}

static void lfm3_online_gqa_f16_worker_scalar(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    LFM3OnlineGQACtx *c = (LFM3OnlineGQACtx *)arg;
    const uint32_t hd = c->head_dim;
    const uint32_t heads_per_group = (hd == 128U) ? 2U : 1U;
    uint32_t item_start = 0, item_end = 0;
    const size_t stride = c->kv_dim;
    const size_t q_stride = (size_t)c->n_head_q * hd;

    while (lfm2_threadpool_claim_chunk(c->pool, &item_start, &item_end)) {
        for (uint32_t idx = item_start; idx < item_end; ++idx) {
            const uint32_t b = idx / c->n_groups;
            const uint32_t g = idx % c->n_groups;
            const uint32_t abs_pos = c->pos_0 + b;
            const uint32_t h_base  = g * heads_per_group;
            const uint32_t h_end   = (h_base + heads_per_group <= c->n_head_q) ? (h_base + heads_per_group) : c->n_head_q;

            const uint32_t t_start = (c->window_len > 0U && abs_pos + 1U > c->window_len)
                                   ? (abs_pos + 1U - c->window_len) : 0U;
            const uint32_t count = abs_pos - t_start + 1U;
            const uint32_t seg1_slot = (c->window_len > 0U) ? (t_start % c->swa_cap) : t_start;
            const uint32_t seg1_len  = (c->window_len > 0U && seg1_slot + count > c->swa_cap)
                                     ? (c->swa_cap - seg1_slot) : count;
            const uint32_t seg2_len  = count - seg1_len;

            const float *q_tok = c->q_batch + (size_t)b * q_stride;
            float       *o_tok = c->xb_out  + (size_t)b * q_stride;
            const float *g_tok = c->gate_sig ? (c->gate_sig + (size_t)b * c->n_head_q) : NULL;

            for (uint32_t h = h_base; h < h_end; ++h) {
                const uint32_t kv_h = h / c->kv_mul;
                const size_t kv_off = (size_t)kv_h * hd;
                float vo[256] = {0.0f};
                float m_prev = -1e30f, l_prev = 0.0f;
                const float *q_h = q_tok + (size_t)h * hd;

                lfm3_gqa_head_seg_scalar(
                    q_h,
                    c->k_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    c->v_cache_f16 + (size_t)seg1_slot * stride + kv_off,
                    stride, seg1_len, hd, c->scale, vo, &m_prev, &l_prev);
                if (seg2_len > 0U) {
                    lfm3_gqa_head_seg_scalar(
                        q_h,
                        c->k_cache_f16 + kv_off,
                        c->v_cache_f16 + kv_off,
                        stride, seg2_len, hd, c->scale, vo, &m_prev, &l_prev);
                }
                const float gv = g_tok ? g_tok[h] : 1.0f;
                const float inv_l = (l_prev > 0.0f) ? (gv / l_prev) : 0.0f;
                float *dst = o_tok + (size_t)h * hd;
                for (uint32_t i = 0; i < hd; ++i) dst[i] = vo[i] * inv_l;
            }

            if (c->B == 1U && c->xb_q8_out != NULL && hd == 128U && (h_end - h_base) == 2U) {
                lfm2_quantize_row_q8_K_scalar(o_tok + (size_t)h_base * 128U, &c->xb_q8_out[g], 256U);
            }
        }
    }
}

static inline void lfm3_online_gqa_f16_dispatch(LFM2ThreadPool *pool,
                                                const float *q_batch,
                                                const float *gate_sig,
                                                const uint16_t *k_cache_f16,
                                                const uint16_t *v_cache_f16,
                                                float *xb_out,
                                                block_q8_K_t *xb_q8_out,
                                                uint32_t B,
                                                uint32_t pos_0,
                                                uint32_t window_len,
                                                uint32_t swa_cap,
                                                uint32_t n_head_q,
                                                uint32_t n_head_kv,
                                                uint32_t head_dim) {
    const uint32_t heads_per_group = (head_dim == 128U) ? 2U : 1U;
    const uint32_t n_groups = (n_head_q + heads_per_group - 1U) / heads_per_group;
    LFM3OnlineGQACtx ctx = {
        .pool        = pool,
        .q_batch     = q_batch,
        .gate_sig    = gate_sig,
        .k_cache_f16 = k_cache_f16,
        .v_cache_f16 = v_cache_f16,
        .xb_out      = xb_out,
        .xb_q8_out   = xb_q8_out,
        .B           = B,
        .pos_0       = pos_0,
        .window_len  = window_len,
        .swa_cap     = swa_cap,
        .n_head_q    = n_head_q,
        .kv_mul      = (n_head_kv > 0U) ? (n_head_q / n_head_kv) : 1U,
        .kv_dim      = n_head_kv * head_dim,
        .head_dim    = head_dim,
        .n_groups    = n_groups,
        .scale       = 1.0f / sqrtf((float)head_dim)
    };
    const uint32_t total_items = B * n_groups;

#if LFM2_HAS_AVX2
    if (head_dim == 128U && lfm3_use_avx2()) {
        lfm2_parallel_for(pool, total_items, 1U, lfm3_online_gqa_128_f16_worker_avx2, &ctx);
        return;
    }
#elif defined(__aarch64__)
    if (head_dim == 128U && lfm3_use_neon()) {
        lfm2_parallel_for(pool, total_items, 1U, lfm3_online_gqa_128_f16_worker_neon, &ctx);
        return;
    }
#endif
    lfm2_parallel_for(pool, total_items, 1U, lfm3_online_gqa_f16_worker_scalar, &ctx);
}

/* ============================================================================
 * 5. Sandwich Post-Norm + Residual Add, Fused Norm+Q8K, and MatMul Wrappers
 * ============================================================================ */
static inline void lfm3_post_norm_add_scalar(float * restrict x_res,
                                             const float * restrict h,
                                             const GGUFTensor * restrict w,
                                             uint32_t D,
                                             float eps) {
    if (!w || !w->data) {
        for (uint32_t d = 0; d < D; ++d) x_res[d] += h[d];
        return;
    }
    float ss = 0.0f;
    for (uint32_t d = 0; d < D; ++d) ss += h[d] * h[d];
    const float inv_rms = 1.0f / sqrtf(ss / (float)D + eps);
    for (uint32_t d = 0; d < D; ++d) {
        x_res[d] += (h[d] * inv_rms) * lfm3_read_float_elem(w, d);
    }
}

#if LFM2_HAS_AVX2
LFM2_TARGET_AVX2
static inline void lfm3_post_norm_add_avx2(float * restrict x_res,
                                           const float * restrict h,
                                           const GGUFTensor * restrict w,
                                           uint32_t D,
                                           float eps) {
    if (!w || !w->data) {
        uint32_t d = 0;
        for (; d + 8U <= D; d += 8U) {
            _mm256_storeu_ps(x_res + d, _mm256_add_ps(_mm256_loadu_ps(x_res + d), _mm256_loadu_ps(h + d)));
        }
        for (; d < D; ++d) x_res[d] += h[d];
        return;
    }

    __m256 ss0 = _mm256_setzero_ps(), ss1 = _mm256_setzero_ps();
    uint32_t d = 0;
    for (; d + 16U <= D; d += 16U) {
        const __m256 v0 = _mm256_loadu_ps(h + d + 0U);
        const __m256 v1 = _mm256_loadu_ps(h + d + 8U);
        ss0 = _mm256_fmadd_ps(v0, v0, ss0);
        ss1 = _mm256_fmadd_ps(v1, v1, ss1);
    }
    const __m256 sum = _mm256_add_ps(ss0, ss1);
    const __m128 lo  = _mm256_castps256_ps128(sum);
    const __m128 hi  = _mm256_extractf128_ps(sum, 1);
    __m128 s4        = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float ss = _mm_cvtss_f32(s4);
    for (; d < D; ++d) ss += h[d] * h[d];

    const float inv_rms = 1.0f / sqrtf(ss / (float)D + eps);
    const __m256 virms  = _mm256_set1_ps(inv_rms);

    if (w->type == GGML_TYPE_F32) {
        const float * restrict wf = (const float *)w->data;
        d = 0;
        for (; d + 16U <= D; d += 16U) {
            __m256 n0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(h + d + 0U), virms), _mm256_loadu_ps(wf + d + 0U));
            __m256 n1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(h + d + 8U), virms), _mm256_loadu_ps(wf + d + 8U));
            _mm256_storeu_ps(x_res + d + 0U, _mm256_add_ps(_mm256_loadu_ps(x_res + d + 0U), n0));
            _mm256_storeu_ps(x_res + d + 8U, _mm256_add_ps(_mm256_loadu_ps(x_res + d + 8U), n1));
        }
        for (; d < D; ++d) x_res[d] += (h[d] * inv_rms) * wf[d];
    } else if (w->type == GGML_TYPE_F16) {
        const uint16_t * restrict wh = (const uint16_t *)w->data;
        d = 0;
        for (; d + 8U <= D; d += 8U) {
            const __m256 wf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wh + d)));
            const __m256 nv = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(h + d), virms), wf);
            _mm256_storeu_ps(x_res + d, _mm256_add_ps(_mm256_loadu_ps(x_res + d), nv));
        }
        for (; d < D; ++d) x_res[d] += (h[d] * inv_rms) * lfm2_fp16_to_fp32(wh[d]);
    } else {
        for (uint32_t k = 0; k < D; ++k) x_res[k] += (h[k] * inv_rms) * lfm3_read_float_elem(w, k);
    }
}
#endif

#if defined(__aarch64__)
LFM2_TARGET_ARM_FP16
static inline void lfm3_post_norm_add_neon(float * restrict x_res,
                                           const float * restrict h,
                                           const GGUFTensor * restrict w,
                                           uint32_t D,
                                           float eps) {
    if (!w || !w->data) {
        uint32_t d = 0;
        for (; d + 4U <= D; d += 4U) {
            vst1q_f32(x_res + d, vaddq_f32(vld1q_f32(x_res + d), vld1q_f32(h + d)));
        }
        for (; d < D; ++d) x_res[d] += h[d];
        return;
    }
    float32x4_t ss0 = vdupq_n_f32(0.0f), ss1 = vdupq_n_f32(0.0f);
    uint32_t d = 0;
    for (; d + 8U <= D; d += 8U) {
        const float32x4_t v0 = vld1q_f32(h + d + 0U);
        const float32x4_t v1 = vld1q_f32(h + d + 4U);
        ss0 = vfmaq_f32(ss0, v0, v0);
        ss1 = vfmaq_f32(ss1, v1, v1);
    }
    float ss = vaddvq_f32(vaddq_f32(ss0, ss1));
    for (; d < D; ++d) ss += h[d] * h[d];
    const float inv_rms = 1.0f / sqrtf(ss / (float)D + eps);
    const float32x4_t virms = vdupq_n_f32(inv_rms);

    if (w->type == GGML_TYPE_F32) {
        const float * restrict wf = (const float *)w->data;
        d = 0;
        for (; d + 4U <= D; d += 4U) {
            const float32x4_t nv = vmulq_f32(vmulq_f32(vld1q_f32(h + d), virms), vld1q_f32(wf + d));
            vst1q_f32(x_res + d, vaddq_f32(vld1q_f32(x_res + d), nv));
        }
        for (; d < D; ++d) x_res[d] += (h[d] * inv_rms) * wf[d];
    } else {
        for (uint32_t k = 0; k < D; ++k) x_res[k] += (h[k] * inv_rms) * lfm3_read_float_elem(w, k);
    }
}
#endif

static inline void lfm3_post_norm_add_simd(float * restrict x_res,
                                           const float * restrict h,
                                           const GGUFTensor * restrict w,
                                           uint32_t D,
                                           float eps) {
#if LFM2_HAS_AVX2
    if (lfm3_use_avx2()) {
        lfm3_post_norm_add_avx2(x_res, h, w, D, eps);
        return;
    }
#elif defined(__aarch64__)
    if (lfm3_use_neon()) {
        lfm3_post_norm_add_neon(x_res, h, w, D, eps);
        return;
    }
#endif
    lfm3_post_norm_add_scalar(x_res, h, w, D, eps);
}

static inline void lfm3_post_norm_add_batch(float * restrict x_batch,
                                            const float * restrict h_batch,
                                            const GGUFTensor * restrict w,
                                            uint32_t B,
                                            uint32_t D,
                                            float eps) {
    for (uint32_t t = 0; t < B; ++t) {
        lfm3_post_norm_add_simd(x_batch + (size_t)t * D,
                                h_batch + (size_t)t * D,
                                w, D, eps);
    }
}

static inline void lfm3_rmsnorm_quant_scalar_safe(float * restrict xb,
                                                  block_q8_K_t * restrict xq,
                                                  const float * restrict x,
                                                  const GGUFTensor * restrict norm_w,
                                                  uint32_t D,
                                                  float eps) {
    if (norm_w) {
        lfm2_rmsnorm(xb, x, norm_w, D, eps);
    } else {
        memcpy(xb, x, (size_t)D * sizeof(float));
    }
    if (xq && (D % 256U) == 0U) {
        lfm3_quantize_q8_K(xb, xq, D);
    }
}

/* MatMul dispatchers that honor LFM2_FORCE_ISA=scalar AND handle F16/BF16 safely */
static inline void lfm3_matmul_1x4_or_vec(LFM2ThreadPool *pool,
                                          float * restrict out,
                                          const GGUFTensor * restrict W,
                                          const float * restrict x_f32,
                                          block_q8_K_t * restrict x_q8) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    const uint32_t n_rows = (uint32_t)W->ne[1];

    if ((n_cols % 256U) == 0U && lfm3_type_supports_1x4(W->type) &&
        !lfm3_is_forced_scalar() && lfm3_use_avx2()) {
        lfm2_matmul_1x4_prequant_mt(pool, out, W, x_f32, x_q8);
        return;
    }

    if ((n_cols % 256U) == 0U && lfm3_type_supports_1x4(W->type) &&
        W->type != GGML_TYPE_F32) {
        lfm2_matmul_batch_avx2_mt(pool, out, W, x_f32, 1U, x_q8);
        return;
    }

    /* Scalar fallback (or F16/BF16 / non-256 aligned cols): strictly C11 without AVX2 override */
    if (W->type == GGML_TYPE_F32) {
        const float *w_data = (const float *)W->data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            const float *row = w_data + (size_t)r * n_cols;
            float sum = 0.0f;
            for (uint32_t k = 0; k < n_cols; ++k) sum += row[k] * x_f32[k];
            out[r] = sum;
        }
        return;
    }
    if (W->type == GGML_TYPE_F16) {
        const uint16_t *w_data = (const uint16_t *)W->data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            const uint16_t *row = w_data + (size_t)r * n_cols;
            float sum = 0.0f;
            for (uint32_t k = 0; k < n_cols; ++k) sum += lfm2_fp16_to_fp32(row[k]) * x_f32[k];
            out[r] = sum;
        }
        return;
    }
    if (W->type == GGML_TYPE_BF16) {
        const uint16_t *w_data = (const uint16_t *)W->data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            const uint16_t *row = w_data + (size_t)r * n_cols;
            float sum = 0.0f;
            for (uint32_t k = 0; k < n_cols; ++k) sum += ggml_bf16_to_fp32(row[k]) * x_f32[k];
            out[r] = sum;
        }
        return;
    }

    ggml_type_traits_t traits;
    get_ggml_type_traits(W->type, &traits);
    const size_t row_bytes = ((size_t)n_cols / traits.block_elements) * traits.block_bytes;
    const uint8_t *base = (const uint8_t *)W->data;
    for (uint32_t r = 0; r < n_rows; ++r) {
        out[r] = lfm2_vec_dot_row(W, base + (size_t)r * row_bytes, x_f32, n_cols);
    }
}

static inline void lfm3_matmul_batch_mt(LFM2ThreadPool *pool,
                                        float * restrict out_batch,
                                        const GGUFTensor * restrict W,
                                        const float * restrict x_batch,
                                        uint32_t B,
                                        block_q8_K_t * restrict xq_scratch) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    const uint32_t n_rows = (uint32_t)W->ne[1];

    if ((n_cols % 256U) == 0U && lfm3_type_supports_1x4(W->type) &&
        !(W->type == GGML_TYPE_F32 && lfm3_is_forced_scalar())) {
        if (B == 1U && !lfm3_is_forced_scalar() && lfm3_use_avx2() && xq_scratch != NULL) {
            lfm3_quantize_q8_K(x_batch, xq_scratch, n_cols);
            lfm2_matmul_1x4_prequant_mt(pool, out_batch, W, x_batch, xq_scratch);
            return;
        }
        lfm2_matmul_batch_avx2_mt(pool, out_batch, W, x_batch, B, xq_scratch);
        return;
    }

    for (uint32_t t = 0; t < B; ++t) {
        lfm3_matmul_1x4_or_vec(pool, out_batch + (size_t)t * n_rows, W,
                               x_batch + (size_t)t * n_cols, xq_scratch);
    }
}

/* Single-barrier fused Q, K, V, and attn_gate projection for LFM3 M=1 decode */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *q_out;
    float              *k_out;
    float              *v_out;
    float              *g_out;
    const GGUFTensor   *wq;
    const GGUFTensor   *wk;
    const GGUFTensor   *wv;
    const GGUFTensor   *wg;
    uint32_t            q_rows;
    uint32_t            k_rows;
    uint32_t            v_rows;
    uint32_t            g_rows;
    const float        *x_f32;
    const block_q8_K_t *x_q8;
} LFM3FusedQKVGateCtx;

static void lfm3_gqa_qkvg_fused_worker(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    LFM3FusedQKVGateCtx *c = (LFM3FusedQKVGateCtx *)arg;
    const uint32_t q_end = c->q_rows;
    const uint32_t k_end = q_end + c->k_rows;
    const uint32_t v_end = k_end + c->v_rows;
    const uint32_t total = v_end + c->g_rows;
    uint32_t r_start = 0, r_end = 0;

    while (lfm2_threadpool_claim_chunk(c->pool, &r_start, &r_end)) {
        uint32_t r = r_start;
        for (; r + 4U <= r_end; ) {
            if (r + 4U <= q_end) {
                lfm2_compute_4rows_avx2(c->wq, r, c->x_f32, c->x_q8, c->q_out + r);
                r += 4U;
            } else if (r < q_end) {
                c->q_out[r] = lfm2_compute_1row_avx2(c->wq, r, c->x_f32, c->x_q8);
                r += 1U;
            } else if (r + 4U <= k_end) {
                const uint32_t kr = r - q_end;
                lfm2_compute_4rows_avx2(c->wk, kr, c->x_f32, c->x_q8, c->k_out + kr);
                r += 4U;
            } else if (r < k_end) {
                const uint32_t kr = r - q_end;
                c->k_out[kr] = lfm2_compute_1row_avx2(c->wk, kr, c->x_f32, c->x_q8);
                r += 1U;
            } else if (r + 4U <= v_end) {
                const uint32_t vr = r - k_end;
                lfm2_compute_4rows_avx2(c->wv, vr, c->x_f32, c->x_q8, c->v_out + vr);
                r += 4U;
            } else if (r < v_end) {
                const uint32_t vr = r - k_end;
                c->v_out[vr] = lfm2_compute_1row_avx2(c->wv, vr, c->x_f32, c->x_q8);
                r += 1U;
            } else if (r + 4U <= total && c->wg != NULL) {
                const uint32_t gr = r - v_end;
                lfm2_compute_4rows_avx2(c->wg, gr, c->x_f32, c->x_q8, c->g_out + gr);
                r += 4U;
            } else {
                const uint32_t gr = r - v_end;
                c->g_out[gr] = c->wg ? lfm2_compute_1row_avx2(c->wg, gr, c->x_f32, c->x_q8) : 0.0f;
                r += 1U;
            }
        }
        for (; r < r_end; ++r) {
            if (r < q_end) {
                c->q_out[r] = lfm2_compute_1row_avx2(c->wq, r, c->x_f32, c->x_q8);
            } else if (r < k_end) {
                const uint32_t kr = r - q_end;
                c->k_out[kr] = lfm2_compute_1row_avx2(c->wk, kr, c->x_f32, c->x_q8);
            } else if (r < v_end) {
                const uint32_t vr = r - k_end;
                c->v_out[vr] = lfm2_compute_1row_avx2(c->wv, vr, c->x_f32, c->x_q8);
            } else if (c->wg != NULL) {
                const uint32_t gr = r - v_end;
                c->g_out[gr] = lfm2_compute_1row_avx2(c->wg, gr, c->x_f32, c->x_q8);
            }
        }
    }
}

static inline void lfm3_gqa_qkvg_1x4_mt(LFM2ThreadPool *pool,
                                        float * restrict q_out,
                                        float * restrict k_out,
                                        float * restrict v_out,
                                        float * restrict gate_sig_out,
                                        const LFM2Layer * restrict layer,
                                        uint32_t n_head_q,
                                        uint32_t D,
                                        const float * restrict x_f32,
                                        block_q8_K_t * restrict x_q8) {
    const GGUFTensor *wq = layer->attn.wq;
    const GGUFTensor *wk = layer->attn.wk;
    const GGUFTensor *wv = layer->attn.wv;
    const GGUFTensor *wg = layer->attn.gate;

    const bool fast_1x4 = ((D % 256U) == 0U) && lfm3_use_avx2() &&
                          lfm3_type_supports_1x4(wq->type) &&
                          lfm3_type_supports_1x4(wk->type) &&
                          lfm3_type_supports_1x4(wv->type) &&
                          (!wg || lfm3_type_supports_1x4(wg->type));

    if (fast_1x4 && pool != NULL && pool->n_threads > 1U) {
        const uint32_t q_rows = (uint32_t)wq->ne[1];
        const uint32_t k_rows = (uint32_t)wk->ne[1];
        const uint32_t v_rows = (uint32_t)wv->ne[1];
        const uint32_t g_rows = wg ? (uint32_t)wg->ne[1] : 0U;
        LFM3FusedQKVGateCtx ctx = {
            .pool = pool,
            .q_out = q_out, .k_out = k_out, .v_out = v_out, .g_out = gate_sig_out,
            .wq = wq, .wk = wk, .wv = wv, .wg = wg,
            .q_rows = q_rows, .k_rows = k_rows, .v_rows = v_rows, .g_rows = g_rows,
            .x_f32 = x_f32, .x_q8 = x_q8
        };
        lfm2_parallel_for(pool, q_rows + k_rows + v_rows + g_rows, 16U, lfm3_gqa_qkvg_fused_worker, &ctx);
    } else {
        lfm3_matmul_1x4_or_vec(pool, q_out, wq, x_f32, x_q8);
        lfm3_matmul_1x4_or_vec(pool, k_out, wk, x_f32, x_q8);
        lfm3_matmul_1x4_or_vec(pool, v_out, wv, x_f32, x_q8);
        if (wg) {
            lfm3_matmul_1x4_or_vec(pool, gate_sig_out, wg, x_f32, x_q8);
        }
    }

    if (wg) {
        const uint32_t g_rows = (uint32_t)wg->ne[1];
        const uint32_t n_g = (g_rows < n_head_q) ? g_rows : n_head_q;
        for (uint32_t h = 0; h < n_g; ++h) {
            gate_sig_out[h] = 1.0f / (1.0f + expf(-gate_sig_out[h]));
        }
        for (uint32_t h = n_g; h < n_head_q; ++h) gate_sig_out[h] = 1.0f;
    } else {
        for (uint32_t h = 0; h < n_head_q; ++h) gate_sig_out[h] = 1.0f;
    }
}

/* ============================================================================
 * 6. Persistent LFM3 Runtime State (SWA Ring Buffers + Dual RoPE + Conv7)
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool pool;
    uint32_t max_seq_len;
    uint32_t sliding_window;
    uint32_t swa_cap;  /* sliding_window + LFM2_MAX_BATCH */
    uint32_t hist_len; /* LFM3_SHORTCONV_HIST = 6 */
    uint32_t head_dim; /* 128 for LFM3-1.7B */

    /* Batched activation scratch [LFM2_MAX_BATCH, dim_pad] */
    float        *x;
    float        *xb;
    float        *bcx;
    float        *conv_out;
    float        *q;
    float        *k;
    float        *v;
    float        *attn_gate; /* [LFM2_MAX_BATCH * n_head_q] */
    float        *hb;
    float        *hb2;
    block_q8_K_t *xq;
    float        *logits;
    float        *logits_batch;

    /* Dual RoPE Lookup Tables */
    LFM2RoPELUT   rope_lut_full;
    LFM2RoPELUT   rope_lut_swa;

    /* ShortConv 7-tap planar weights + circular ring buffers + speculative trace */
    LFM3PlanarConv7 planar_conv[LFM2_MAX_LAYERS];
    float          *conv_ring[LFM2_MAX_LAYERS];
    uint32_t        conv_ring_head[LFM2_MAX_LAYERS];
    float          *conv_trace[LFM2_MAX_BATCH][LFM2_MAX_LAYERS];
    uint32_t        conv_ring_head_trace[LFM2_MAX_BATCH][LFM2_MAX_LAYERS];

    /* Persistent FP16 KV Cache (SWA layers use swa_cap; Global uses max_seq_len) */
    uint16_t       *k_cache_f16[LFM2_MAX_LAYERS];
    uint16_t       *v_cache_f16[LFM2_MAX_LAYERS];
    uint32_t        kv_cap[LFM2_MAX_LAYERS];
} LFM3State;

void lfm3_state_free(LFM3State *s) {
    if (!s) return;
    lfm2_threadpool_free(&s->pool);
    free(s->x); free(s->xb); free(s->bcx); free(s->conv_out);
    free(s->q); free(s->k);  free(s->v);   free(s->attn_gate);
    free(s->hb); free(s->hb2); free(s->xq);
    free(s->logits); free(s->logits_batch);
    lfm2_rope_lut_free(&s->rope_lut_full);
    lfm2_rope_lut_free(&s->rope_lut_swa);

    for (uint32_t l = 0; l < LFM2_MAX_LAYERS; ++l) {
        lfm3_planar_conv7_free(&s->planar_conv[l]);
        free(s->conv_ring[l]);
        free(s->k_cache_f16[l]);
        free(s->v_cache_f16[l]);
        for (uint32_t b = 0; b < LFM2_MAX_BATCH; ++b) {
            free(s->conv_trace[b][l]);
        }
    }
    memset(s, 0, sizeof(*s));
}

bool lfm3_state_init(LFM3State *s, const LFM2Model *m, uint32_t max_seq_len, uint32_t n_threads) {
    if (!s || !m) return false;
    memset(s, 0, sizeof(*s));
    if (!lfm2_threadpool_init(&s->pool, n_threads)) return false;

    const LFM2Config *c = &m->cfg;
    s->max_seq_len    = lfm_resolve_ctx_len(max_seq_len, c->n_ctx);
    s->sliding_window = (c->sliding_window > 0U) ? c->sliding_window : 256U;
    s->swa_cap        = s->sliding_window + LFM2_MAX_BATCH;
    s->hist_len       = (c->shortconv_l_cache > 1U) ? (c->shortconv_l_cache - 1U) : LFM3_SHORTCONV_HIST;
    const size_t D    = c->n_embd;
    s->head_dim       = (c->head_dim > 0U) ? c->head_dim : (c->n_head_q > 0U ? (uint32_t)(D / c->n_head_q) : 128U);

    size_t max_ff = c->n_ff;
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].ffn_gate) {
            const size_t l_ff = (size_t)m->layers[l].ffn_gate->ne[1];
            if (l_ff > max_ff) max_ff = l_ff;
        }
    }
    if (max_ff == 0U) max_ff = D * 4U;

    size_t max_cols = (3U * D > max_ff) ? (3U * D) : max_ff;
    if (c->n_vocab > max_cols) max_cols = c->n_vocab;

    const size_t D_pad        = ((D + 255U) / 256U) * 256U;
    const size_t max_ff_pad   = ((max_ff + 255U) / 256U) * 256U;
    const size_t max_cols_pad = ((max_cols + 255U) / 256U) * 256U;
    const size_t n_vocab_pad  = ((c->n_vocab + 255U) / 256U) * 256U;
    const size_t B            = LFM2_MAX_BATCH;

    s->x            = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->xb           = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->bcx          = (float *)lfm3_aligned_calloc_64(B * 3U * D_pad * sizeof(float));
    s->conv_out     = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->q            = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->k            = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->v            = (float *)lfm3_aligned_calloc_64(B * D_pad * sizeof(float));
    s->attn_gate    = (float *)lfm3_aligned_calloc_64(B * ((size_t)c->n_head_q + 16U) * sizeof(float));
    s->hb           = (float *)lfm3_aligned_calloc_64(B * max_ff_pad * sizeof(float));
    s->hb2          = (float *)lfm3_aligned_calloc_64(B * max_ff_pad * sizeof(float));
    const size_t xq_blocks = (B * (max_cols_pad / 256U)) + ((D_pad + max_ff_pad) / 256U) + 4U;
    s->xq           = (block_q8_K_t *)lfm3_aligned_calloc_64(xq_blocks * sizeof(block_q8_K_t));
    s->logits       = (float *)lfm3_aligned_calloc_64(n_vocab_pad * sizeof(float));
    s->logits_batch = (float *)lfm3_aligned_calloc_64(B * n_vocab_pad * sizeof(float));

    if (!s->x || !s->xb || !s->bcx || !s->conv_out || !s->q || !s->k || !s->v ||
        !s->attn_gate || !s->hb || !s->hb2 || !s->xq || !s->logits || !s->logits_batch) {
        lfm3_state_free(s);
        return false;
    }

    const float freq_full = (c->rope_freq_base > 0.0f) ? c->rope_freq_base : 1000000.0f;
    const float freq_swa  = (c->rope_freq_base_swa > 0.0f) ? c->rope_freq_base_swa : 10000.0f;

    if (!lfm3_rope_lut_init(&s->rope_lut_full, s->max_seq_len, s->head_dim,
                            c->rope_dim, freq_full, m->rope_freqs) ||
        !lfm3_rope_lut_init(&s->rope_lut_swa, s->max_seq_len, s->head_dim,
                            s->head_dim, freq_swa, NULL)) {
        lfm3_state_free(s);
        return false;
    }

    const size_t conv_bytes = (size_t)s->hist_len * D * sizeof(float);
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        const LFM2Layer *ly = &m->layers[l];
        if (ly->block_type == LFM2_BLOCK_SHORTCONV) {
            if (!lfm3_planar_conv7_init(&s->planar_conv[l], ly->shortconv.conv, (uint32_t)D, c->shortconv_l_cache)) {
                lfm3_state_free(s);
                return false;
            }
            s->conv_ring[l] = (float *)lfm3_aligned_calloc_64(conv_bytes);
            if (!s->conv_ring[l]) {
                lfm3_state_free(s);
                return false;
            }
            for (uint32_t b = 0; b < B; ++b) {
                s->conv_trace[b][l] = (float *)lfm3_aligned_calloc_64(conv_bytes);
                if (!s->conv_trace[b][l]) {
                    lfm3_state_free(s);
                    return false;
                }
            }
        } else {
            const uint32_t kv_cap = (ly->block_type == LFM2_BLOCK_SWA_ATTN) ? s->swa_cap : s->max_seq_len;
            const size_t kv_dim   = (size_t)c->n_head_kv[l] * s->head_dim;
            const size_t kv_bytes = (size_t)kv_cap * kv_dim * sizeof(uint16_t);
            s->kv_cap[l]      = kv_cap;
            s->k_cache_f16[l] = (uint16_t *)lfm3_aligned_calloc_64(kv_bytes);
            s->v_cache_f16[l] = (uint16_t *)lfm3_aligned_calloc_64(kv_bytes);
            if (!s->k_cache_f16[l] || !s->v_cache_f16[l]) {
                lfm3_state_free(s);
                return false;
            }
#if defined(MADV_HUGEPAGE)
            if (kv_bytes >= (2U << 20)) {
                madvise(s->k_cache_f16[l], kv_bytes, MADV_HUGEPAGE);
                madvise(s->v_cache_f16[l], kv_bytes, MADV_HUGEPAGE);
            }
#endif
        }
    }

    return true;
}

void lfm3_state_reset(LFM3State *s, const LFM2Model *m) {
    if (!s || !m) return;
    const size_t D = m->cfg.n_embd;
    const size_t conv_bytes = (size_t)s->hist_len * D * sizeof(float);
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            if (s->conv_ring[l]) memset(s->conv_ring[l], 0, conv_bytes);
            s->conv_ring_head[l] = 0U;
        }
    }
}

void lfm3_restore_shortconv_trace_slot(LFM3State *s, const LFM2Model *m, uint32_t slot_b) {
    if (!s || !m || slot_b >= LFM2_MAX_BATCH) return;
    const size_t conv_bytes = (size_t)s->hist_len * m->cfg.n_embd * sizeof(float);
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV &&
            s->conv_ring[l] && s->conv_trace[slot_b][l]) {
            memcpy(s->conv_ring[l], s->conv_trace[slot_b][l], conv_bytes);
            s->conv_ring_head[l] = s->conv_ring_head_trace[slot_b][l];
        }
    }
}

/* ============================================================================
 * 7. Fast Single-Token Decode Kernel & Batched Prefill / Speculative Forward
 * ============================================================================ */
static void lfm3_forward_ffn_token(const LFM2Model *m, LFM3State *s, const LFM2Layer *layer) {
    const uint32_t D    = m->cfg.n_embd;
    const float    eps  = m->cfg.rms_norm_eps;
    const uint32_t n_ff = (uint32_t)layer->ffn_gate->ne[1];

    lfm3_rmsnorm_quant_scalar_safe(s->xb, s->xq, s->x, layer->ffn_norm, D, eps);

    const bool use_fused_1x4 = ((D % 256U) == 0U) && lfm3_use_avx2() &&
                               lfm3_type_supports_1x4(layer->ffn_gate->type) &&
                               lfm3_type_supports_1x4(layer->ffn_up->type);

    if (use_fused_1x4) {
        block_q8_K_t *hb_q8 = s->xq + (D / 256U);
        lfm2_ffn_swiglu_fused_1x4_mt(&s->pool, s->hb, hb_q8,
                                     layer->ffn_gate, layer->ffn_up, s->xb, s->xq);
        lfm3_matmul_1x4_or_vec(&s->pool, s->xb, layer->ffn_down, s->hb, hb_q8);
    } else {
        lfm3_matmul_batch_mt(&s->pool, s->hb,  layer->ffn_gate, s->xb, 1U, s->xq);
        lfm3_matmul_batch_mt(&s->pool, s->hb2, layer->ffn_up,   s->xb, 1U, s->xq);
        for (uint32_t i = 0; i < n_ff; ++i) {
            const float g = s->hb[i];
            s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
        }
        if ((n_ff % 256U) == 0U && s->xq) {
            lfm3_quantize_q8_K(s->hb, s->xq, n_ff);
        }
        lfm3_matmul_1x4_or_vec(&s->pool, s->xb, layer->ffn_down, s->hb, s->xq);
    }

    /* Sandwich Post-FFN RMSNorm + Residual Add */
    lfm3_post_norm_add_simd(s->x, s->xb, layer->post_ffn_norm, D, eps);
}

static float *lfm3_forward_token_internal(const LFM2Model *m,
                                          LFM3State *s,
                                          const float *embd_opt,
                                          uint32_t token_id,
                                          uint32_t pos,
                                          bool compute_logits,
                                          float *logits_out,
                                          bool save_trace,
                                          uint32_t trace_b) {
    const uint32_t D        = m->cfg.n_embd;
    const uint32_t head_dim = s->head_dim;
    const uint32_t n_head_q = m->cfg.n_head_q;
    const float    eps      = m->cfg.rms_norm_eps;

    if (pos >= s->max_seq_len) return NULL;
    if (embd_opt) {
        memcpy(s->x, embd_opt, (size_t)D * sizeof(float));
    } else {
        if (token_id >= m->cfg.n_vocab) return NULL;
        lfm2_dequantize_row(m->token_embd, token_id, s->x);
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];

        /* 1. Operator Pre-Norm + Quantize (shared across ShortConv in_proj or Q/K/V/Gate) */
        lfm3_rmsnorm_quant_scalar_safe(s->xb, s->xq, s->x, layer->attn_norm, D, eps);

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm3_matmul_1x4_or_vec(&s->pool, s->bcx, layer->shortconv.in_proj, s->xb, s->xq);
            lfm3_shortconv7_gate_conv_quant_simd(
                s->conv_out, ((D % 256U) == 0U) ? s->xq : NULL,
                s->bcx, s->conv_ring[l], &s->conv_ring_head[l],
                &s->planar_conv[l], D);

            if (save_trace && s->conv_trace[trace_b][l]) {
                memcpy(s->conv_trace[trace_b][l], s->conv_ring[l], (size_t)s->hist_len * D * sizeof(float));
                s->conv_ring_head_trace[trace_b][l] = s->conv_ring_head[l];
            }

            lfm3_matmul_1x4_or_vec(&s->pool, s->xb, layer->shortconv.out_proj, s->conv_out, s->xq);
        } else {
            const bool     is_swa     = (layer->block_type == LFM2_BLOCK_SWA_ATTN);
            const uint32_t n_head_kv  = m->cfg.n_head_kv[l];
            const size_t   kv_dim     = (size_t)n_head_kv * head_dim;
            const uint32_t kv_cap     = s->kv_cap[l];
            const uint32_t kv_slot    = is_swa ? (pos % kv_cap) : pos;
            const uint32_t window_len = is_swa ? s->sliding_window : 0U;
            const LFM2RoPELUT *lut    = is_swa ? &s->rope_lut_swa : &s->rope_lut_full;

            /* Fused Q, K, V, Gate projection in ONE barrier */
            lfm3_gqa_qkvg_1x4_mt(&s->pool, s->q, s->k, s->v, s->attn_gate,
                                 layer, n_head_q, D, s->xb, s->xq);

            uint16_t *dst_k16 = s->k_cache_f16[l] + (size_t)kv_slot * kv_dim;
            uint16_t *dst_v16 = s->v_cache_f16[l] + (size_t)kv_slot * kv_dim;

            lfm3_qk_norm_rope_simd(s->q, NULL,    n_head_q,  head_dim, layer->attn.q_norm, eps, lut, pos);
            lfm3_qk_norm_rope_simd(s->k, dst_k16, n_head_kv, head_dim, layer->attn.k_norm, eps, lut, pos);
            lfm3_store_f32_to_f16(dst_v16, s->v, (uint32_t)kv_dim);

            lfm3_online_gqa_f16_dispatch(&s->pool, s->q, s->attn_gate,
                                         s->k_cache_f16[l], s->v_cache_f16[l],
                                         s->k, ((D % 256U) == 0U && head_dim == 128U) ? s->xq : NULL,
                                         1U, pos, window_len, kv_cap,
                                         n_head_q, n_head_kv, head_dim);

            if ((D % 256U) == 0U && head_dim != 128U) {
                lfm3_quantize_q8_K(s->k, s->xq, D);
            }
            lfm3_matmul_1x4_or_vec(&s->pool, s->xb, layer->attn.wo, s->k, s->xq);
        }

        /* 2. Sandwich Post-Attention / Post-ShortConv RMSNorm + Residual Add */
        lfm3_post_norm_add_simd(s->x, s->xb, layer->post_attn_norm, D, eps);

        /* 3. SwiGLU FFN + Sandwich Post-FFN Norm + Residual Add */
        lfm3_forward_ffn_token(m, s, layer);
    }

    if (!compute_logits) return NULL;
    float *out_ptr = logits_out ? logits_out : s->logits;
    lfm3_rmsnorm_quant_scalar_safe(s->xb, s->xq, s->x, m->output_norm, D, eps);
    lfm3_matmul_1x4_or_vec(&s->pool, out_ptr, m->output, s->xb, s->xq);
    return out_ptr;
}

static float *lfm3_forward_batch_core(const LFM2Model *m,
                                      LFM3State *s,
                                      const uint32_t *tokens,
                                      const float *embd_batch,
                                      uint32_t B,
                                      uint32_t start_pos,
                                      bool compute_last_logits,
                                      bool compute_all_logits) {
    if (!m || !s || B == 0U || B > LFM2_MAX_BATCH) return NULL;
    if (start_pos + B > s->max_seq_len) return NULL;

    if (B == 1U && !compute_all_logits) {
        return lfm3_forward_token_internal(m, s, embd_batch, tokens ? tokens[0] : 0U,
                                           start_pos, compute_last_logits, s->logits, false, 0U);
    }
    if (B == 1U && compute_all_logits) {
        return lfm3_forward_token_internal(m, s, embd_batch, tokens ? tokens[0] : 0U,
                                           start_pos, true, s->logits_batch, true, 0U);
    }

    const uint32_t D        = m->cfg.n_embd;
    const uint32_t head_dim = s->head_dim;
    const uint32_t n_head_q = m->cfg.n_head_q;
    const uint32_t V        = m->cfg.n_vocab;
    const float    eps      = m->cfg.rms_norm_eps;

    if (embd_batch) {
        memcpy(s->x, embd_batch, (size_t)B * D * sizeof(float));
    } else {
        for (uint32_t b = 0; b < B; ++b) {
            if (tokens[b] >= V) return NULL;
            lfm2_dequantize_row(m->token_embd, tokens[b], s->x + (size_t)b * D);
        }
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];
        lfm2_rmsnorm_batch(s->xb, s->x, layer->attn_norm, B, D, eps);

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm3_matmul_batch_mt(&s->pool, s->bcx, layer->shortconv.in_proj, s->xb, B, s->xq);
            for (uint32_t b = 0; b < B; ++b) {
                lfm3_shortconv7_gate_conv_quant_simd(
                    s->conv_out + (size_t)b * D, NULL,
                    s->bcx + (size_t)b * 3U * D,
                    s->conv_ring[l], &s->conv_ring_head[l],
                    &s->planar_conv[l], D);
                if (compute_all_logits && s->conv_trace[b][l]) {
                    memcpy(s->conv_trace[b][l], s->conv_ring[l], (size_t)s->hist_len * D * sizeof(float));
                    s->conv_ring_head_trace[b][l] = s->conv_ring_head[l];
                }
            }
            lfm3_matmul_batch_mt(&s->pool, s->xb, layer->shortconv.out_proj, s->conv_out, B, s->xq);
        } else {
            const bool     is_swa     = (layer->block_type == LFM2_BLOCK_SWA_ATTN);
            const uint32_t n_head_kv  = m->cfg.n_head_kv[l];
            const size_t   kv_dim     = (size_t)n_head_kv * head_dim;
            const uint32_t kv_cap     = s->kv_cap[l];
            const uint32_t window_len = is_swa ? s->sliding_window : 0U;
            const LFM2RoPELUT *lut    = is_swa ? &s->rope_lut_swa : &s->rope_lut_full;

            lfm3_matmul_batch_mt(&s->pool, s->q, layer->attn.wq, s->xb, B, s->xq);
            lfm3_matmul_batch_mt(&s->pool, s->k, layer->attn.wk, s->xb, B, s->xq);
            lfm3_matmul_batch_mt(&s->pool, s->v, layer->attn.wv, s->xb, B, s->xq);

            if (layer->attn.gate) {
                const uint32_t g_rows = (uint32_t)layer->attn.gate->ne[1];
                lfm3_matmul_batch_mt(&s->pool, s->attn_gate, layer->attn.gate, s->xb, B, s->xq);
                for (uint32_t b = 0; b < B; ++b) {
                    float *gb = s->attn_gate + (size_t)b * g_rows;
                    for (uint32_t h = 0; h < g_rows; ++h) {
                        gb[h] = 1.0f / (1.0f + expf(-gb[h]));
                    }
                }
                /* If g_rows != n_head_q, pack stride=n_head_q */
                if (g_rows != n_head_q) {
                    for (int32_t b = (int32_t)B - 1; b >= 0; --b) {
                        float *src = s->attn_gate + (size_t)b * g_rows;
                        float *dst = s->attn_gate + (size_t)b * n_head_q;
                        for (int32_t h = (int32_t)n_head_q - 1; h >= 0; --h) {
                            dst[h] = ((uint32_t)h < g_rows) ? src[h] : 1.0f;
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < (size_t)B * n_head_q; ++i) s->attn_gate[i] = 1.0f;
            }

            for (uint32_t b = 0; b < B; ++b) {
                const uint32_t abs_pos = start_pos + b;
                const uint32_t kv_slot = is_swa ? (abs_pos % kv_cap) : abs_pos;
                lfm3_qk_norm_rope_simd(s->q + (size_t)b * D, NULL,
                                       n_head_q, head_dim, layer->attn.q_norm, eps, lut, abs_pos);
                lfm3_qk_norm_rope_simd(s->k + (size_t)b * kv_dim,
                                       s->k_cache_f16[l] + (size_t)kv_slot * kv_dim,
                                       n_head_kv, head_dim, layer->attn.k_norm, eps, lut, abs_pos);
                lfm3_store_f32_to_f16(s->v_cache_f16[l] + (size_t)kv_slot * kv_dim,
                                      s->v + (size_t)b * kv_dim, (uint32_t)kv_dim);
            }

            lfm3_online_gqa_f16_dispatch(&s->pool, s->q, s->attn_gate,
                                         s->k_cache_f16[l], s->v_cache_f16[l],
                                         s->conv_out, NULL,
                                         B, start_pos, window_len, kv_cap,
                                         n_head_q, n_head_kv, head_dim);

            lfm3_matmul_batch_mt(&s->pool, s->xb, layer->attn.wo, s->conv_out, B, s->xq);
        }

        /* Sandwich Post-Attention Norm + Residual Add */
        lfm3_post_norm_add_batch(s->x, s->xb, layer->post_attn_norm, B, D, eps);

        /* Batched SwiGLU FFN + Sandwich Post-FFN Norm + Residual Add */
        const uint32_t n_ff = (uint32_t)layer->ffn_gate->ne[1];
        lfm2_rmsnorm_batch(s->xb, s->x, layer->ffn_norm, B, D, eps);
        lfm3_matmul_batch_mt(&s->pool, s->hb,  layer->ffn_gate, s->xb, B, s->xq);
        lfm3_matmul_batch_mt(&s->pool, s->hb2, layer->ffn_up,   s->xb, B, s->xq);
        const size_t total_ff = (size_t)B * n_ff;
        for (size_t i = 0; i < total_ff; ++i) {
            const float g = s->hb[i];
            s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
        }
        lfm3_matmul_batch_mt(&s->pool, s->xb, layer->ffn_down, s->hb, B, s->xq);
        lfm3_post_norm_add_batch(s->x, s->xb, layer->post_ffn_norm, B, D, eps);
    }

    if (compute_all_logits) {
        lfm2_rmsnorm_batch(s->xb, s->x, m->output_norm, B, D, eps);
        lfm3_matmul_batch_mt(&s->pool, s->logits_batch, m->output, s->xb, B, s->xq);
        return s->logits_batch;
    }

    if (!compute_last_logits) return NULL;
    float *last_x = s->x + (size_t)(B - 1U) * D;
    lfm3_rmsnorm_quant_scalar_safe(s->xb, s->xq, last_x, m->output_norm, D, eps);
    lfm3_matmul_1x4_or_vec(&s->pool, s->logits, m->output, s->xb, s->xq);
    return s->logits;
}

float *lfm3_forward_token(const LFM2Model *m, LFM3State *s, uint32_t token_id, uint32_t pos, bool compute_logits) {
    return lfm3_forward_token_internal(m, s, NULL, token_id, pos, compute_logits, s->logits, false, 0U);
}

float *lfm3_forward_batch(const LFM2Model *m, LFM3State *s, const uint32_t *tokens,
                          uint32_t B, uint32_t start_pos, bool compute_last_logits) {
    return lfm3_forward_batch_core(m, s, tokens, NULL, B, start_pos, compute_last_logits, false);
}

float *lfm3_forward_speculative_batch(const LFM2Model *m, LFM3State *s, const uint32_t *tokens,
                                      uint32_t B, uint32_t start_pos) {
    return lfm3_forward_batch_core(m, s, tokens, NULL, B, start_pos, false, true);
}

void lfm3_forward_embd_batch(const LFM2Model *m, LFM3State *s, const float *embd,
                             uint32_t B, uint32_t start_pos, bool compute_last_logits) {
    lfm3_forward_batch_core(m, s, NULL, embd, B, start_pos, compute_last_logits, false);
}

typedef uint32_t (*lfm3_spec_sample_fn)(float *logits, uint32_t n_vocab, void *user_data);

uint32_t lfm3_speculative_decode_step(const LFM2Model *m,
                                      LFM3State *s,
                                      LFM2NgramIndex *ngram_idx,
                                      uint32_t *inout_pending_token,
                                      uint32_t *inout_pos,
                                      lfm3_spec_sample_fn sample_fn,
                                      void *user_sampler,
                                      uint32_t out_emitted_tokens[LFM2_MAX_BATCH]) {
    if (!m || !s || !ngram_idx || !inout_pending_token || !inout_pos || !sample_fn) return 0U;

    const uint32_t V       = m->cfg.n_vocab;
    const uint32_t cur_pos = *inout_pos;
    const uint32_t t_0     = *inout_pending_token;

    if (cur_pos >= s->max_seq_len) return 0U;

    uint32_t drafts[LFM2_PLD_MAX_DRAFT];
    uint32_t K = lfm2_ngram_draft(ngram_idx, t_0, drafts);

    uint32_t max_k_allowed = s->max_seq_len - cur_pos - 1U;
    if (K > max_k_allowed) K = max_k_allowed;
    if (K > LFM2_MAX_BATCH - 1U) K = LFM2_MAX_BATCH - 1U;

    uint32_t batch_in[LFM2_MAX_BATCH];
    batch_in[0] = t_0;
    for (uint32_t k = 0; k < K; ++k) batch_in[1U + k] = drafts[k];
    const uint32_t B = 1U + K;

    ngram_idx->total_steps++;
    ngram_idx->total_draft_tokens_proposed += K;

    float *logits_batch = lfm3_forward_speculative_batch(m, s, batch_in, B, cur_pos);
    if (!logits_batch) return 0U;

    lfm2_ngram_index_ingest(ngram_idx, t_0);

    uint32_t n_accepted = 0U, n_emitted = 0U, final_tok = 0U, last_b = 0U;
    for (uint32_t b = 0; b < B; ++b) {
        last_b = b;
        float *slot_logits = logits_batch + (size_t)b * V;
        const uint32_t sampled_id = sample_fn(slot_logits, V, user_sampler);

        if (b < K && sampled_id == drafts[b]) {
            out_emitted_tokens[n_emitted++] = sampled_id;
            lfm2_ngram_index_ingest(ngram_idx, sampled_id);
            n_accepted++;
            if (sampled_id == ngram_idx->eos_token_id) {
                final_tok = sampled_id;
                break;
            }
        } else {
            out_emitted_tokens[n_emitted++] = sampled_id;
            final_tok = sampled_id;
            break;
        }
    }

    if (n_accepted < K) {
        lfm3_restore_shortconv_trace_slot(s, m, n_accepted);
    }
    if (s->logits && logits_batch) {
        memcpy(s->logits, logits_batch + (size_t)last_b * V, (size_t)V * sizeof(float));
    }

    *inout_pos           = cur_pos + n_accepted + 1U;
    *inout_pending_token = final_tok;
    ngram_idx->total_draft_tokens_accepted += n_accepted;
    lfm2_ngram_update_adaptive_window(ngram_idx, K, n_accepted);
    return n_emitted;
}

#ifndef LFM3_ENGINE_NO_MAIN
int main(int argc, char **argv) {
    const char *path = (argc >= 2) ? argv[1] : "models/LFM3-1.7B-Q4_K_M.gguf";
    LFM2Model m;
    if (!lfm2_load_gguf(path, &m)) {
        fprintf(stderr, "Failed to load LFM3 model at %s\n", path);
        return EXIT_FAILURE;
    }
    if (!m.cfg.is_lfm3) {
        fprintf(stderr, "Model %s is arch='%s', not lfm3\n", path, m.cfg.arch);
        lfm2_free_model(&m);
        return EXIT_FAILURE;
    }
    LFM3State st;
    if (!lfm3_state_init(&st, &m, 512U, 0U)) {
        fprintf(stderr, "Failed to init LFM3State\n");
        lfm2_free_model(&m);
        return EXIT_FAILURE;
    }
    uint32_t prompt[4] = {124894U, 124899U, 5922U, 124900U};
    double t0 = now_ms();
    float *logits = lfm3_forward_batch(&m, &st, prompt, 4U, 0U, true);
    double dt = now_ms() - t0;
    if (!logits) {
        lfm3_state_free(&st);
        lfm2_free_model(&m);
        return EXIT_FAILURE;
    }
    uint32_t best = 0U;
    for (uint32_t i = 1U; i < m.cfg.n_vocab; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    printf("LFM3 Engine Smoke OK: prefill 4 tok in %.2f ms -> top next token=%u (logit=%.4f)\n",
           dt, best, logits[best]);
    lfm3_state_free(&st);
    lfm2_free_model(&m);
    return EXIT_SUCCESS;
}
#endif

#endif /* LFM3_ENGINE_INCLUDED */
