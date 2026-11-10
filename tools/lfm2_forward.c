#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#ifndef LFM2_FORWARD_INCLUDED
#define LFM2_FORWARD_INCLUDED 1
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LFM2_LOADER_NO_MAIN 1
#include "lfm2_loader.c"

/* ============================================================================
 * 1. IEEE 754 FP16 / BF16 & Quantized Block Structures (GGML Format)
 * ============================================================================ */

static inline float ggml_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
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

static inline float ggml_bf16_to_fp32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

#include "lfm2_simd_avx2.h"
#ifndef LFM2_FP32_TO_FP16_DEFINED
#define LFM2_FP32_TO_FP16_DEFINED 1
static inline uint16_t lfm2_fp32_to_fp16(float f) {
#if defined(__F16C__)
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (exp == 128 + 15 && mant != 0)
             ? (uint16_t)(sign | 0x7E00u | (mant >> 13))
             : (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | (((uint32_t)exp << 10) + ((mant + 0x1000u) >> 13)));
#endif
}
#endif
#define get_scale_min_k4 lfm2_get_scale_min_k4

#define LFM2_THREADPOOL_NO_MAIN 1
#if defined(__has_include)
#if __has_include("tools/lfm2_threadpool.c")
#include "tools/lfm2_threadpool.c"
#elif __has_include("lfm2_threadpool.c")
#include "lfm2_threadpool.c"
#endif
#else
#include "lfm2_threadpool.c"
#endif

/* ============================================================================
 * 2. Quantized Vector Dot-Product & Row Dequantization Kernels
 * ============================================================================ */

/* Dequantizes a single row of any supported GGUF tensor into float dst[n_cols]
 * Used for token embedding lookup (token_embd.weight[token_id]). */
static void lfm2_dequantize_row(const GGUFTensor *t, uint32_t row_idx, float *dst) {
    const uint32_t n_cols = (uint32_t)t->ne[0];
    ggml_type_traits_t traits;
    get_ggml_type_traits(t->type, &traits);

    const size_t row_bytes = ((size_t)n_cols / traits.block_elements) * traits.block_bytes;
    const uint8_t *row_ptr = (const uint8_t *)t->data + (size_t)row_idx * row_bytes;

    switch (t->type) {
        case GGML_TYPE_F32: {
            memcpy(dst, row_ptr, n_cols * sizeof(float));
            break;
        }
        case GGML_TYPE_F16: {
            const uint16_t *src = (const uint16_t *)row_ptr;
            for (uint32_t i = 0; i < n_cols; ++i) dst[i] = ggml_fp16_to_fp32(src[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t *src = (const uint16_t *)row_ptr;
            for (uint32_t i = 0; i < n_cols; ++i) dst[i] = ggml_bf16_to_fp32(src[i]);
            break;
        }
        case GGML_TYPE_Q8_0: {
            const uint32_t nb = n_cols / 32;
            const block_q8_0_t *blocks = (const block_q8_0_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                for (uint32_t j = 0; j < 32; ++j) {
                    dst[b * 32 + j] = d * (float)blocks[b].qs[j];
                }
            }
            break;
        }
        case GGML_TYPE_Q4_K: {
            const uint32_t nb = n_cols / 256;
            const block_q4_K_t *blocks = (const block_q4_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d   = ggml_fp16_to_fp32(blocks[b].d);
                const float min = ggml_fp16_to_fp32(blocks[b].dmin);
                const uint8_t *q = blocks[b].qs;
                float *out = dst + b * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blocks[b].scales, &sc1, &m1);
                    get_scale_min_k4(is + 1, blocks[b].scales, &sc2, &m2);
                    const float d1 = d * sc1, min1 = min * m1;
                    const float d2 = d * sc2, min2 = min * m2;
                    for (int l = 0; l < 32; ++l) out[j + l]      = d1 * (q[l] & 0x0F) - min1;
                    for (int l = 0; l < 32; ++l) out[j + 32 + l] = d2 * (q[l] >> 4)   - min2;
                    q += 32;
                    is += 2;
                }
            }
            break;
        }
        case GGML_TYPE_Q6_K: {
            const uint32_t nb = n_cols / 256;
            const block_q6_K_t *blocks = (const block_q6_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const uint8_t *ql = blocks[b].ql;
                const uint8_t *qh = blocks[b].qh;
                const int8_t  *sc = blocks[b].scales;
                float *y = dst + b * 256;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y  += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            const uint32_t nb = n_cols / 32;
            const block_q4_0_t *blocks = (const block_q4_0_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                for (uint32_t l = 0; l < 16; ++l) {
                    const int8_t v0 = (int8_t)(blocks[b].qs[l] & 0x0F) - 8;
                    const int8_t v1 = (int8_t)(blocks[b].qs[l] >> 4)   - 8;
                    dst[b * 32 + l]      = d * (float)v0;
                    dst[b * 32 + l + 16] = d * (float)v1;
                }
            }
            break;
        }
        case GGML_TYPE_Q5_K: {
            const uint32_t nb = n_cols / 256;
            const block_q5_K_t *blocks = (const block_q5_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d   = ggml_fp16_to_fp32(blocks[b].d);
                const float min = ggml_fp16_to_fp32(blocks[b].dmin);
                const uint8_t *ql = blocks[b].qs;
                const uint8_t *qh = blocks[b].qh;
                float *out = dst + b * 256;
                uint8_t u1 = 1, u2 = 2;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blocks[b].scales, &sc1, &m1);
                    get_scale_min_k4(is + 1, blocks[b].scales, &sc2, &m2);
                    const float d1 = d * (float)sc1, min1 = min * (float)m1;
                    const float d2 = d * (float)sc2, min2 = min * (float)m2;
                    for (int l = 0; l < 32; ++l) {
                        int32_t q_lo = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                        int32_t q_hi = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                        out[j + l]      = d1 * (float)q_lo - min1;
                        out[j + 32 + l] = d2 * (float)q_hi - min2;
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            break;
        }
        case GGML_TYPE_IQ4_XS: {
            const uint32_t nb = n_cols / 256;
            const block_iq4_xs_t *blocks = (const block_iq4_xs_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const uint8_t *qs = blocks[b].qs;
                float *out = dst + b * 256;
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = get_scale_iq4_xs(ib, blocks[b].scales_l, blocks[b].scales_h);
                    const float d_sub = d * (float)ls;
                    for (int l = 0; l < 16; ++l) {
                        out[ib * 32 + l]      = d_sub * (float)kvalues_iq4nl[qs[l] & 0x0F];
                        out[ib * 32 + l + 16] = d_sub * (float)kvalues_iq4nl[qs[l] >> 4];
                    }
                    qs += 16;
                }
            }
            break;
        }
        default:
            fprintf(stderr, "[FATAL] Unsupported tensor type %u in dequantize_row (%s)\n",
                    t->type, t->name);
            exit(EXIT_FAILURE);
    }
}

/* Computes dot product between one quantized row of W and float vector x[n_cols] */
static float lfm2_vec_dot_row(const GGUFTensor *W, const uint8_t *row_ptr, const float *x, uint32_t n_cols) {
    float sumf = 0.0f;

    switch (W->type) {
        case GGML_TYPE_F32: {
            const float *w = (const float *)row_ptr;
            for (uint32_t i = 0; i < n_cols; ++i) sumf += w[i] * x[i];
            return sumf;
        }
        case GGML_TYPE_F16: {
            const uint16_t *w = (const uint16_t *)row_ptr;
            for (uint32_t i = 0; i < n_cols; ++i) sumf += ggml_fp16_to_fp32(w[i]) * x[i];
            return sumf;
        }
        case GGML_TYPE_BF16: {
            const uint16_t *w = (const uint16_t *)row_ptr;
            for (uint32_t i = 0; i < n_cols; ++i) sumf += ggml_bf16_to_fp32(w[i]) * x[i];
            return sumf;
        }
        case GGML_TYPE_Q8_0: {
            const uint32_t nb = n_cols / 32;
            const block_q8_0_t *blocks = (const block_q8_0_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const float *xb = x + b * 32;
                float block_sum = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    block_sum += (float)blocks[b].qs[j] * xb[j];
                }
                sumf += d * block_sum;
            }
            return sumf;
        }
        case GGML_TYPE_Q4_K: {
            const uint32_t nb = n_cols / 256;
            const block_q4_K_t *blocks = (const block_q4_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d   = ggml_fp16_to_fp32(blocks[b].d);
                const float min = ggml_fp16_to_fp32(blocks[b].dmin);
                const uint8_t *q = blocks[b].qs;
                const float *xb  = x + b * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blocks[b].scales, &sc1, &m1);
                    get_scale_min_k4(is + 1, blocks[b].scales, &sc2, &m2);
                    const float d1 = d * sc1, min1 = min * m1;
                    const float d2 = d * sc2, min2 = min * m2;

                    float s1 = 0.0f, s1_sumx = 0.0f;
                    float s2 = 0.0f, s2_sumx = 0.0f;
                    for (int l = 0; l < 32; ++l) {
                        s1      += (float)(q[l] & 0x0F) * xb[j + l];
                        s1_sumx += xb[j + l];
                        s2      += (float)(q[l] >> 4)   * xb[j + 32 + l];
                        s2_sumx += xb[j + 32 + l];
                    }
                    sumf += (d1 * s1 - min1 * s1_sumx) + (d2 * s2 - min2 * s2_sumx);
                    q += 32;
                    is += 2;
                }
            }
            return sumf;
        }
        case GGML_TYPE_Q6_K: {
            const uint32_t nb = n_cols / 256;
            const block_q6_K_t *blocks = (const block_q6_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const uint8_t *ql = blocks[b].ql;
                const uint8_t *qh = blocks[b].qh;
                const int8_t  *sc = blocks[b].scales;
                const float   *xb = x + b * 256;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        sumf += d * (
                            sc[is + 0] * q1 * xb[l +  0] +
                            sc[is + 2] * q2 * xb[l + 32] +
                            sc[is + 4] * q3 * xb[l + 64] +
                            sc[is + 6] * q4 * xb[l + 96]
                        );
                    }
                    xb += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return sumf;
        }
        case GGML_TYPE_Q4_0: {
            const uint32_t nb = n_cols / 32;
            const block_q4_0_t *blocks = (const block_q4_0_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const float *xb = x + b * 32;
                float block_sum = 0.0f;
                for (int l = 0; l < 16; ++l) {
                    const int8_t v0 = (int8_t)(blocks[b].qs[l] & 0x0F) - 8;
                    const int8_t v1 = (int8_t)(blocks[b].qs[l] >> 4)   - 8;
                    block_sum += (float)v0 * xb[l] + (float)v1 * xb[l + 16];
                }
                sumf += d * block_sum;
            }
            return sumf;
        }
        case GGML_TYPE_Q5_K: {
            const uint32_t nb = n_cols / 256;
            const block_q5_K_t *blocks = (const block_q5_K_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d   = ggml_fp16_to_fp32(blocks[b].d);
                const float min = ggml_fp16_to_fp32(blocks[b].dmin);
                const uint8_t *ql = blocks[b].qs;
                const uint8_t *qh = blocks[b].qh;
                const float *xb  = x + b * 256;
                uint8_t u1 = 1, u2 = 2;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blocks[b].scales, &sc1, &m1);
                    get_scale_min_k4(is + 1, blocks[b].scales, &sc2, &m2);
                    const float d1 = d * (float)sc1, min1 = min * (float)m1;
                    const float d2 = d * (float)sc2, min2 = min * (float)m2;
                    for (int l = 0; l < 32; ++l) {
                        int32_t q_lo = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                        int32_t q_hi = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                        sumf += (d1 * (float)q_lo - min1) * xb[j + l]
                              + (d2 * (float)q_hi - min2) * xb[j + 32 + l];
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            return sumf;
        }
        case GGML_TYPE_IQ4_XS: {
            const uint32_t nb = n_cols / 256;
            const block_iq4_xs_t *blocks = (const block_iq4_xs_t *)row_ptr;
            for (uint32_t b = 0; b < nb; ++b) {
                const float d = ggml_fp16_to_fp32(blocks[b].d);
                const uint8_t *qs = blocks[b].qs;
                const float *xb = x + b * 256;
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = get_scale_iq4_xs(ib, blocks[b].scales_l, blocks[b].scales_h);
                    const float d_sub = d * (float)ls;
                    for (int l = 0; l < 16; ++l) {
                        sumf += d_sub * (float)kvalues_iq4nl[qs[l] & 0x0F] * xb[ib * 32 + l]
                              + d_sub * (float)kvalues_iq4nl[qs[l] >> 4]   * xb[ib * 32 + l + 16];
                    }
                    qs += 16;
                }
            }
            return sumf;
        }
        default:
            fprintf(stderr, "[FATAL] Unsupported tensor type %u in matmul (%s)\n",
                    W->type, W->name);
            exit(EXIT_FAILURE);
    }
}

/* Matrix-Vector Multiplication: out[n_rows] = W[n_cols, n_rows]^T * x[n_cols]
 * Note: In GGUF, ne[0] = n_cols (contiguous row length), ne[1] = n_rows (output dim). */
static void lfm2_matmul_vec(float *out, const GGUFTensor *W, const float *x,
                            block_q8_K_t *xq_scratch, LFM2ThreadPool *pool) {
    const uint32_t n_cols = (uint32_t)W->ne[0];
    const uint32_t n_rows = (uint32_t)W->ne[1];

#if LFM2_HAS_AVX2
    if (xq_scratch && (W->type == GGML_TYPE_Q4_K || W->type == GGML_TYPE_Q6_K ||
                       W->type == GGML_TYPE_Q8_0 || W->type == GGML_TYPE_F32)) {
        lfm2_matmul_vec_avx2_mt(pool, out, W->data, W->type, n_cols, n_rows, x, xq_scratch);
        return;
    }
#endif
    (void)xq_scratch;
    (void)pool;

    ggml_type_traits_t traits;
    get_ggml_type_traits(W->type, &traits);
    const size_t row_bytes = ((size_t)n_cols / traits.block_elements) * traits.block_bytes;
    const uint8_t *base = (const uint8_t *)W->data;

    for (uint32_t r = 0; r < n_rows; ++r) {
        out[r] = lfm2_vec_dot_row(W, base + (size_t)r * row_bytes, x, n_cols);
    }
}

/* ============================================================================
 * 3. Pre-Allocated Inference Runtime State (Arena for Hybrid Cache & Buffers)
 * ============================================================================ */
typedef struct {
    /* Ping-pong residual & normalized activation buffers */
    float *x;           /* [n_embd] Current residual stream                  */
    float *xb;          /* [n_embd] RMSNorm output / operator output         */
    float *bcx;         /* [3 * n_embd] ShortConv in_proj buffer (B, C, x)   */
    float *conv_out;    /* [n_embd] ShortConv depthwise conv1d output        */

    /* GQA Attention scratch buffers */
    float *q;           /* [n_embd] Query vector (n_head_q * head_dim)       */
    float *k;           /* [n_embd] Current step Key vector                  */
    float *v;           /* [n_embd] Current step Value vector                */
    float *att;         /* [n_head_q * max_seq_len] Attention score buffer   */

    /* SwiGLU FFN scratch buffers */
    float *hb;          /* [n_ff] ffn_gate projection & SwiGLU output        */
    float *hb2;         /* [n_ff] ffn_up projection                          */

    /* Output logits */
    float *logits;      /* [n_vocab] Next-token log-probabilities            */

    /* Fast integer SIMD activation scratch buffer */
    block_q8_K_t *xq_scratch;

    /* Custom C11 physical-core spin-barrier thread pool */
    LFM2ThreadPool pool;

    /* Fast trigonometric tables for NeoX RoPE */
    float   *rope_cos;          /* [max_seq_len * (head_dim / 2)] */
    float   *rope_sin;          /* [max_seq_len * (head_dim / 2)] */

    /* Pre-unpacked 3-tap ShortConv weights: conv_w[l][3][D] for 8-wide AVX2 FMA depthwise conv */
    float   *conv_w[LFM2_MAX_LAYERS][3];

    /* Hybrid Persistent State across Tokens:
     * - conv_state[l] is allocated ONLY for LFM2_BLOCK_SHORTCONV layers:
     *   size = (l_cache - 1) * n_embd floats (typically 2 * n_embd = 16 KB!)
     * - k_cache[l], v_cache[l] are allocated ONLY for LFM2_BLOCK_GQA_ATTN layers:
     *   size = max_seq_len * (n_head_kv[l] * head_dim) floats
     * - k_cache_f16[l], v_cache_f16[l] are FP16 KV caches for hardware F16C single-pass online softmax */
    uint32_t  max_seq_len;
    float    *conv_state[LFM2_MAX_LAYERS];
    float    *k_cache[LFM2_MAX_LAYERS];
    float    *v_cache[LFM2_MAX_LAYERS];
    uint16_t *k_cache_f16[LFM2_MAX_LAYERS];
    uint16_t *v_cache_f16[LFM2_MAX_LAYERS];
} LFM2State;

bool lfm2_state_init(LFM2State *s, const LFM2Model *m, uint32_t max_seq_len) {
    memset(s, 0, sizeof(*s));
    const LFM2Config *c = &m->cfg;
    s->max_seq_len = (max_seq_len > 0 && max_seq_len <= c->n_ctx) ? max_seq_len : 4096;

    const uint32_t D         = c->n_embd;
    const uint32_t head_dim  = c->n_head_q > 0 ? (D / c->n_head_q) : 64;
    const uint32_t half_dim  = head_dim / 2;
    const uint32_t hist_len  = (c->shortconv_l_cache > 1) ? (c->shortconv_l_cache - 1) : 2;

    /* Compute max FFN dimension across all layers */
    uint32_t max_ff = c->n_ff;
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].ffn_gate) {
            uint32_t l_ff = (uint32_t)m->layers[l].ffn_gate->ne[1];
            if (l_ff > max_ff) max_ff = l_ff;
        }
    }
    if (max_ff == 0) max_ff = D * 4;

    uint32_t max_cols = (3 * D > max_ff) ? (3 * D) : max_ff;
    if (c->n_vocab > max_cols) max_cols = c->n_vocab;

    s->x        = (float *)calloc(D, sizeof(float));
    s->xb       = (float *)calloc(D, sizeof(float));
    s->bcx      = (float *)calloc(3 * D, sizeof(float));
    s->conv_out = (float *)calloc(D, sizeof(float));
    s->q        = (float *)calloc(D, sizeof(float));
    s->k        = (float *)calloc(D, sizeof(float));
    s->v        = (float *)calloc(D, sizeof(float));
    s->att      = (float *)calloc((size_t)c->n_head_q * s->max_seq_len, sizeof(float));
    s->hb       = (float *)calloc(max_ff, sizeof(float));
    s->hb2      = (float *)calloc(max_ff, sizeof(float));
    s->logits   = (float *)calloc(c->n_vocab, sizeof(float));
    s->xq_scratch = (block_q8_K_t *)calloc(max_cols / 256 + 1, sizeof(block_q8_K_t));

    s->rope_cos = (float *)calloc((size_t)s->max_seq_len * half_dim, sizeof(float));
    s->rope_sin = (float *)calloc((size_t)s->max_seq_len * half_dim, sizeof(float));

    if (!s->x || !s->xb || !s->bcx || !s->conv_out || !s->q ||
        !s->k || !s->v  || !s->att || !s->hb || !s->hb2 || !s->logits || !s->xq_scratch ||
        !s->rope_cos || !s->rope_sin) {
        return false;
    }

    /* Precompute RoPE cos/sin tables across all max_seq_len positions */
    const float freq_base = (c->rope_freq_base > 0.0f) ? c->rope_freq_base : 10000.0f;
    for (uint32_t i = 0; i < half_dim; ++i) {
        float freq = powf(freq_base, -((float)(2 * i) / (float)head_dim));
        for (uint32_t p = 0; p < s->max_seq_len; ++p) {
            float angle = (float)p * freq;
            s->rope_cos[(size_t)p * half_dim + i] = cosf(angle);
            s->rope_sin[(size_t)p * half_dim + i] = sinf(angle);
        }
    }

    if (!lfm2_threadpool_init(&s->pool, 0)) {
        return false;
    }

    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            s->conv_state[l] = (float *)calloc((size_t)hist_len * D, sizeof(float));
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
                            val = stride_is_lcache ? ggml_fp16_to_fp32(w_base[d * l_cache + k]) : ggml_fp16_to_fp32(w_base[k * D + d]);
                        }
                        s->conv_w[l][k][d] = val;
                    }
                }
            }
        } else {
            const uint32_t kv_dim = c->n_head_kv[l] * head_dim;
            s->k_cache[l]     = (float *)calloc((size_t)s->max_seq_len * kv_dim, sizeof(float));
            s->v_cache[l]     = (float *)calloc((size_t)s->max_seq_len * kv_dim, sizeof(float));
            s->k_cache_f16[l] = (uint16_t *)calloc((size_t)s->max_seq_len * kv_dim, sizeof(uint16_t));
            s->v_cache_f16[l] = (uint16_t *)calloc((size_t)s->max_seq_len * kv_dim, sizeof(uint16_t));
#if defined(MADV_HUGEPAGE)
            madvise(s->k_cache_f16[l], (size_t)s->max_seq_len * kv_dim * sizeof(uint16_t), MADV_HUGEPAGE);
            madvise(s->v_cache_f16[l], (size_t)s->max_seq_len * kv_dim * sizeof(uint16_t), MADV_HUGEPAGE);
#endif
            if (!s->k_cache[l] || !s->v_cache[l] || !s->k_cache_f16[l] || !s->v_cache_f16[l]) return false;
        }
    }
    return true;
}

void lfm2_state_free(LFM2State *s) {
    lfm2_threadpool_free(&s->pool);
    free(s->x);        free(s->xb);       free(s->bcx);
    free(s->conv_out); free(s->q);        free(s->k);
    free(s->v);        free(s->att);      free(s->hb);
    free(s->hb2);      free(s->logits);   free(s->xq_scratch);
    free(s->rope_cos); free(s->rope_sin);
    for (int l = 0; l < LFM2_MAX_LAYERS; ++l) {
        free(s->conv_state[l]);
        for (int k = 0; k < 3; ++k) {
            free(s->conv_w[l][k]);
        }
        free(s->k_cache[l]);
        free(s->v_cache[l]);
        free(s->k_cache_f16[l]);
        free(s->v_cache_f16[l]);
    }
    memset(s, 0, sizeof(*s));
}

/* ============================================================================
 * 4. Core Math Primitives: RMSNorm, Per-Head QK-Norm, and NeoX RoPE
 * ============================================================================ */

/* Standard RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i] */
static inline void lfm2_fwd_vec_add_f32(float * restrict dst, const float * restrict src, uint32_t n) {
    uint32_t d = 0;
#if LFM2_HAS_AVX2
    for (; d + 8 <= n; d += 8) {
        _mm256_storeu_ps(dst + d, _mm256_add_ps(_mm256_loadu_ps(dst + d), _mm256_loadu_ps(src + d)));
    }
#elif defined(__aarch64__)
    for (; d + 4 <= n; d += 4) {
        vst1q_f32(dst + d, vaddq_f32(vld1q_f32(dst + d), vld1q_f32(src + d)));
    }
#endif
    for (; d < n; ++d) {
        dst[d] += src[d];
    }
}

static void lfm2_rmsnorm(float *out, const float *x, const GGUFTensor *weight_tensor,
                         uint32_t dim, float eps) {
#if LFM2_HAS_AVX2
    __m256 sum256 = _mm256_setzero_ps();
    uint32_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 vx = _mm256_loadu_ps(x + d);
        sum256 = _mm256_fmadd_ps(vx, vx, sum256);
    }
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum256);
    double ss = 0.0;
    for (int i = 0; i < 8; ++i) ss += (double)sum_arr[i];
    for (; d < dim; ++d) ss += (double)x[d] * (double)x[d];
    const float inv_rms = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
    __m256 vinv = _mm256_set1_ps(inv_rms);

    if (weight_tensor->type == GGML_TYPE_F32) {
        const float *w = (const float *)weight_tensor->data;
        d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 vx = _mm256_loadu_ps(x + d);
            __m256 vw = _mm256_loadu_ps(w + d);
            _mm256_storeu_ps(out + d, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
        for (; d < dim; ++d) out[d] = x[d] * inv_rms * w[d];
    } else {
        const uint16_t *w = (const uint16_t *)weight_tensor->data;
        d = 0;
#if defined(__F16C__)
        for (; d + 8 <= dim; d += 8) {
            __m256 vx = _mm256_loadu_ps(x + d);
            __m128i hw = _mm_loadu_si128((const __m128i *)(w + d));
            __m256 vw = _mm256_cvtph_ps(hw);
            _mm256_storeu_ps(out + d, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
#endif
        for (; d < dim; ++d) out[d] = x[d] * inv_rms * ggml_fp16_to_fp32(w[d]);
    }
#elif defined(__aarch64__)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    uint32_t d = 0;
    for (; d + 4 <= dim; d += 4) {
        float32x4_t vx = vld1q_f32(x + d);
        vsum = vfmaq_f32(vsum, vx, vx);
    }
    double ss = (double)vaddvq_f32(vsum);
    for (; d < dim; ++d) ss += (double)x[d] * (double)x[d];
    const float inv_rms = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
    float32x4_t vinv = vdupq_n_f32(inv_rms);

    if (weight_tensor->type == GGML_TYPE_F32) {
        const float *w = (const float *)weight_tensor->data;
        d = 0;
        for (; d + 4 <= dim; d += 4) {
            float32x4_t vx = vld1q_f32(x + d);
            float32x4_t vw = vld1q_f32(w + d);
            vst1q_f32(out + d, vmulq_f32(vmulq_f32(vx, vinv), vw));
        }
        for (; d < dim; ++d) out[d] = x[d] * inv_rms * w[d];
    } else {
        const uint16_t *w = (const uint16_t *)weight_tensor->data;
        d = 0;
        for (; d + 4 <= dim; d += 4) {
            float32x4_t vx = vld1q_f32(x + d);
            float32x4_t vw = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(w + d)));
            vst1q_f32(out + d, vmulq_f32(vmulq_f32(vx, vinv), vw));
        }
        for (; d < dim; ++d) out[d] = x[d] * inv_rms * ggml_fp16_to_fp32(w[d]);
    }
#else
    double ss = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        ss += (double)x[i] * (double)x[i];
    }
    const float inv_rms = 1.0f / sqrtf((float)(ss / (double)dim) + eps);

    if (weight_tensor->type == GGML_TYPE_F32) {
        const float *w = (const float *)weight_tensor->data;
        for (uint32_t i = 0; i < dim; ++i) {
            out[i] = x[i] * inv_rms * w[i];
        }
    } else {
        const uint16_t *w = (const uint16_t *)weight_tensor->data;
        for (uint32_t i = 0; i < dim; ++i) {
            out[i] = x[i] * inv_rms * ggml_fp16_to_fp32(w[i]);
        }
    }
#endif
}

/* Fused RMSNorm + Q8_K Quantization: normalizes x into out and quantizes out into out_q8 */
static inline void lfm2_rmsnorm_and_quantize_q8_K(float *out, block_q8_K_t *out_q8, const float *x,
                                                  const GGUFTensor *weight_tensor, uint32_t dim, float eps) {
    lfm2_rmsnorm(out, x, weight_tensor, dim, eps);
    if (out_q8) {
#if LFM2_HAS_AVX2
        lfm2_quantize_row_q8_K_avx2(out, out_q8, dim);
#else
        lfm2_quantize_row_q8_K_fallback(out, out_q8, dim);
#endif
    }
}

/* Per-Head QK-Norm + NeoX Rotary Position Embedding (RoPE)
 * Uses precomputed cos/sin LUTs with AVX2 FMA rotation */
static void lfm2_qk_norm_and_rope(const LFM2State *s, float *vec, uint32_t n_heads, uint32_t head_dim,
                                  const GGUFTensor *norm_weight, float eps,
                                  uint32_t pos) {
    const uint32_t half_dim = head_dim / 2;
    const float *cos_table = (s && s->rope_cos) ? (s->rope_cos + (size_t)pos * half_dim) : NULL;
    const float *sin_table = (s && s->rope_sin) ? (s->rope_sin + (size_t)pos * half_dim) : NULL;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float *head = vec + (size_t)h * head_dim;

        /* 1. Per-Head RMSNorm (in-place on head[0 .. head_dim-1]) */
        if (norm_weight != NULL) {
            lfm2_rmsnorm(head, head, norm_weight, head_dim, eps);
        }

        /* 2. NeoX Half-Split RoPE: pairs (head[i], head[i + half_dim]) */
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

/* ============================================================================
 * 5. Operator 1: LFM2 LIV Double-Gated Short Convolution Block
 * ============================================================================ */
static void lfm2_forward_shortconv(LFM2State *s, const LFM2Model *m,
                                   const LFM2Layer *layer, uint32_t l_idx) {
    const uint32_t D       = m->cfg.n_embd;
    const uint32_t l_cache = m->cfg.shortconv_l_cache ? m->cfg.shortconv_l_cache : 3;
    const uint32_t hist    = (l_cache > 1) ? (l_cache - 1) : 2;

    /* 1. Fused input projection: xb [D] -> bcx [3 * D] */
    lfm2_matmul_vec(s->bcx, layer->shortconv.in_proj, s->xb, s->xq_scratch, &s->pool);

    /* Split bcx into 3 contiguous chunks: B, C, x_tilde */
    float *B       = s->bcx;
    float *C       = s->bcx + D;
    float *x_tilde = s->bcx + 2 * D;

    /* 2. First multiplicative gate: Bx[d] = B[d] * x_tilde[d] (stored in B) */
    uint32_t d = 0;
#if LFM2_HAS_AVX2
    for (; d + 8 <= D; d += 8) {
        __m256 vb = _mm256_loadu_ps(B + d);
        __m256 vx = _mm256_loadu_ps(x_tilde + d);
        _mm256_storeu_ps(B + d, _mm256_mul_ps(vb, vx));
    }
#elif defined(__aarch64__)
    for (; d + 4 <= D; d += 4) {
        float32x4_t vb = vld1q_f32(B + d);
        float32x4_t vx = vld1q_f32(x_tilde + d);
        vst1q_f32(B + d, vmulq_f32(vb, vx));
    }
#endif
    for (; d < D; ++d) {
        B[d] = B[d] * x_tilde[d];
    }

    /* 3. Causal 1D Depthwise Convolution across (l_cache - 1) state steps + current B[d] */
    float *state = s->conv_state[l_idx]; /* shape: [hist][D] */
    const float *w0 = s->conv_w[l_idx][0];
    const float *w1 = s->conv_w[l_idx][1];
    const float *w2 = s->conv_w[l_idx][2];

    if (w0 && w1 && w2 && hist == 2) {
        const float *s0 = state;
        const float *s1 = state + D;
        d = 0;
#if LFM2_HAS_AVX2
        for (; d + 8 <= D; d += 8) {
            __m256 vw0 = _mm256_loadu_ps(w0 + d);
            __m256 vs0 = _mm256_loadu_ps(s0 + d);
            __m256 vw1 = _mm256_loadu_ps(w1 + d);
            __m256 vs1 = _mm256_loadu_ps(s1 + d);
            __m256 vw2 = _mm256_loadu_ps(w2 + d);
            __m256 vb  = _mm256_loadu_ps(B + d);
            __m256 vc  = _mm256_loadu_ps(C + d);

            __m256 vacc = _mm256_fmadd_ps(vw0, vs0, _mm256_mul_ps(vw1, vs1));
            vacc = _mm256_fmadd_ps(vw2, vb, vacc);
            __m256 vy = _mm256_mul_ps(vc, vacc);
            _mm256_storeu_ps(s->conv_out + d, vy);
        }
#elif defined(__aarch64__)
        for (; d + 4 <= D; d += 4) {
            float32x4_t vw0 = vld1q_f32(w0 + d);
            float32x4_t vs0 = vld1q_f32(s0 + d);
            float32x4_t vw1 = vld1q_f32(w1 + d);
            float32x4_t vs1 = vld1q_f32(s1 + d);
            float32x4_t vw2 = vld1q_f32(w2 + d);
            float32x4_t vb  = vld1q_f32(B + d);
            float32x4_t vc  = vld1q_f32(C + d);

            float32x4_t vacc = vfmaq_f32(vmulq_f32(vw1, vs1), vw0, vs0);
            vacc = vfmaq_f32(vacc, vw2, vb);
            vst1q_f32(s->conv_out + d, vmulq_f32(vc, vacc));
        }
#endif
        for (; d < D; ++d) {
            float acc = w0[d] * s0[d] + w1[d] * s1[d] + w2[d] * B[d];
            s->conv_out[d] = C[d] * acc;
        }
    } else {
        const GGUFTensor *conv_t = layer->shortconv.conv;
        const bool stride_is_lcache = (conv_t->ne[0] == l_cache);
        for (uint32_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            if (conv_t->type == GGML_TYPE_F32) {
                const float *w_base = (const float *)conv_t->data;
                for (uint32_t k = 0; k < hist; ++k) {
                    float w_k = stride_is_lcache ? w_base[d * l_cache + k] : w_base[k * D + d];
                    acc += w_k * state[k * D + d];
                }
                float w_last = stride_is_lcache ? w_base[d * l_cache + hist] : w_base[hist * D + d];
                acc += w_last * B[d];
            } else {
                const uint16_t *w_base = (const uint16_t *)conv_t->data;
                for (uint32_t k = 0; k < hist; ++k) {
                    float w_k = stride_is_lcache ? ggml_fp16_to_fp32(w_base[d * l_cache + k]) : ggml_fp16_to_fp32(w_base[k * D + d]);
                    acc += w_k * state[k * D + d];
                }
                float w_last = stride_is_lcache ? ggml_fp16_to_fp32(w_base[d * l_cache + hist]) : ggml_fp16_to_fp32(w_base[hist * D + d]);
                acc += w_last * B[d];
            }
            s->conv_out[d] = C[d] * acc;
        }
    }

    /* 5. Update rolling ShortConv state buffer for the next token */
    if (hist > 1) {
        memmove(state, state + D, (size_t)(hist - 1) * D * sizeof(float));
    }
    memcpy(state + (size_t)(hist - 1) * D, B, D * sizeof(float));

    /* 6. Output linear projection: conv_out [D] -> xb [D], then add to residual x */
    lfm2_matmul_vec(s->xb, layer->shortconv.out_proj, s->conv_out, s->xq_scratch, &s->pool);
    lfm2_fwd_vec_add_f32(s->x, s->xb, D);
}

/* ============================================================================
 * 6. Operator 2: LFM2 Grouped-Query Attention (GQA) with Per-Head QK-Norm
 * ============================================================================ */
static void lfm2_forward_gqa(LFM2State *s, const LFM2Model *m,
                             const LFM2Layer *layer, uint32_t l_idx, uint32_t pos) {
    const uint32_t D         = m->cfg.n_embd;
    const uint32_t n_head_q  = m->cfg.n_head_q;
    const uint32_t n_head_kv = m->cfg.n_head_kv[l_idx];
    const uint32_t head_dim  = n_head_q > 0 ? (D / n_head_q) : 64;
    const uint32_t kv_dim    = n_head_kv * head_dim;
    const uint32_t kv_mul    = (n_head_kv > 0) ? (n_head_q / n_head_kv) : 1;
    const float    scale     = 1.0f / sqrtf((float)head_dim);

    /* 1. Q, K, V Linear Projections */
    lfm2_matmul_vec(s->q, layer->attn.wq, s->xb, s->xq_scratch, &s->pool);
    lfm2_matmul_vec(s->k, layer->attn.wk, s->xb, s->xq_scratch, &s->pool);
    lfm2_matmul_vec(s->v, layer->attn.wv, s->xb, s->xq_scratch, &s->pool);

    /* 2. Per-Head QK-Norm + NeoX RoPE */
    lfm2_qk_norm_and_rope(s, s->q, n_head_q,  head_dim, layer->attn.q_norm,
                          m->cfg.rms_norm_eps, pos);
    lfm2_qk_norm_and_rope(s, s->k, n_head_kv, head_dim, layer->attn.k_norm,
                          m->cfg.rms_norm_eps, pos);

    /* 3. Write current K and V into this layer's KV caches at step `pos` */
    if (s->k_cache[l_idx] && s->v_cache[l_idx]) {
        memcpy(s->k_cache[l_idx] + (size_t)pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->v_cache[l_idx] + (size_t)pos * kv_dim, s->v, kv_dim * sizeof(float));
    }
    if (s->k_cache_f16[l_idx] && s->v_cache_f16[l_idx]) {
        uint16_t *k_step = s->k_cache_f16[l_idx] + (size_t)pos * kv_dim;
        uint16_t *v_step = s->v_cache_f16[l_idx] + (size_t)pos * kv_dim;
        uint32_t i = 0;
#if defined(__F16C__)
        for (; i + 8 <= kv_dim; i += 8) {
            __m256 vk = _mm256_loadu_ps(s->k + i);
            __m256 vv = _mm256_loadu_ps(s->v + i);
            _mm_storeu_si128((__m128i *)(k_step + i), _mm256_cvtps_ph(vk, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm_storeu_si128((__m128i *)(v_step + i), _mm256_cvtps_ph(vv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
#elif defined(__aarch64__)
        for (; i + 8 <= kv_dim; i += 8) {
            float16x4_t hk0 = vcvt_f16_f32(vld1q_f32(s->k + i + 0));
            float16x4_t hk1 = vcvt_f16_f32(vld1q_f32(s->k + i + 4));
            float16x4_t hv0 = vcvt_f16_f32(vld1q_f32(s->v + i + 0));
            float16x4_t hv1 = vcvt_f16_f32(vld1q_f32(s->v + i + 4));
            vst1q_u16(k_step + i, vreinterpretq_u16_f16(vcombine_f16(hk0, hk1)));
            vst1q_u16(v_step + i, vreinterpretq_u16_f16(vcombine_f16(hv0, hv1)));
        }
#endif
        for (; i < kv_dim; ++i) {
            k_step[i] = lfm2_fp32_to_fp16(s->k[i]);
            v_step[i] = lfm2_fp32_to_fp16(s->v[i]);
        }
    }

    /* 4. Multi-Head Grouped-Query Scaled Dot-Product Attention */
    GQAAttnTaskCtx gqa_ctx = {
        .pool        = &s->pool,
        .q           = s->q,
        .k_cache     = s->k_cache[l_idx],
        .v_cache     = s->v_cache[l_idx],
        .k_cache_f16 = s->k_cache_f16[l_idx],
        .v_cache_f16 = s->v_cache_f16[l_idx],
        .att         = s->att,
        .xb          = s->xb,
        .n_head_q    = n_head_q,
        .kv_mul      = kv_mul,
        .head_dim    = head_dim,
        .kv_dim      = kv_dim,
        .max_seq_len = s->max_seq_len,
        .pos         = pos,
        .scale       = scale
    };
    lfm2_parallel_for(&s->pool, n_head_q, 1, lfm2_gqa_heads_worker_task, &gqa_ctx);

    /* 5. Output Projection (attn_output.weight) & Residual Connection */
    lfm2_matmul_vec(s->conv_out, layer->attn.wo, s->xb, s->xq_scratch, &s->pool);
    lfm2_fwd_vec_add_f32(s->x, s->conv_out, D);
}

/* ============================================================================
 * 7. Operator 3: SwiGLU Feed-Forward Network (MLP)
 * ============================================================================ */
static void lfm2_forward_ffn(LFM2State *s, const LFM2Model *m, const LFM2Layer *layer) {
    const uint32_t D    = m->cfg.n_embd;
    const uint32_t n_ff = (uint32_t)layer->ffn_gate->ne[1]; /* Exact per-layer FFN dim */

    /* 1. Fused FFN Pre-Norm & Q8_K Quantization: s->x -> s->xb & s->xq_scratch */
    lfm2_rmsnorm_and_quantize_q8_K(s->xb, s->xq_scratch, s->x, layer->ffn_norm, D, m->cfg.rms_norm_eps);

    /* 2 & 3. Gate & Up Projections + SwiGLU Non-Linearity */
#if LFM2_HAS_AVX2
    if ((layer->ffn_gate->type == GGML_TYPE_Q4_K || layer->ffn_gate->type == GGML_TYPE_Q6_K) &&
        (layer->ffn_gate->type == layer->ffn_up->type)) {
        lfm2_matmul_swiglu_avx2_mt(&s->pool, s->hb, layer->ffn_gate->data, layer->ffn_up->data,
                                   layer->ffn_gate->type, D, n_ff, s->xb, s->xq_scratch);
    } else {
        lfm2_matmul_vec(s->hb,  layer->ffn_gate, s->xb, s->xq_scratch, &s->pool);
        lfm2_matmul_vec(s->hb2, layer->ffn_up,   s->xb, s->xq_scratch, &s->pool);
        for (uint32_t i = 0; i < n_ff; ++i) {
            const float g = s->hb[i];
            const float silu = g / (1.0f + expf(-g));
            s->hb[i] = silu * s->hb2[i];
        }
    }
#else
    lfm2_matmul_vec(s->hb,  layer->ffn_gate, s->xb, s->xq_scratch, &s->pool);
    lfm2_matmul_vec(s->hb2, layer->ffn_up,   s->xb, s->xq_scratch, &s->pool);
    for (uint32_t i = 0; i < n_ff; ++i) {
        const float g = s->hb[i];
        const float silu = g / (1.0f + expf(-g));
        s->hb[i] = silu * s->hb2[i];
    }
#endif

    /* 4. Down Projection: hb [n_ff] -> xb [D] & Residual Addition */
    lfm2_matmul_vec(s->xb, layer->ffn_down, s->hb, s->xq_scratch, &s->pool);
    lfm2_fwd_vec_add_f32(s->x, s->xb, D);
}

/* ============================================================================
 * 8. Complete Single-Token Forward Pass Entry Point
 * ============================================================================ */
float *lfm2_forward_ex(const LFM2Model *m, LFM2State *s, uint32_t token_id, uint32_t pos, bool compute_logits) {
    const uint32_t D = m->cfg.n_embd;

    if (token_id >= m->cfg.n_vocab || pos >= s->max_seq_len) {
        fprintf(stderr, "[ERROR] token_id (%u) or pos (%u) out of bounds.\n", token_id, pos);
        return NULL;
    }

    /* 1. Dequantize token embedding row directly from mmap into residual stream s->x */
    lfm2_dequantize_row(m->token_embd, token_id, s->x);

    /* 2. Execute the 16+ Hybrid Layers (ShortConv + GQA Attention) */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];

        /* Operator Pre-Norm (blk.l.attn_norm.weight) */
        lfm2_rmsnorm(s->xb, s->x, layer->attn_norm, D, m->cfg.rms_norm_eps);

        /* Route to either LIV ShortConv or GQA Attention */
        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv(s, m, layer, l);
        } else {
            lfm2_forward_gqa(s, m, layer, l, pos);
        }

        /* SwiGLU Feed-Forward Block */
        lfm2_forward_ffn(s, m, layer);
    }

    /* Optimization 1: Skip LM Head & Output Norm during Prompt Prefill */
    if (!compute_logits) {
        return NULL;
    }

    /* 3. Final Output RMSNorm (token_embd_norm.weight) */
    lfm2_rmsnorm(s->x, s->x, m->output_norm, D, m->cfg.rms_norm_eps);

    /* 4. Classifier LM Head Projection (tied token_embd.weight or output.weight) */
    lfm2_matmul_vec(s->logits, m->output, s->x, s->xq_scratch, &s->pool);

    return s->logits;
}

float *lfm2_forward(const LFM2Model *m, LFM2State *s, uint32_t token_id, uint32_t pos) {
    return lfm2_forward_ex(m, s, token_id, pos, true);
}

#ifndef LFM2_FORWARD_NO_MAIN
/* ============================================================================
 * 9. Verification CLI: Runs a Multi-Token Greedy Decode Loop & Measures Speed
 * ============================================================================ */
static uint32_t sample_argmax(const float *logits, uint32_t n_vocab) {
    uint32_t best_id = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id  = i;
        }
    }
    return best_id;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-lfm2.gguf> [steps]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t steps = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;

    LFM2Model model;
    if (!lfm2_load_gguf(argv[1], &model)) {
        return EXIT_FAILURE;
    }

    LFM2State state;
    if (!lfm2_state_init(&state, &model, 512)) {
        fprintf(stderr, "Failed to allocate LFM2 runtime state arena.\n");
        lfm2_free_model(&model);
        return EXIT_FAILURE;
    }

    printf("Loaded %s (%u layers, %u embd, %u vocab). Running %u greedy steps...\n",
           model.cfg.arch, model.cfg.n_layers, model.cfg.n_embd, model.cfg.n_vocab, steps);

    /* Start with LFM2 BOS token (token_id = 1, <s>) */
    uint32_t token = 1;
    for (uint32_t pos = 0; pos < steps; ++pos) {
        float *logits = lfm2_forward(&model, &state, token, pos);
        if (!logits) break;

        uint32_t next_token = sample_argmax(logits, model.cfg.n_vocab);
        printf("Step %2u | in_token=%6u -> next_token=%6u | top_logit=%+.4f\n",
               pos, token, next_token, logits[next_token]);
        token = next_token;
    }

    lfm2_state_free(&state);
    lfm2_free_model(&model);
    return EXIT_SUCCESS;
}
#endif


#endif /* LFM2_FORWARD_INCLUDED */
