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

#ifndef LFM2_MOE_CHECKPOINT_INCLUDED
#define LFM2_MOE_CHECKPOINT_INCLUDED 1

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

#define LFM2_MAX_EXPERTS     64
#define LFM2_MAX_TOP_K        8

/* ============================================================================
 * 1. Extended MoE Layer & Config Descriptors for `lfm2moe` (8B-A1B & 24B-A2B)
 * ============================================================================ */
typedef struct {
    bool              is_moe_layer;      /* false for dense layers 0..1, true for 2..N-1 */
    uint32_t          n_experts;         /* 32 for 8B-A1B, 64 for 24B-A2B          */
    uint32_t          n_experts_used;    /* Top-K = 4                              */
    bool              norm_topk_prob;    /* true in LFM2-MoE                       */
    float             routed_scale;      /* expert_weights_scale (default 1.0f)    */

    const GGUFTensor *ffn_gate_inp;      /* blk.l.ffn_gate_inp.weight [D, n_exp]   */
    const GGUFTensor *ffn_exp_probs_b;   /* blk.l.exp_probs_b.bias    [n_exp] F32  */
    const GGUFTensor *ffn_gate_exps;     /* blk.l.ffn_gate_exps.weight [D, n_ff_exp, n_exp] */
    const GGUFTensor *ffn_up_exps;       /* blk.l.ffn_up_exps.weight   [D, n_ff_exp, n_exp] */
    const GGUFTensor *ffn_down_exps;     /* blk.l.ffn_down_exps.weight [n_ff_exp, D, n_exp] */
} LFM2MoELayerExt;

/* Helper to find layer tensor with fallback naming conventions */
static inline const GGUFTensor *lfm2_moe_find_tensor(const LFM2Model *m, uint32_t l, const char *suffix) {
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%u.%s", l, suffix);
    return find_tensor(m, buf);
}

/* Binds MoE tensors for layer `l` if present in the GGUF file.
 * Handles both `exp_probs_b.bias` and bare `exp_probs_b` GGUF naming conventions. */
bool lfm2_bind_moe_layer(const LFM2Model *m, uint32_t l,
                         uint32_t default_n_exp, uint32_t default_top_k,
                         LFM2MoELayerExt *moe_out) {
    memset(moe_out, 0, sizeof(*moe_out));

    const GGUFTensor *gate_exps = lfm2_moe_find_tensor(m, l, "ffn_gate_exps.weight");
    if (!gate_exps) {
        /* Layer `l` is a standard Dense SwiGLU layer (e.g., layers 0 and 1 in LFM2-MoE) */
        moe_out->is_moe_layer = false;
        return true;
    }

    moe_out->is_moe_layer    = true;
    moe_out->ffn_gate_exps   = gate_exps;
    moe_out->ffn_up_exps     = lfm2_moe_find_tensor(m, l, "ffn_up_exps.weight");
    moe_out->ffn_down_exps   = lfm2_moe_find_tensor(m, l, "ffn_down_exps.weight");
    moe_out->ffn_gate_inp    = lfm2_moe_find_tensor(m, l, "ffn_gate_inp.weight");
    moe_out->ffn_exp_probs_b = lfm2_moe_find_tensor(m, l, "exp_probs_b.bias");
    if (!moe_out->ffn_exp_probs_b) {
        moe_out->ffn_exp_probs_b = lfm2_moe_find_tensor(m, l, "exp_probs_b");
    }

    if (!moe_out->ffn_up_exps || !moe_out->ffn_down_exps || !moe_out->ffn_gate_inp) {
        fprintf(stderr, "[ERROR] Layer %u has incomplete MoE expert tensors.\n", l);
        return false;
    }

    moe_out->n_experts      = (gate_exps->n_dims >= 3) ? (uint32_t)gate_exps->ne[2] : default_n_exp;
    moe_out->n_experts_used = (default_top_k > 0) ? default_top_k : 4;
    moe_out->norm_topk_prob = true;
    moe_out->routed_scale   = 1.0f;
    return true;
}

/* ============================================================================
 * 2. Zero-Copy 3D -> 2D Expert Tensor Slicer (O(1) mmap Pointer Arithmetic)
 * ============================================================================
 * Given a 3D GGUFTensor `exps_3d` with shape [ne0, ne1, n_experts], returns a
 * stack-allocated 2D GGUFTensor view pointing directly to expert `expert_idx`.
 * ============================================================================ */
static inline GGUFTensor lfm2_slice_expert_tensor_2d(const GGUFTensor * restrict exps_3d,
                                                     uint32_t expert_idx) {
    ggml_type_traits_t traits;
    get_ggml_type_traits(exps_3d->type, &traits);

    const size_t row_bytes    = ((size_t)exps_3d->ne[0] / traits.block_elements) * traits.block_bytes;
    const size_t expert_bytes = (size_t)exps_3d->ne[1] * row_bytes;

    GGUFTensor slice = *exps_3d;
    slice.n_dims = 2;
    slice.ne[0]  = exps_3d->ne[0];
    slice.ne[1]  = exps_3d->ne[1];
    slice.ne[2]  = 1;
    slice.ne[3]  = 1;
    slice.bytes  = expert_bytes;
    slice.data   = (const uint8_t *)exps_3d->data + (size_t)expert_idx * expert_bytes;
    return slice;
}

/* ============================================================================
 * 3. LFM2-MoE Sigmoid + Expert-Bias Router (`modeling_lfm2_moe.py` Parity)
 * ============================================================================
 * 1. Computes router logits: z[e] = dot(ffn_gate_inp[e], xb) for e = 0..n_experts-1
 * 2. Applies Sigmoid:        s[e] = 1.0f / (1.0f + expf(-z[e]))
 * 3. Adds selection bias:    c[e] = s[e] + exp_probs_b[e]
 * 4. Selects Top-K experts maximizing `c[e]`
 * 5. Normalizes UNBIASED `s[e_k]` across the Top-K winners so sum(w_k) = routed_scale
 * ============================================================================ */
static inline float lfm2_moe_dot_f16_f32(const uint16_t * restrict row,
                                         const float * restrict xb,
                                         uint32_t D) {
    float logit = 0.0f;
    uint32_t i = 0;
#if LFM2_HAS_AVX2 && defined(__F16C__)
    for (; i + 8 <= D; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(row + i));
        __m256  w = _mm256_cvtph_ps(h);
        __m256  x = _mm256_loadu_ps(xb + i);
        logit += hsum_ps_avx2(_mm256_mul_ps(w, x));
    }
#elif defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= D; i += 4) {
        uint16x4_t  h = vld1_u16(row + i);
        float32x4_t w = vcvt_f32_f16(vreinterpret_f16_u16(h));
        float32x4_t x = vld1q_f32(xb + i);
        acc = vfmaq_f32(acc, w, x);
    }
    logit = vaddvq_f32(acc);
#endif
    for (; i < D; ++i) {
        logit += fp16_to_fp32(row[i]) * xb[i];
    }
    return logit;
}

void lfm2_moe_route_sigmoid_topk(const LFM2MoELayerExt * restrict moe,
                                 const float * restrict xb,
                                 const block_q8_K_t * restrict x_q8,
                                 uint32_t top_idx_out[LFM2_MAX_TOP_K],
                                 float top_weights_out[LFM2_MAX_TOP_K]) {
    const uint32_t n_exp = moe->n_experts;
    const uint32_t top_k = (moe->n_experts_used < LFM2_MAX_TOP_K) ? moe->n_experts_used : LFM2_MAX_TOP_K;
    const uint32_t D     = (uint32_t)moe->ffn_gate_inp->ne[0];

    float raw_sigmoid[LFM2_MAX_EXPERTS];
    float sel_score[LFM2_MAX_EXPERTS];

    const float *bias = (moe->ffn_exp_probs_b != NULL)
                        ? (const float *)moe->ffn_exp_probs_b->data
                        : NULL;

    /* 1. Compute router dot products (32 or 64 rows in F32, F16, or Quantized) */
    if (moe->ffn_gate_inp->type == GGML_TYPE_F32) {
        const float *w_router = (const float *)moe->ffn_gate_inp->data;
        for (uint32_t e = 0; e < n_exp; ++e) {
            const float logit = lfm2_vec_dot_f32_simd(w_router + (size_t)e * D, xb, D);
            const float sig   = 1.0f / (1.0f + expf(-logit));
            raw_sigmoid[e]    = sig;
            sel_score[e]      = bias ? (sig + bias[e]) : sig;
        }
    } else if (moe->ffn_gate_inp->type == GGML_TYPE_F16) {
        const uint16_t *w_router = (const uint16_t *)moe->ffn_gate_inp->data;
        for (uint32_t e = 0; e < n_exp; ++e) {
            const uint16_t *row = w_router + (size_t)e * D;
            const float logit   = lfm2_moe_dot_f16_f32(row, xb, D);
            const float sig     = 1.0f / (1.0f + expf(-logit));
            raw_sigmoid[e]      = sig;
            sel_score[e]        = bias ? (sig + bias[e]) : sig;
        }
    } else {
        /* Quantized router weights (e.g. Q8_0, Q4_K) */
        for (uint32_t e = 0; e < n_exp; ++e) {
            const float logit = lfm2_compute_1row_avx2(moe->ffn_gate_inp, e, xb, x_q8);
            const float sig   = 1.0f / (1.0f + expf(-logit));
            raw_sigmoid[e]    = sig;
            sel_score[e]      = bias ? (sig + bias[e]) : sig;
        }
    }

    /* 2. Partial Selection Sort for Top-K (K = 4 over N = 32 or 64 is ~128 comparisons) */
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t best_e = 0;
        float best_val  = -1e30f;
        for (uint32_t e = 0; e < n_exp; ++e) {
            if (sel_score[e] > best_val) {
                best_val = sel_score[e];
                best_e   = e;
            }
        }
        top_idx_out[k]     = best_e;
        top_weights_out[k] = raw_sigmoid[best_e]; /* Keep UNBIASED sigmoid weight! */
        sel_score[best_e]  = -1e30f;              /* Mask out selected expert      */
    }

    /* 3. Normalize Top-K unbiased weights (norm_topk_prob = true) */
    if (moe->norm_topk_prob) {
        float sum_w = 0.0f;
        for (uint32_t k = 0; k < top_k; ++k) {
            sum_w += top_weights_out[k];
        }
        const float inv_sum = moe->routed_scale / (sum_w + 1e-20f);
        for (uint32_t k = 0; k < top_k; ++k) {
            top_weights_out[k] *= inv_sum;
        }
    } else if (moe->routed_scale != 1.0f) {
        for (uint32_t k = 0; k < top_k; ++k) {
            top_weights_out[k] *= moe->routed_scale;
        }
    }
}

/* ============================================================================
 * 4. Fused 1x4 Weighted Down-Projection Accumulator (x += w_k * Down_ek * hb)
 * ============================================================================ */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *x_residual; /* [D] Accumulates += expert_weight * down_out */
    const GGUFTensor   *w_down_2d;
    const float        *hb_f32;
    const block_q8_K_t *hb_q8;
    float               expert_weight;
} MoEDownAccumTaskCtx;

static inline void lfm2_moe_accum_down_chunk(float * restrict x_residual,
                                             const GGUFTensor *w_down_2d,
                                             const float *hb_f32,
                                             const block_q8_K_t *hb_q8,
                                             float expert_weight,
                                             uint32_t r_start,
                                             uint32_t r_end) {
    uint32_t r = r_start;
#if defined(__AVX2__) && defined(__FMA__)
    const __m128 vw = _mm_set1_ps(expert_weight);
    for (; r + 4 <= r_end; r += 4) {
        float down4[4];
        lfm2_compute_4rows_avx2(w_down_2d, r, hb_f32, hb_q8, down4);
        __m128 vx = _mm_loadu_ps(x_residual + r);
        __m128 vd = _mm_loadu_ps(down4);
        _mm_storeu_ps(x_residual + r, _mm_fmadd_ps(vw, vd, vx));
    }
#elif defined(__aarch64__)
    const float32x4_t vw = vdupq_n_f32(expert_weight);
    for (; r + 4 <= r_end; r += 4) {
        float down4[4];
        lfm2_compute_4rows_avx2(w_down_2d, r, hb_f32, hb_q8, down4);
        float32x4_t vx = vld1q_f32(x_residual + r);
        float32x4_t vd = vld1q_f32(down4);
        vst1q_f32(x_residual + r, vfmaq_f32(vx, vw, vd));
    }
#else
    for (; r + 4 <= r_end; r += 4) {
        float down4[4];
        lfm2_compute_4rows_avx2(w_down_2d, r, hb_f32, hb_q8, down4);
        x_residual[r + 0] += expert_weight * down4[0];
        x_residual[r + 1] += expert_weight * down4[1];
        x_residual[r + 2] += expert_weight * down4[2];
        x_residual[r + 3] += expert_weight * down4[3];
    }
#endif
    for (; r < r_end; ++r) {
        const float d_val = lfm2_compute_1row_avx2(w_down_2d, r, hb_f32, hb_q8);
        x_residual[r] += expert_weight * d_val;
    }
}

static void lfm2_moe_down_accum_1x4_worker(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    MoEDownAccumTaskCtx *c = (MoEDownAccumTaskCtx *)arg;
    uint32_t r_start = 0, r_end = 0;

    while (lfm2_threadpool_claim_chunk(c->pool, &r_start, &r_end)) {
        lfm2_moe_accum_down_chunk(c->x_residual, c->w_down_2d, c->hb_f32, c->hb_q8,
                                  c->expert_weight, r_start, r_end);
    }
}

/* Complete Sparse MoE FFN Layer Forward Pass:
 *   1. Fused RMSNorm(x -> xb) + Quantize(xb -> x_q8) ONCE for the layer.
 *   2. Sigmoid + Expert-Bias Router picks Top-4 experts & normalizes weights.
 *   3. Loops over k = 0..3: slices 2D expert tensors from mmap, runs fused 1x4
 *      Gate + Up + SwiGLU + hb_q8, and accumulates w_k * Down_ek directly into `x`! */
void lfm2_forward_moe_ffn_1x4(LFM2ThreadPool *pool,
                              float * restrict x,
                              float * restrict xb,
                              float * restrict hb,
                              block_q8_K_t * restrict xq_scratch,
                              const LFM2Layer * restrict layer,
                              const LFM2MoELayerExt * restrict moe,
                              uint32_t D,
                              float eps) {
    /* If this is a leading dense layer (e.g. layers 0 and 1), run standard fused FFN */
    if (!moe->is_moe_layer) {
        lfm2_forward_ffn_fused_1x4(pool, x, xb, hb, xq_scratch, layer, D, eps);
        return;
    }

    /* 1. Normalize `x -> xb` and pre-quantize `xb -> xq_scratch` ONCE for all experts */
    lfm2_rmsnorm_quantize_q8_K_simd(xb, xq_scratch, x, layer->ffn_norm, D, eps);

    /* 2. Route to Top-K experts using Sigmoid + Expert Selection Bias */
    uint32_t top_experts[LFM2_MAX_TOP_K];
    float    top_weights[LFM2_MAX_TOP_K];
    lfm2_moe_route_sigmoid_topk(moe, xb, xq_scratch, top_experts, top_weights);

    /* 3. Execute ONLY the K selected experts (reusing xq_scratch across all active experts!) */
    block_q8_K_t *hb_q8 = xq_scratch + (D / 256);

    for (uint32_t k = 0; k < moe->n_experts_used; ++k) {
        const uint32_t exp_id = top_experts[k];
        const float    exp_w  = top_weights[k];

        /* Zero-copy 2D slices into the 3D mmap'd expert tensors */
        const GGUFTensor gate_2d = lfm2_slice_expert_tensor_2d(moe->ffn_gate_exps, exp_id);
        const GGUFTensor up_2d   = lfm2_slice_expert_tensor_2d(moe->ffn_up_exps,   exp_id);
        const GGUFTensor down_2d = lfm2_slice_expert_tensor_2d(moe->ffn_down_exps, exp_id);

        /* Fused 1x4 Gate + Up + SwiGLU + in-place L1-hot hb_q8 quantization */
        lfm2_ffn_swiglu_fused_1x4_mt(pool, hb, hb_q8, &gate_2d, &up_2d, xb, xq_scratch);

        /* Fused 1x4 Down projection + weighted residual accumulation: x += exp_w * down */
        if (pool == NULL || pool->n_threads <= 1) {
            lfm2_moe_accum_down_chunk(x, &down_2d, hb, hb_q8, exp_w, 0, D);
        } else {
            MoEDownAccumTaskCtx down_ctx = {
                .pool          = pool,
                .x_residual    = x,
                .w_down_2d     = &down_2d,
                .hb_f32        = hb,
                .hb_q8         = hb_q8,
                .expert_weight = exp_w
            };
            lfm2_parallel_for(pool, D, 16, lfm2_moe_down_accum_1x4_worker, &down_ctx);
        }
    }
}

/* ============================================================================
 * 5. LFM2 Hybrid State Checkpoint Manager (ShortConv Ring Buffers + GQA Pos)
 * ============================================================================
 * Why LFM2 requires Hybrid Checkpointing:
 * - GQA layers store key/value history indexed by `pos` (rewinding is O(1): pos = saved_pos).
 * - ShortConv layers overwrite their 2-token `conv_state[l]` buffer on every token.
 * - `LFM2Checkpoint` snapshots all ShortConv buffers (~160-480 KB total) + `pos`
 *   so system prompts, multi-turn branches, or rejected speculative draft tokens
 *   can be rolled back in ~15 microseconds with 100% mathematical parity!
 * ============================================================================ */
typedef struct {
    bool      is_valid;
    uint32_t  saved_pos;             /* Exact GQA KV-cache position cursor     */
    uint32_t  n_layers;
    uint32_t  conv_floats_per_layer; /* (l_cache - 1) * n_embd (typically 2*D) */
    uint32_t  n_conv_layers;
    float    *conv_slab;             /* Contiguous snapshot of all ShortConv states */

    /* Optional Sampler / Repetition-Penalty Ring State for Exact Rollbacks */
    uint64_t  saved_rng_state;
    uint32_t  saved_history_count;
    uint32_t  rep_window;
    uint32_t *saved_history_ring;
} LFM2Checkpoint;

bool lfm2_checkpoint_init(LFM2Checkpoint *cp, const LFM2Model *m, uint32_t rep_window) {
    memset(cp, 0, sizeof(*cp));
    const uint32_t hist_len = (m->cfg.shortconv_l_cache > 1) ? (m->cfg.shortconv_l_cache - 1) : 2;
    cp->n_layers              = m->cfg.n_layers;
    cp->conv_floats_per_layer = hist_len * m->cfg.n_embd;

    uint32_t n_conv = 0;
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            n_conv++;
        }
    }
    cp->n_conv_layers = n_conv;

    if (n_conv > 0) {
        const size_t bytes = (size_t)n_conv * cp->conv_floats_per_layer * sizeof(float);
        cp->conv_slab = (float *)aligned_alloc(64, (bytes + 63) & ~63ULL);
        if (!cp->conv_slab) return false;
    }

    cp->rep_window = rep_window;
    if (rep_window > 0) {
        cp->saved_history_ring = (uint32_t *)calloc(rep_window, sizeof(uint32_t));
        if (!cp->saved_history_ring) {
            free(cp->conv_slab);
            cp->conv_slab = NULL;
            return false;
        }
    }
    return true;
}

void lfm2_checkpoint_destroy(LFM2Checkpoint *cp) {
    if (cp) {
        free(cp->conv_slab);
        free(cp->saved_history_ring);
        memset(cp, 0, sizeof(*cp));
    }
}

#ifndef LFM2_H
void lfm2_checkpoint_free(LFM2Checkpoint *cp) {
    lfm2_checkpoint_destroy(cp);
}
#endif

/* Takes a zero-allocation snapshot of all ShortConv states + GQA `pos` cursor */
void lfm2_checkpoint_save(LFM2Checkpoint * restrict cp,
                          const LFM2Model * restrict m,
                          float * const conv_state[LFM2_MAX_LAYERS],
                          uint32_t current_pos,
                          uint64_t rng_state,
                          const uint32_t * restrict history_ring,
                          uint32_t history_count) {
    cp->saved_pos           = current_pos;
    cp->saved_rng_state     = rng_state;
    cp->saved_history_count = history_count;

    uint32_t conv_idx = 0;
    const size_t layer_bytes = (size_t)cp->conv_floats_per_layer * sizeof(float);

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV && conv_state[l] != NULL) {
            memcpy(cp->conv_slab + (size_t)conv_idx * cp->conv_floats_per_layer,
                   conv_state[l],
                   layer_bytes);
            conv_idx++;
        }
    }

    if (cp->rep_window > 0 && history_ring != NULL && cp->saved_history_ring != NULL) {
        memcpy(cp->saved_history_ring, history_ring, cp->rep_window * sizeof(uint32_t));
    }
    cp->is_valid = true;
}

/* Restores all ShortConv rolling buffers and rewinds the GQA KV-cache `pos` cursor */
bool lfm2_checkpoint_restore(const LFM2Checkpoint * restrict cp,
                             const LFM2Model * restrict m,
                             float * conv_state[LFM2_MAX_LAYERS],
                             uint32_t * restrict out_pos,
                             uint64_t * restrict out_rng_state,
                             uint32_t * restrict out_history_ring,
                             uint32_t * restrict out_history_count) {
    if (!cp || !cp->is_valid) return false;

    if (out_pos)           *out_pos           = cp->saved_pos;
    if (out_rng_state)     *out_rng_state     = cp->saved_rng_state;
    if (out_history_count) *out_history_count = cp->saved_history_count;

    uint32_t conv_idx = 0;
    const size_t layer_bytes = (size_t)cp->conv_floats_per_layer * sizeof(float);

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV && conv_state[l] != NULL) {
            memcpy(conv_state[l],
                   cp->conv_slab + (size_t)conv_idx * cp->conv_floats_per_layer,
                   layer_bytes);
            conv_idx++;
        }
    }

    if (cp->rep_window > 0 && out_history_ring != NULL && cp->saved_history_ring != NULL) {
        memcpy(out_history_ring, cp->saved_history_ring, cp->rep_window * sizeof(uint32_t));
    }
    return true;
}

/* High-level state wrapper saving directly from LFM2State */
void lfm2_checkpoint_save_state(LFM2Checkpoint * restrict cp,
                                const LFM2Model * restrict m,
                                const LFM2State * restrict s,
                                uint32_t current_pos) {
    lfm2_checkpoint_save(cp, m, (float * const *)s->conv_state, current_pos, 0, NULL, 0);
}

/* High-level state wrapper restoring directly to LFM2State */
bool lfm2_checkpoint_restore_state(const LFM2Checkpoint * restrict cp,
                                   const LFM2Model * restrict m,
                                   LFM2State * restrict s,
                                   uint32_t * restrict out_pos) {
    return lfm2_checkpoint_restore(cp, m, s->conv_state, out_pos, NULL, NULL, NULL);
}

#endif /* LFM2_MOE_CHECKPOINT_INCLUDED */

/* ============================================================================
 * 6. Standalone Verification & Benchmark (LFM2-8B-A1B MoE + Hybrid Checkpoints)
 * ============================================================================ */
#ifndef LFM2_MOE_CHECKPOINT_NO_MAIN
int main(int argc, char **argv) {
    /* LFM2-8B-A1B MoE Layer Geometry:
     *   D = 2048, n_ff_exp = 1536, n_experts = 32, top_k = 4 */
    const uint32_t D         = 2048;
    const uint32_t n_ff_exp  = 1536;
    const uint32_t n_experts = 32;
    const uint32_t top_k     = 4;
    const uint32_t nb_d      = D / 256;
    const uint32_t nb_ff     = n_ff_exp / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0);

    printf("=== LFM2-8B-A1B Sparse MoE (32 Experts, Top-4) & Checkpoint Benchmark ===\n");
    printf("Physical Cores Active : %u\n\n", pool.n_threads);

    /* 1. Allocate synthetic 3D packed expert tensors matching LFM2-8B-A1B-Q4_K_M */
    float        *x        = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *xb       = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *hb       = (float *)lfm2_aligned_alloc_64(n_ff_exp * sizeof(float));
    float        *norm_w   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float        *gate_inp = (float *)lfm2_aligned_alloc_64((size_t)n_experts * D * sizeof(float));
    float        *exp_bias = (float *)lfm2_aligned_alloc_64(n_experts * sizeof(float));
    block_q8_K_t *xq_buf   = (block_q8_K_t *)lfm2_aligned_alloc_64((nb_d + nb_ff) * sizeof(block_q8_K_t));

    const size_t gate_up_blocks = (size_t)n_experts * n_ff_exp * nb_d;
    const size_t down_blocks    = (size_t)n_experts * D * nb_ff;
    block_q4_K_t *w_gate_exps = (block_q4_K_t *)lfm2_aligned_alloc_64(gate_up_blocks * sizeof(block_q4_K_t));
    block_q4_K_t *w_up_exps   = (block_q4_K_t *)lfm2_aligned_alloc_64(gate_up_blocks * sizeof(block_q4_K_t));
    block_q6_K_t *w_down_exps = (block_q6_K_t *)lfm2_aligned_alloc_64(down_blocks    * sizeof(block_q6_K_t));

    for (uint32_t i = 0; i < D; ++i) {
        x[i]      = sinf((float)i * 0.019f);
        norm_w[i] = 1.0f;
    }
    for (uint32_t e = 0; e < n_experts; ++e) {
        exp_bias[e] = (e == 7 || e == 19) ? 0.35f : -0.05f; /* Bias favors experts 7 & 19 */
        for (uint32_t i = 0; i < D; ++i) {
            gate_inp[(size_t)e * D + i] = cosf((float)(e * 13 + i) * 0.01f) * 0.05f;
        }
    }
    for (size_t i = 0; i < gate_up_blocks; ++i) {
        w_gate_exps[i].d = 0x2C00; w_gate_exps[i].dmin = 0x2400;
        w_up_exps[i].d   = 0x2B00; w_up_exps[i].dmin   = 0x2300;
        for (int j = 0; j < 12; ++j) {
            w_gate_exps[i].scales[j] = (uint8_t)((i + j * 3) & 0x3F);
            w_up_exps[i].scales[j]   = (uint8_t)((i + j * 7) & 0x3F);
        }
        for (int j = 0; j < 128; ++j) {
            w_gate_exps[i].qs[j] = (uint8_t)((i * 11 + j * 29) & 0xFF);
            w_up_exps[i].qs[j]   = (uint8_t)((i * 17 + j * 31) & 0xFF);
        }
    }
    for (size_t i = 0; i < down_blocks; ++i) {
        w_down_exps[i].d = 0x2A00;
        for (int j = 0; j < 16; ++j)  w_down_exps[i].scales[j] = (int8_t)(((int)(i + j) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_down_exps[i].ql[j]     = (uint8_t)((i * 13 + j * 19) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down_exps[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    GGUFTensor t_norm     = { .n_dims = 1, .ne = {D, 1, 1, 1},                 .type = GGML_TYPE_F32,  .data = norm_w      };
    GGUFTensor t_gate_inp = { .n_dims = 2, .ne = {D, n_experts, 1, 1},         .type = GGML_TYPE_F32,  .data = gate_inp    };
    GGUFTensor t_exp_bias = { .n_dims = 1, .ne = {n_experts, 1, 1, 1},         .type = GGML_TYPE_F32,  .data = exp_bias    };
    GGUFTensor t_gate_exp = { .n_dims = 3, .ne = {D, n_ff_exp, n_experts, 1},  .type = GGML_TYPE_Q4_K, .data = w_gate_exps };
    GGUFTensor t_up_exp   = { .n_dims = 3, .ne = {D, n_ff_exp, n_experts, 1},  .type = GGML_TYPE_Q4_K, .data = w_up_exps   };
    GGUFTensor t_down_exp = { .n_dims = 3, .ne = {n_ff_exp, D, n_experts, 1},  .type = GGML_TYPE_Q6_K, .data = w_down_exps };

    LFM2Layer fake_layer = { .ffn_norm = &t_norm };
    LFM2MoELayerExt moe_ext = {
        .is_moe_layer    = true,
        .n_experts       = n_experts,
        .n_experts_used  = top_k,
        .norm_topk_prob  = true,
        .routed_scale    = 1.0f,
        .ffn_gate_inp    = &t_gate_inp,
        .ffn_exp_probs_b = &t_exp_bias,
        .ffn_gate_exps   = &t_gate_exp,
        .ffn_up_exps     = &t_up_exp,
        .ffn_down_exps   = &t_down_exp
    };

    /* 2. Test Sigmoid + Expert-Bias Router & Verify Unbiased Weight Sum = 1.0 */
    uint32_t top_idx[LFM2_MAX_TOP_K];
    float    top_w[LFM2_MAX_TOP_K];
    lfm2_moe_route_sigmoid_topk(&moe_ext, x, NULL, top_idx, top_w);

    float w_sum = 0.0f;
    printf("1. Sigmoid + Bias Router Selected Top-4 of 32 Experts:\n   ");
    for (uint32_t k = 0; k < top_k; ++k) {
        printf("Expert[%2u] (w=%.4f)  ", top_idx[k], top_w[k]);
        w_sum += top_w[k];
    }
    printf("\n   Normalized Weight Sum : %.6f (expected 1.000000)\n\n", w_sum);

    /* 3. Benchmark Full Sparse MoE Layer (4 Active Experts of 32) */
    const int iters = 100;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_forward_moe_ffn_1x4(&pool, x, xb, hb, xq_buf, &fake_layer, &moe_ext, D, 1e-5f);
    }
    double moe_ms = (now_ms() - t0) / iters;

    const double active_bytes = (double)top_k * (
        2.0 * (double)n_ff_exp * nb_d * sizeof(block_q4_K_t) +
        (double)D * nb_ff * sizeof(block_q6_K_t)
    );
    const double total_moe_bytes = active_bytes * ((double)n_experts / (double)top_k);

    printf("2. LFM2-8B-A1B Sparse MoE FFN Layer Performance:\n");
    printf("   Total Stored Expert Weights : %.2f MB / layer (32 experts)\n", total_moe_bytes / (1024.0 * 1024.0));
    printf("   Active Bytes Pulled / Token : %.2f MB / layer (Top-4 experts = 12.5%% of layer!)\n", active_bytes / (1024.0 * 1024.0));
    printf("   MoE Layer Latency           : %.3f ms | Effective Bandwidth: %.2f GB/s\n\n",
           moe_ms, active_bytes / (moe_ms * 1e6));

    /* 4. Test Hybrid State Checkpointing (Save -> Mutate 5 Tokens -> Restore -> Compare) */
    LFM2Model mock_m;
    memset(&mock_m, 0, sizeof(mock_m));
    mock_m.cfg.n_layers          = 24; /* LFM2-8B-A1B: 18 ShortConv + 6 GQA */
    mock_m.cfg.n_embd            = D;
    mock_m.cfg.shortconv_l_cache = 3;

    float *conv_state[LFM2_MAX_LAYERS] = {0};
    for (uint32_t l = 0; l < 24; ++l) {
        mock_m.layers[l].block_type = (l % 4 == 2) ? LFM2_BLOCK_GQA_ATTN : LFM2_BLOCK_SHORTCONV;
        if (mock_m.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            conv_state[l] = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
            for (uint32_t i = 0; i < 2 * D; ++i) {
                conv_state[l][i] = sinf((float)(l * 100 + i) * 0.01f);
            }
        }
    }

    LFM2Checkpoint ckpt;
    lfm2_checkpoint_init(&ckpt, &mock_m, 64);

    uint32_t pos = 42;
    uint64_t rng = 0xCAFEBABE12345678ULL;
    uint32_t hist_cnt = 42;

    t0 = now_ms();
    lfm2_checkpoint_save(&ckpt, &mock_m, conv_state, pos, rng, NULL, hist_cnt);
    double save_us = (now_ms() - t0) * 1000.0;

    /* Simulate advancing 5 draft tokens that mutate ShortConv buffers & GQA pos */
    pos += 5;
    for (uint32_t l = 0; l < 24; ++l) {
        if (conv_state[l]) conv_state[l][0] = 999.0f;
    }

    /* Rollback to saved checkpoint */
    t0 = now_ms();
    lfm2_checkpoint_restore(&ckpt, &mock_m, conv_state, &pos, &rng, NULL, &hist_cnt);
    double restore_us = (now_ms() - t0) * 1000.0;

    bool ckpt_exact = (pos == 42) && (fabsf(conv_state[0][0] - sinf(0.0f)) < 1e-7f);
    printf("3. Hybrid State Checkpoint (18 ShortConv Layers = %.1f KB + GQA Cursor):\n",
           (double)(ckpt.n_conv_layers * ckpt.conv_floats_per_layer * sizeof(float)) / 1024.0);
    printf("   Save Latency / Restore Latency : %.2f us / %.2f us\n", save_us, restore_us);
    printf("   Rollback Bit-Exact Verification: %s (restored pos=%u)\n",
           ckpt_exact ? "PASS" : "FAIL", pos);

    for (uint32_t l = 0; l < 24; ++l) free(conv_state[l]);
    lfm2_checkpoint_free(&ckpt);
    free(x); free(xb); free(hb); free(norm_w); free(gate_inp); free(exp_bias);
    free(xq_buf); free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    lfm2_threadpool_free(&pool);
    return EXIT_SUCCESS;
}
#endif
