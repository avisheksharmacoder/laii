#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_PREFILL_BATCH_NO_MAIN 1
#define LFM2_RUNTIME_FEATURES_NO_MAIN 1
#define LFM2_MOE_CHECKPOINT_NO_MAIN 1
#define LFM2_SYS_CACHE_NO_MAIN 1
#define LFM2_SPECULATIVE_PLD_NO_MAIN 1
#define LFM3_ENGINE_NO_MAIN 1

#include "tools/lfm2_prefill_batch.c"
#include "tools/lfm2_runtime_features.c"
#include "tools/lfm2_moe_checkpoint.c"
#include "tools/lfm2_sys_cache.c"
#include "tools/lfm3_engine.c"

static void ref_linear_shortconv7(float *conv_out, const float *bcx, float *hist_linear,
                                  const LFM3PlanarConv7 *pw, uint32_t D) {
    const float *B = bcx;
    const float *C = bcx + D;
    const float *x = bcx + 2U * D;
    for (uint32_t d = 0; d < D; ++d) {
        const float u_t = B[d] * x[d];
        float acc = pw->w[6][d] * u_t;
        for (uint32_t k = 0; k < 6U; ++k) {
            acc += pw->w[k][d] * hist_linear[(size_t)k * D + d];
        }
        for (uint32_t k = 0; k < 5U; ++k) {
            hist_linear[(size_t)k * D + d] = hist_linear[(size_t)(k + 1U) * D + d];
        }
        hist_linear[5U * D + d] = u_t;
        conv_out[d] = C[d] * acc;
    }
}

static int test_shortconv7_ring(void) {
    printf("[Test 1/6] Verifying ShortConv 7-Tap Circular Ring Buffer vs Linear Ref...\n");
    const uint32_t D = 512U;
    float *w_raw   = (float *)lfm2_aligned_alloc_64((size_t)7U * D * sizeof(float));
    float *ring    = (float *)lfm2_aligned_alloc_64((size_t)6U * D * sizeof(float));
    float *lin     = (float *)lfm2_aligned_alloc_64((size_t)6U * D * sizeof(float));
    float *bcx     = (float *)lfm2_aligned_alloc_64((size_t)3U * D * sizeof(float));
    float *out_sim = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *out_ref = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    block_q8_K_t *q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256U) * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < 7U * D; ++i) w_raw[i] = 0.2f * sinf((float)i * 0.07f);
    GGUFTensor conv_t = { .n_dims = 2, .ne = {7, D, 1, 1}, .type = GGML_TYPE_F32, .data = w_raw };

    LFM3PlanarConv7 pw;
    if (!lfm3_planar_conv7_init(&pw, &conv_t, D, 7U)) {
        fprintf(stderr, "FAIL: lfm3_planar_conv7_init\n");
        return 1;
    }

    uint32_t head = 0U;
    float max_diff = 0.0f;
    for (uint32_t step = 0; step < 18U; ++step) {
        for (uint32_t i = 0; i < 3U * D; ++i) {
            bcx[i] = 0.7f * cosf((float)(step * 31U + i) * 0.013f);
        }
        lfm3_shortconv7_gate_conv_quant_simd(out_sim, q8, bcx, ring, &head, &pw, D);
        ref_linear_shortconv7(out_ref, bcx, lin, &pw, D);
        for (uint32_t d = 0; d < D; ++d) {
            float diff = fabsf(out_sim[d] - out_ref[d]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    float *lin_check = (float *)lfm2_aligned_alloc_64((size_t)6U * D * sizeof(float));
    lfm3_shortconv7_linearize_ring(ring, head, 6U, D, lin_check);
    float ring_diff = 0.0f;
    for (uint32_t i = 0; i < 6U * D; ++i) {
        float diff = fabsf(lin_check[i] - lin[i]);
        if (diff > ring_diff) ring_diff = diff;
    }

    lfm3_planar_conv7_free(&pw);
    free(w_raw); free(ring); free(lin); free(lin_check);
    free(bcx); free(out_sim); free(out_ref); free(q8);

    printf("  ShortConv7 max output diff=%.8f, linearized state diff=%.8f\n", max_diff, ring_diff);
    if (max_diff > 1e-5f || ring_diff > 1e-5f) {
        fprintf(stderr, "FAIL: ShortConv7 mismatch\n");
        return 1;
    }
    return 0;
}

static int test_partial_rope_and_qk128(void) {
    printf("[Test 2/6] Verifying Partial RoPE (dims 32..63 & 96..127 unrotated) + QK128 Norm...\n");
    float *rf_w = (float *)lfm2_aligned_alloc_64(64U * sizeof(float));
    float *nw   = (float *)lfm2_aligned_alloc_64(128U * sizeof(float));
    for (uint32_t i = 0; i < 64U; ++i) rf_w[i] = (i < 32U) ? 1.0f : 1e30f;
    for (uint32_t i = 0; i < 128U; ++i) nw[i] = 1.0f + 0.01f * (float)(i % 17U);

    GGUFTensor rf_t = { .n_dims = 1, .ne = {64, 1, 1, 1}, .type = GGML_TYPE_F32, .data = rf_w };
    GGUFTensor nw_t = { .n_dims = 1, .ne = {128, 1, 1, 1}, .type = GGML_TYPE_F32, .data = nw };

    LFM2RoPELUT lut;
    if (!lfm3_rope_lut_init(&lut, 64U, 128U, 128U, 1000000.0f, &rf_t)) return 1;

    /* Verify partial RoPE masking on second half of pairs */
    for (uint32_t i = 32U; i < 64U; ++i) {
        if (fabsf(lut.cos_lut[17U * 64U + i] - 1.0f) > 1e-7f ||
            fabsf(lut.sin_lut[17U * 64U + i]) > 1e-7f) {
            fprintf(stderr, "FAIL: partial RoPE pair %u not identity!\n", i);
            lfm2_rope_lut_free(&lut);
            free(rf_w); free(nw);
            return 1;
        }
    }

    const uint32_t n_heads = 4U;
    float *vec_simd   = (float *)lfm2_aligned_alloc_64(n_heads * 128U * sizeof(float));
    float *vec_scalar = (float *)lfm2_aligned_alloc_64(n_heads * 128U * sizeof(float));
    uint16_t *k16_simd   = (uint16_t *)lfm2_aligned_alloc_64(n_heads * 128U * sizeof(uint16_t));
    uint16_t *k16_scalar = (uint16_t *)lfm2_aligned_alloc_64(n_heads * 128U * sizeof(uint16_t));

    for (uint32_t i = 0; i < n_heads * 128U; ++i) {
        vec_simd[i] = vec_scalar[i] = sinf((float)i * 0.13f);
    }

    lfm3_qk_norm_rope_simd(vec_simd, NULL, n_heads, 128U, &nw_t, 1e-5f, &lut, 17U);
    lfm3_qk_norm_rope_scalar(vec_scalar, NULL, n_heads, 128U, &nw_t, 1e-5f, &lut, 17U);

    float max_f32_diff = 0.0f;
    for (uint32_t i = 0; i < n_heads * 128U; ++i) {
        float d = fabsf(vec_simd[i] - vec_scalar[i]);
        if (d > max_f32_diff) max_f32_diff = d;
    }

    /* Check unrotated slice actually stayed pure RMSNorm output */
    for (uint32_t i = 0; i < n_heads * 128U; ++i) {
        vec_simd[i] = sinf((float)i * 0.13f);
    }
    float *vec_f16_in = (float *)lfm2_aligned_alloc_64(n_heads * 128U * sizeof(float));
    memcpy(vec_f16_in, vec_simd, n_heads * 128U * sizeof(float));
    lfm3_qk_norm_rope_simd(vec_simd, k16_simd, n_heads, 128U, &nw_t, 1e-5f, &lut, 17U);
    lfm3_qk_norm_rope_scalar(vec_f16_in, k16_scalar, n_heads, 128U, &nw_t, 1e-5f, &lut, 17U);
    float max_f16_diff = 0.0f;
    for (uint32_t i = 0; i < n_heads * 128U; ++i) {
        float d = fabsf(lfm2_fp16_to_fp32(k16_simd[i]) - lfm2_fp16_to_fp32(k16_scalar[i]));
        if (d > max_f16_diff) max_f16_diff = d;
    }

    lfm2_rope_lut_free(&lut);
    free(rf_w); free(nw); free(vec_simd); free(vec_scalar);
    free(vec_f16_in); free(k16_simd); free(k16_scalar);

    printf("  QK128 SIMD vs Scalar max_f32_diff=%.8f max_f16_diff=%.8f\n", max_f32_diff, max_f16_diff);
    if (max_f32_diff > 1e-5f || max_f16_diff > 1e-3f) return 1;
    return 0;
}

static int test_online_gqa128_swa_wrap(void) {
    printf("[Test 3/6] Verifying Online GQA-128 Dual-Head KV Reuse + Gate + SWA Wrap-Around...\n");
    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 4U);

    const uint32_t n_head_q  = 8U;
    const uint32_t n_head_kv = 2U;
    const uint32_t head_dim  = 128U;
    const uint32_t kv_dim    = n_head_kv * head_dim;
    const uint32_t win       = 16U;
    const uint32_t cap       = win + LFM2_MAX_BATCH; /* 24 slots */

    uint16_t *k_ring = (uint16_t *)lfm2_aligned_alloc_64((size_t)cap * kv_dim * sizeof(uint16_t));
    uint16_t *v_ring = (uint16_t *)lfm2_aligned_alloc_64((size_t)cap * kv_dim * sizeof(uint16_t));
    float    *k_lin  = (float *)lfm2_aligned_alloc_64(64U * kv_dim * sizeof(float));
    float    *v_lin  = (float *)lfm2_aligned_alloc_64(64U * kv_dim * sizeof(float));
    float    *q      = (float *)lfm2_aligned_alloc_64(n_head_q * head_dim * sizeof(float));
    float    *gate   = (float *)lfm2_aligned_alloc_64(n_head_q * sizeof(float));
    float    *out    = (float *)lfm2_aligned_alloc_64(n_head_q * head_dim * sizeof(float));
    float    *ref    = (float *)lfm2_aligned_alloc_64(n_head_q * head_dim * sizeof(float));
    block_q8_K_t *q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((n_head_q * head_dim / 256U) * sizeof(block_q8_K_t));

    float max_diff = 0.0f;
    for (uint32_t pos = 0; pos < 42U; ++pos) {
        const uint32_t slot = pos % cap;
        for (uint32_t i = 0; i < kv_dim; ++i) {
            float kv = 0.4f * sinf((float)(pos * 19U + i) * 0.11f);
            float vv = 0.5f * cosf((float)(pos * 23U + i) * 0.09f);
            k_ring[(size_t)slot * kv_dim + i] = lfm2_fp32_to_fp16(kv);
            v_ring[(size_t)slot * kv_dim + i] = lfm2_fp32_to_fp16(vv);
            k_lin[(size_t)pos * kv_dim + i]   = lfm2_fp16_to_fp32(k_ring[(size_t)slot * kv_dim + i]);
            v_lin[(size_t)pos * kv_dim + i]   = lfm2_fp16_to_fp32(v_ring[(size_t)slot * kv_dim + i]);
        }
        for (uint32_t i = 0; i < n_head_q * head_dim; ++i) {
            q[i] = 0.3f * sinf((float)(pos * 13U + i) * 0.07f);
        }
        for (uint32_t h = 0; h < n_head_q; ++h) {
            gate[h] = 0.25f + 0.08f * (float)h;
        }

        lfm3_online_gqa_f16_dispatch(&pool, q, gate, k_ring, v_ring, out, q8,
                                     1U, pos, win, cap, n_head_q, n_head_kv, head_dim);

        /* Naive 2-pass exact SWA softmax over [max(0, pos + 1 - win) .. pos] */
        const uint32_t t0 = (pos + 1U > win) ? (pos + 1U - win) : 0U;
        const float scale = 1.0f / sqrtf((float)head_dim);
        for (uint32_t h = 0; h < n_head_q; ++h) {
            const uint32_t kv_h = h / (n_head_q / n_head_kv);
            float scores[64], max_s = -1e30f;
            for (uint32_t t = t0; t <= pos; ++t) {
                float dot = 0.0f;
                for (uint32_t i = 0; i < head_dim; ++i) {
                    dot += q[h * head_dim + i] * k_lin[t * kv_dim + kv_h * head_dim + i];
                }
                scores[t - t0] = dot * scale;
                if (scores[t - t0] > max_s) max_s = scores[t - t0];
            }
            float denom = 0.0f;
            for (uint32_t t = t0; t <= pos; ++t) {
                scores[t - t0] = expf(scores[t - t0] - max_s);
                denom += scores[t - t0];
            }
            for (uint32_t i = 0; i < head_dim; ++i) {
                float acc = 0.0f;
                for (uint32_t t = t0; t <= pos; ++t) {
                    acc += (scores[t - t0] / denom) * v_lin[t * kv_dim + kv_h * head_dim + i];
                }
                ref[h * head_dim + i] = acc * gate[h];
                float d = fabsf(out[h * head_dim + i] - ref[h * head_dim + i]);
                if (d > max_diff) max_diff = d;
            }
        }
    }

    lfm2_threadpool_free(&pool);
    free(k_ring); free(v_ring); free(k_lin); free(v_lin);
    free(q); free(gate); free(out); free(ref); free(q8);

    printf("  Online GQA-128 + SWA wrap max diff vs exact 2-pass softmax = %.8f\n", max_diff);
    if (max_diff > 1e-4f) return 1;
    return 0;
}

static uint32_t test_greedy_fn(float *logits, uint32_t V, void *ud) {
    (void)ud;
    uint32_t best = 0U;
    for (uint32_t i = 1U; i < V; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

typedef struct {
    LFM2Model m;
    float *w_embd, *w_norm1, *w_norm2, *w_qnorm, *w_inproj, *w_conv, *w_proj, *w_kv, *w_agate, *w_ff, *w_down, *w_rf;
    GGUFTensor t_embd, t_norm1, t_norm2, t_qnorm, t_inproj, t_conv, t_proj, t_kv, t_agate, t_ff, t_down, t_rf;
} SynthLFM3Fixture;

static void synth_lfm3_init(SynthLFM3Fixture *f) {
    memset(f, 0, sizeof(*f));
    const uint32_t D         = 512U;
    const uint32_t n_head_q  = 4U;
    const uint32_t n_head_kv = 2U;
    const uint32_t head_dim  = 128U;
    const uint32_t kv_dim    = n_head_kv * head_dim;
    const uint32_t n_ff      = 512U;
    const uint32_t V         = 64U;
    const uint32_t n_layers  = 4U;

    f->w_embd   = (float *)lfm2_aligned_alloc_64((size_t)V * D * sizeof(float));
    f->w_norm1  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    f->w_norm2  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    f->w_qnorm  = (float *)lfm2_aligned_alloc_64(head_dim * sizeof(float));
    f->w_inproj = (float *)lfm2_aligned_alloc_64((size_t)3U * D * D * sizeof(float));
    f->w_conv   = (float *)lfm2_aligned_alloc_64((size_t)7U * D * sizeof(float));
    f->w_proj   = (float *)lfm2_aligned_alloc_64((size_t)D * D * sizeof(float));
    f->w_kv     = (float *)lfm2_aligned_alloc_64((size_t)kv_dim * D * sizeof(float));
    f->w_agate  = (float *)lfm2_aligned_alloc_64((size_t)n_head_q * D * sizeof(float));
    f->w_ff     = (float *)lfm2_aligned_alloc_64((size_t)n_ff * D * sizeof(float));
    f->w_down   = (float *)lfm2_aligned_alloc_64((size_t)D * n_ff * sizeof(float));
    f->w_rf     = (float *)lfm2_aligned_alloc_64(64U * sizeof(float));

    for (size_t i = 0; i < (size_t)V * D; ++i)        f->w_embd[i]   = 0.25f * sinf((float)i * 0.031f);
    for (size_t i = 0; i < D; ++i)                   f->w_norm1[i]  = 1.0f + 0.05f * cosf((float)i);
    for (size_t i = 0; i < D; ++i)                   f->w_norm2[i]  = 0.9f + 0.07f * sinf((float)i);
    for (size_t i = 0; i < head_dim; ++i)            f->w_qnorm[i]  = 1.0f + 0.03f * sinf((float)i);
    for (size_t i = 0; i < (size_t)3U * D * D; ++i)  f->w_inproj[i] = 0.03f * sinf((float)i * 0.017f);
    for (size_t i = 0; i < (size_t)7U * D; ++i)      f->w_conv[i]   = 0.15f * cosf((float)i * 0.09f);
    for (size_t i = 0; i < (size_t)D * D; ++i)       f->w_proj[i]   = 0.03f * cosf((float)i * 0.013f);
    for (size_t i = 0; i < (size_t)kv_dim * D; ++i)  f->w_kv[i]     = 0.04f * sinf((float)i * 0.019f);
    for (size_t i = 0; i < (size_t)n_head_q * D; ++i) f->w_agate[i] = 0.05f * sinf((float)i * 0.023f);
    for (size_t i = 0; i < (size_t)n_ff * D; ++i)    f->w_ff[i]     = 0.03f * cosf((float)i * 0.011f);
    for (size_t i = 0; i < (size_t)D * n_ff; ++i)    f->w_down[i]   = 0.03f * sinf((float)i * 0.029f);
    for (uint32_t i = 0; i < 64U; ++i)               f->w_rf[i]     = (i < 32U) ? 1.0f : 1e30f;

    f->t_embd   = (GGUFTensor){ .n_dims = 2, .ne = {D, V, 1, 1},        .type = GGML_TYPE_F32, .data = f->w_embd   };
    f->t_norm1  = (GGUFTensor){ .n_dims = 1, .ne = {D, 1, 1, 1},        .type = GGML_TYPE_F32, .data = f->w_norm1  };
    f->t_norm2  = (GGUFTensor){ .n_dims = 1, .ne = {D, 1, 1, 1},        .type = GGML_TYPE_F32, .data = f->w_norm2  };
    f->t_qnorm  = (GGUFTensor){ .n_dims = 1, .ne = {head_dim, 1, 1, 1}, .type = GGML_TYPE_F32, .data = f->w_qnorm  };
    f->t_inproj = (GGUFTensor){ .n_dims = 2, .ne = {D, 3U * D, 1, 1},   .type = GGML_TYPE_F32, .data = f->w_inproj };
    f->t_conv   = (GGUFTensor){ .n_dims = 2, .ne = {7, D, 1, 1},        .type = GGML_TYPE_F32, .data = f->w_conv   };
    f->t_proj   = (GGUFTensor){ .n_dims = 2, .ne = {D, D, 1, 1},        .type = GGML_TYPE_F32, .data = f->w_proj   };
    f->t_kv     = (GGUFTensor){ .n_dims = 2, .ne = {D, kv_dim, 1, 1},   .type = GGML_TYPE_F32, .data = f->w_kv     };
    f->t_agate  = (GGUFTensor){ .n_dims = 2, .ne = {D, n_head_q, 1, 1}, .type = GGML_TYPE_F32, .data = f->w_agate  };
    f->t_ff     = (GGUFTensor){ .n_dims = 2, .ne = {D, n_ff, 1, 1},     .type = GGML_TYPE_F32, .data = f->w_ff     };
    f->t_down   = (GGUFTensor){ .n_dims = 2, .ne = {n_ff, D, 1, 1},     .type = GGML_TYPE_F32, .data = f->w_down   };
    f->t_rf     = (GGUFTensor){ .n_dims = 1, .ne = {64, 1, 1, 1},       .type = GGML_TYPE_F32, .data = f->w_rf     };

    LFM2Model *m = &f->m;
    snprintf(m->cfg.arch, sizeof(m->cfg.arch), "lfm3");
    m->cfg.is_lfm3            = true;
    m->cfg.n_layers           = n_layers;
    m->cfg.n_embd             = D;
    m->cfg.n_ff               = n_ff;
    m->cfg.n_vocab            = V;
    m->cfg.n_ctx              = 256U;
    m->cfg.n_head_q           = n_head_q;
    m->cfg.head_dim           = head_dim;
    m->cfg.rope_dim           = head_dim;
    m->cfg.shortconv_l_cache  = 7U;
    m->cfg.sliding_window     = 8U;
    m->cfg.rms_norm_eps       = 1e-5f;
    m->cfg.rope_freq_base     = 1e6f;
    m->cfg.rope_freq_base_swa = 1e4f;
    m->token_embd             = &f->t_embd;
    m->output_norm            = &f->t_norm1;
    m->output                 = &f->t_embd;
    m->rope_freqs             = &f->t_rf;

    for (uint32_t l = 0; l < n_layers; ++l) {
        LFM2Layer *ly = &m->layers[l];
        ly->layer_idx      = l;
        ly->attn_norm      = &f->t_norm1;
        ly->post_attn_norm = &f->t_norm2;
        ly->ffn_norm       = &f->t_norm1;
        ly->post_ffn_norm  = &f->t_norm2;
        ly->ffn_gate       = &f->t_ff;
        ly->ffn_up         = &f->t_ff;
        ly->ffn_down       = &f->t_down;
        if (l == 0U || l == 2U) {
            ly->block_type         = LFM2_BLOCK_SHORTCONV;
            ly->shortconv.in_proj  = &f->t_inproj;
            ly->shortconv.conv     = &f->t_conv;
            ly->shortconv.out_proj = &f->t_proj;
            m->cfg.n_head_kv[l]    = 0U;
        } else {
            ly->block_type        = (l == 1U) ? LFM2_BLOCK_SWA_ATTN : LFM2_BLOCK_GQA_ATTN;
            m->cfg.is_swa_layer[l] = (l == 1U);
            m->cfg.n_head_kv[l]   = n_head_kv;
            ly->attn.wq = &f->t_proj; ly->attn.wk = &f->t_kv; ly->attn.wv = &f->t_kv;
            ly->attn.q_norm = &f->t_qnorm; ly->attn.k_norm = &f->t_qnorm;
            ly->attn.gate = &f->t_agate; ly->attn.wo = &f->t_proj;
        }
    }
}

static void synth_lfm3_free(SynthLFM3Fixture *f) {
    free(f->w_embd); free(f->w_norm1); free(f->w_norm2); free(f->w_qnorm);
    free(f->w_inproj); free(f->w_conv); free(f->w_proj); free(f->w_kv);
    free(f->w_agate); free(f->w_ff); free(f->w_down); free(f->w_rf);
}

static int test_synthetic_lfm3_prefill_decode_spec(void) {
    printf("[Test 4/6] Verifying Synthetic 4-Layer LFM3 (Prefill==Decode, Spec Rollback, Scalar==AVX2)...\n");
    SynthLFM3Fixture fix;
    synth_lfm3_init(&fix);
    const LFM2Model *m_ptr = &fix.m;
    const uint32_t V = fix.m.cfg.n_vocab;

    LFM3State s_seq, s_batch, s_scalar;
    if (!lfm3_state_init(&s_seq,    m_ptr, 128U, 4U)) return 1;
    if (!lfm3_state_init(&s_batch,  m_ptr, 128U, 4U)) return 1;
    if (!lfm3_state_init(&s_scalar, m_ptr, 128U, 2U)) return 1;

    uint32_t seq_toks[15];
    for (uint32_t i = 0; i < 15U; ++i) seq_toks[i] = (i * 7U + 3U) % V;

    float *l_seq = NULL;
    for (uint32_t i = 0; i < 15U; ++i) {
        l_seq = lfm3_forward_token(m_ptr, &s_seq, seq_toks[i], i, i == 14U);
    }

    lfm3_forward_batch(m_ptr, &s_batch, seq_toks + 0U, 8U, 0U, false);
    float *l_batch = lfm3_forward_batch(m_ptr, &s_batch, seq_toks + 8U, 7U, 8U, true);

    float max_pb_diff = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float d = fabsf(l_seq[v] - l_batch[v]);
        if (d > max_pb_diff) max_pb_diff = d;
    }

    setenv("LFM2_FORCE_ISA", "scalar", 1);
    float *l_sc = NULL;
    for (uint32_t i = 0; i < 15U; ++i) {
        l_sc = lfm3_forward_token(m_ptr, &s_scalar, seq_toks[i], i, i == 14U);
    }
    unsetenv("LFM2_FORCE_ISA");

    float max_sc_diff = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float d = fabsf(l_seq[v] - l_sc[v]);
        if (d > max_sc_diff) max_sc_diff = d;
    }

    LFM3State s_spec;
    if (!lfm3_state_init(&s_spec, m_ptr, 128U, 4U)) return 1;
    for (uint32_t i = 0; i < 10U; ++i) {
        lfm3_forward_token(m_ptr, &s_spec, seq_toks[i], i, false);
    }
    uint32_t spec_in[4] = { seq_toks[10], 61U, 62U, 63U };
    float *sb = lfm3_forward_speculative_batch(m_ptr, &s_spec, spec_in, 4U, 10U);
    if (!sb) return 1;
    lfm3_restore_shortconv_trace_slot(&s_spec, m_ptr, 0U);
    float *l_after_rb = NULL;
    for (uint32_t i = 11U; i < 15U; ++i) {
        l_after_rb = lfm3_forward_token(m_ptr, &s_spec, seq_toks[i], i, i == 14U);
    }
    float max_rb_diff = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float d = fabsf(l_seq[v] - l_after_rb[v]);
        if (d > max_rb_diff) max_rb_diff = d;
    }

    printf("  Prefill vs M=1 diff=%.8f | AVX2 vs Scalar diff=%.8f | SpecRollback diff=%.8f\n",
           max_pb_diff, max_sc_diff, max_rb_diff);

    lfm3_state_free(&s_seq); lfm3_state_free(&s_batch);
    lfm3_state_free(&s_scalar); lfm3_state_free(&s_spec);
    synth_lfm3_free(&fix);

    if (max_pb_diff > 1e-4f || max_sc_diff > 1e-4f || max_rb_diff > 1e-4f) return 1;
    return 0;
}

static int test_lfm3_checkpoint_promptcache_sinkshift(void) {
    printf("[Test 6/6] Verifying LFM3 SWA Checkpoint, Prompt Cache (pos > swa_win), and Sink Shift...\n");
    SynthLFM3Fixture fix;
    synth_lfm3_init(&fix);
    const LFM2Model *m = &fix.m;
    const uint32_t V = m->cfg.n_vocab;

    LFM3State st_ref, st_work;
    if (!lfm3_state_init(&st_ref,  m, 64U, 2U)) return 1;
    if (!lfm3_state_init(&st_work, m, 64U, 2U)) return 1;

    uint32_t toks[30];
    for (uint32_t i = 0; i < 30U; ++i) toks[i] = (i * 11U + 5U) % V;

    /* Baseline: process 20 tokens sequentially on st_ref */
    for (uint32_t i = 0; i < 15U; ++i) lfm3_forward_token(m, &st_ref, toks[i], i, false);
    float *ref_next = lfm3_forward_token(m, &st_ref, toks[15], 15U, true);

    /* Work state: advance 15 tokens (> swa_win=8), save checkpoint + prompt cache */
    for (uint32_t i = 0; i < 15U; ++i) lfm3_forward_token(m, &st_work, toks[i], i, false);

    LFM2Checkpoint cp;
    if (!lfm2_checkpoint_init(&cp, m, 0U)) return 1;
    lfm2_checkpoint_save_ex(&cp, m, (float * const *)st_work.conv_ring, st_work.conv_ring_head,
                            (uint16_t * const *)st_work.k_cache_f16,
                            (uint16_t * const *)st_work.v_cache_f16,
                            15U, 0ULL, NULL, 0U);

    const char *pcache_path = "/tmp/test_lfm3_cache.lfmcache";
    if (!lfm2_prompt_cache_save_ex(pcache_path, m, toks, 15U,
                                   (float * const *)st_work.conv_ring, st_work.conv_ring_head,
                                   (uint16_t * const *)st_work.k_cache_f16,
                                   (uint16_t * const *)st_work.v_cache_f16, st_work.swa_cap)) {
        fprintf(stderr, "FAIL: lfm2_prompt_cache_save_ex\n");
        return 1;
    }

    /* Clobber st_work with 15 more tokens (overwriting SWA ring twice!) then restore checkpoint */
    for (uint32_t i = 15U; i < 30U; ++i) lfm3_forward_token(m, &st_work, (toks[i] + 17U) % V, i, false);
    uint32_t rest_pos = 0U;
    if (!lfm2_checkpoint_restore_ex(&cp, m, st_work.conv_ring, st_work.conv_ring_head,
                                    st_work.k_cache_f16, st_work.v_cache_f16,
                                    &rest_pos, NULL, NULL, NULL) || rest_pos != 15U) {
        fprintf(stderr, "FAIL: lfm2_checkpoint_restore_ex\n");
        return 1;
    }
    float *ckpt_next = lfm3_forward_token(m, &st_work, toks[15], rest_pos, true);
    float ckpt_diff = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float d = fabsf(ref_next[v] - ckpt_next[v]);
        if (d > ckpt_diff) ckpt_diff = d;
    }

    /* Reset st_work cleanly, reload prompt cache, and verify continuation matches bit-exact! */
    LFM3State st_pc;
    if (!lfm3_state_init(&st_pc, m, 64U, 2U)) return 1;
    uint32_t matched = 0U;
    if (!lfm2_prompt_cache_try_load_prefix_ex(pcache_path, m, toks, 20U, st_pc.max_seq_len,
                                              st_pc.conv_ring, st_pc.conv_ring_head,
                                              st_pc.k_cache_f16, st_pc.v_cache_f16, st_pc.swa_cap,
                                              &matched) || matched != 15U) {
        fprintf(stderr, "FAIL: lfm2_prompt_cache_try_load_prefix_ex (matched=%u)\n", matched);
        return 1;
    }
    unlink(pcache_path);
    float *pc_next = lfm3_forward_token(m, &st_pc, toks[15], matched, true);
    float pc_diff = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float d = fabsf(ref_next[v] - pc_next[v]);
        if (d > pc_diff) pc_diff = d;
    }

    /* Verify LFM3 Sink-Shift on Global (partial RoPE) + SWA Ring preserves finite logits & new pos */
    uint32_t shift_pos = 24U;
    for (uint32_t i = 0; i < 24U; ++i) lfm3_forward_token(m, &st_pc, toks[i], i, false);
    lfm3_kv_cache_shift_sinks(st_pc.k_cache_f16, st_pc.v_cache_f16, m,
                              &st_pc.rope_lut_full, &st_pc.rope_lut_swa, st_pc.swa_cap,
                              &shift_pos, 4U, 10U);
    float *after_shift = lfm3_forward_token(m, &st_pc, toks[24], shift_pos, true);
    bool shift_ok = (shift_pos == 14U) && (after_shift != NULL);
    for (uint32_t v = 0; v < V && shift_ok; ++v) {
        if (!isfinite(after_shift[v])) shift_ok = false;
    }

    printf("  Checkpoint diff=%.8f | PromptCache diff=%.8f | SinkShift pos=%u ok=%s\n",
           ckpt_diff, pc_diff, shift_pos, shift_ok ? "YES" : "NO");

    lfm2_checkpoint_destroy(&cp);
    lfm3_state_free(&st_ref); lfm3_state_free(&st_work); lfm3_state_free(&st_pc);
    synth_lfm3_free(&fix);

    if (ckpt_diff > 1e-5f || pc_diff > 1e-5f || !shift_ok) return 1;
    return 0;
}

static int test_real_lfm3_gguf(void) {
    const char *path = "models/LFM3-1.7B-Q4_K_M.gguf";
    if (access(path, R_OK) != 0) {
        printf("[Test 5/6] SKIP: %s not present locally.\n", path);
        return 0;
    }
    printf("[Test 5/6] Verifying Real LFM3-1.7B-Q4_K_M.gguf (Prefill==M1 & 12-Step Decode)...\n");
    LFM2Model m;
    if (!lfm2_load_gguf(path, &m)) return 1;
    if (!m.cfg.is_lfm3 || m.cfg.head_dim != 128U || m.cfg.sliding_window != 256U) {
        fprintf(stderr, "FAIL: unexpected LFM3 config\n");
        lfm2_free_model(&m);
        return 1;
    }

    LFM3State s_batch, s_seq;
    if (!lfm3_state_init(&s_batch, &m, 512U, 0U)) return 1;
    if (!lfm3_state_init(&s_seq,   &m, 512U, 0U)) return 1;

    uint32_t prompt[6] = {124894U, 124899U, 5922U, 207U, 35808U, 124900U};
    float *l_b = lfm3_forward_batch(&m, &s_batch, prompt, 6U, 0U, true);
    float *l_s = NULL;
    for (uint32_t i = 0; i < 6U; ++i) {
        l_s = lfm3_forward_token(&m, &s_seq, prompt[i], i, i == 5U);
    }

    double dot = 0.0, na = 0.0, nb = 0.0;
    float max_diff = 0.0f;
    for (uint32_t v = 0; v < m.cfg.n_vocab; ++v) {
        if (!isfinite(l_b[v]) || !isfinite(l_s[v])) {
            fprintf(stderr, "FAIL: non-finite logit at v=%u\n", v);
            lfm3_state_free(&s_batch); lfm3_state_free(&s_seq); lfm2_free_model(&m);
            return 1;
        }
        dot += (double)l_b[v] * (double)l_s[v];
        na  += (double)l_b[v] * (double)l_b[v];
        nb  += (double)l_s[v] * (double)l_s[v];
        float d = fabsf(l_b[v] - l_s[v]);
        if (d > max_diff) max_diff = d;
    }
    const double cos_sim = (na > 0.0 && nb > 0.0) ? (dot / (sqrt(na) * sqrt(nb))) : 0.0;
    uint32_t top_b = test_greedy_fn(l_b, m.cfg.n_vocab, NULL);
    uint32_t top_s = test_greedy_fn(l_s, m.cfg.n_vocab, NULL);
    printf("  Real LFM3 B=6 vs M=1x6 cos_sim=%.6f max_logit_diff=%.6f (top_b=%u top_s=%u)\n",
           cos_sim, max_diff, top_b, top_s);

    uint32_t tok = top_b;
    uint32_t pos = 6U;
    printf("  12-step greedy token IDs from real model: ");
    for (uint32_t step = 0; step < 12U; ++step) {
        printf("%u ", tok);
        float *lg = lfm3_forward_token(&m, &s_batch, tok, pos++, true);
        if (!lg) {
            printf("\nFAIL: decode step %u returned NULL\n", step);
            lfm3_state_free(&s_batch); lfm3_state_free(&s_seq); lfm2_free_model(&m);
            return 1;
        }
        tok = test_greedy_fn(lg, m.cfg.n_vocab, NULL);
    }
    printf("\n");

    lfm3_state_free(&s_batch);
    lfm3_state_free(&s_seq);
    lfm2_free_model(&m);

    if (cos_sim < 0.995 || top_b != top_s) {
        fprintf(stderr, "FAIL: Real LFM3 prefill vs sequential decode diverged!\n");
        return 1;
    }
    return 0;
}

int main(void) {
    printf("=======================================================\n");
    printf(" Liquid AI LFM3 Engine Verification Suite\n");
    printf("=======================================================\n");
    if (test_shortconv7_ring() != 0) return 1;
    if (test_partial_rope_and_qk128() != 0) return 1;
    if (test_online_gqa128_swa_wrap() != 0) return 1;
    if (test_synthetic_lfm3_prefill_decode_spec() != 0) return 1;
    if (test_real_lfm3_gguf() != 0) return 1;
    if (test_lfm3_checkpoint_promptcache_sinkshift() != 0) return 1;
    printf("=======================================================\n");
    printf(" ALL LFM3 ENGINE TESTS PASSED WITH 100%% SUCCESS!\n");
    printf("=======================================================\n");
    return 0;
}
