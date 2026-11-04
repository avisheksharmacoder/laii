#define _POSIX_C_SOURCE 200809L
#include "lfm_forward.h"
#include "lfm_simd.h"
#include "lfm_fused.h"
#include "lfm_threadpool.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF);

    if (exp == 0) {
        if (frac == 0) {
            uint32_t u = sign;
            float f;
            memcpy(&f, &u, 4);
            return f;
        }
        while ((frac & 0x0400) == 0) {
            frac <<= 1;
            exp--;
        }
        exp++;
        frac &= ~0x0400;
        exp = (exp + (127 - 15)) << 23;
        frac <<= 13;
        uint32_t u = sign | exp | frac;
        float f;
        memcpy(&f, &u, 4);
        return f;
    } else if (exp == 0x1F) {
        uint32_t u = sign | 0x7F800000 | (frac << 13);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }

    exp = (exp + (127 - 15)) << 23;
    frac <<= 13;
    uint32_t u = sign | exp | frac;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

bool lfm_state_init(LFMState *state, const LFMConfig *config, MemoryArena *arena) {
    if (!state || !config || !arena) return false;
    memset(state, 0, sizeof(LFMState));

    state->current_pos = 0;
    state->max_seq_len = config->max_seq_len;

    for (uint32_t i = 0; i < config->n_layers; ++i) {
        if (config->is_conv_layer[i]) {
            state->conv_states[i].s0 = (float*)arena_alloc(arena, config->hidden_dim * sizeof(float));
            state->conv_states[i].s1 = (float*)arena_alloc(arena, config->hidden_dim * sizeof(float));
            if (!state->conv_states[i].s0 || !state->conv_states[i].s1) return false;
        } else {
            size_t kv_size = (size_t)config->max_seq_len * config->n_heads_kv * config->head_dim * sizeof(float);
            state->kv_caches[i].k_cache = (float*)arena_alloc(arena, kv_size);
            state->kv_caches[i].v_cache = (float*)arena_alloc(arena, kv_size);
            if (!state->kv_caches[i].k_cache || !state->kv_caches[i].v_cache) return false;
        }
    }

    return true;
}

void lfm_state_reset(LFMState *state, const LFMConfig *config) {
    if (!state || !config) return;
    state->current_pos = 0;

    for (uint32_t i = 0; i < config->n_layers; ++i) {
        if (config->is_conv_layer[i]) {
            if (state->conv_states[i].s0) memset(state->conv_states[i].s0, 0, config->hidden_dim * sizeof(float));
            if (state->conv_states[i].s1) memset(state->conv_states[i].s1, 0, config->hidden_dim * sizeof(float));
        } else {
            size_t kv_size = (size_t)config->max_seq_len * config->n_heads_kv * config->head_dim * sizeof(float);
            if (state->kv_caches[i].k_cache) memset(state->kv_caches[i].k_cache, 0, kv_size);
            if (state->kv_caches[i].v_cache) memset(state->kv_caches[i].v_cache, 0, kv_size);
        }
    }
}

void lfm_rmsnorm(float *out, const float *x, const float *weight, size_t dim, float eps) {
    lfm_vec_rmsnorm_f32(out, x, weight, dim, eps);
}

void lfm_silu_vec(float *out, const float *x, size_t dim) {
    lfm_vec_silu_f32(out, x, dim);
}

void lfm_rope_neox(float *vec, uint32_t head_dim, uint32_t pos, float theta) {
    uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float phi = (float)pos * freq;
        float cos_val = cosf(phi);
        float sin_val = sinf(phi);

        float x0 = vec[i];
        float x1 = vec[i + half];

        vec[i]        = x0 * cos_val - x1 * sin_val;
        vec[i + half] = x1 * cos_val + x0 * sin_val;
    }
}

void lfm_matvec(float *out, const void *weight, GGMLType w_type, const float *x, size_t out_dim, size_t in_dim) {
    lfm_matvec_simd(out, weight, w_type, x, out_dim, in_dim);
}

void lfm_liv_shortconv_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    LFMConvState *conv_state,
    uint32_t hidden_dim,
    MemoryArena *scratch
) {
    // 1. in_proj: [hidden_dim] -> [3 * hidden_dim]
    float *bcx = (float*)arena_alloc(scratch, 3 * hidden_dim * sizeof(float));
    lfm_matvec_simd(bcx, layer->shortconv_in_proj, layer->shortconv_in_proj_type, x_norm, 3 * hidden_dim, hidden_dim);

    // 2. Fused LIV Gate & Conv Step:
    // Fuses Gate 1 (B * x), Depthwise Conv1D (k=3), rolling state update, and Gate 2 (C * conv_out)
    // into a single cache-resident SIMD pass!
    float *y = (float*)arena_alloc(scratch, hidden_dim * sizeof(float));
    lfm_fused_liv_gate_conv_step(
        y,
        bcx,
        layer->shortconv_conv,
        layer->shortconv_conv_bias,
        conv_state,
        hidden_dim
    );

    // 3. out_proj: [hidden_dim] -> [hidden_dim]
    lfm_matvec_simd(out, layer->shortconv_out_proj, layer->shortconv_out_proj_type, y, hidden_dim, hidden_dim);
}

void lfm_liv_shortconv_prefill(
    float *out_seq,
    const float *x_norm_seq,
    const LFMLayerWeights *layer,
    LFMConvState *conv_state,
    uint32_t hidden_dim,
    uint32_t seq_len,
    MemoryArena *scratch
) {
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *x_norm_t = x_norm_seq + t * hidden_dim;
        float *out_t = out_seq + t * hidden_dim;
        lfm_liv_shortconv_step(out_t, x_norm_t, layer, conv_state, hidden_dim, scratch);
    }
}

void lfm_gqa_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    LFMKVCache *kv_cache,
    uint32_t pos,
    const LFMConfig *config,
    MemoryArena *scratch
) {
    uint32_t hidden_dim = config->hidden_dim;
    uint32_t n_heads_q = config->n_heads_q;
    uint32_t n_heads_kv = config->n_heads_kv;
    uint32_t head_dim = config->head_dim;
    uint32_t group_size = n_heads_q / n_heads_kv;

    // 1. Projections
    float *q = (float*)arena_alloc(scratch, n_heads_q * head_dim * sizeof(float));
    float *k = (float*)arena_alloc(scratch, n_heads_kv * head_dim * sizeof(float));
    float *v = (float*)arena_alloc(scratch, n_heads_kv * head_dim * sizeof(float));

    lfm_matvec_simd(q, layer->q_proj, layer->q_proj_type, x_norm, n_heads_q * head_dim, hidden_dim);
    lfm_matvec_simd(k, layer->k_proj, layer->k_proj_type, x_norm, n_heads_kv * head_dim, hidden_dim);
    lfm_matvec_simd(v, layer->v_proj, layer->v_proj_type, x_norm, n_heads_kv * head_dim, hidden_dim);

    // 2. QK-RMSNorm per head
    for (uint32_t h = 0; h < n_heads_q; ++h) {
        float *qh = q + h * head_dim;
        lfm_rmsnorm(qh, qh, layer->q_norm, head_dim, config->norm_eps);
        lfm_rope_neox(qh, head_dim, pos, config->rope_theta);
    }

    for (uint32_t h = 0; h < n_heads_kv; ++h) {
        float *kh = k + h * head_dim;
        lfm_rmsnorm(kh, kh, layer->k_norm, head_dim, config->norm_eps);
        lfm_rope_neox(kh, head_dim, pos, config->rope_theta);
    }

    // 3. Store into KV-cache at current pos
    size_t kv_offset = (size_t)pos * n_heads_kv * head_dim;
    memcpy(kv_cache->k_cache + kv_offset, k, n_heads_kv * head_dim * sizeof(float));
    memcpy(kv_cache->v_cache + kv_offset, v, n_heads_kv * head_dim * sizeof(float));

    // 4. Multi-Head Softmax Attention
    float *attn_out = (float*)arena_alloc(scratch, n_heads_q * head_dim * sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float*)arena_alloc(scratch, (pos + 1) * sizeof(float));

    for (uint32_t hq = 0; hq < n_heads_q; ++hq) {
        uint32_t hkv = hq / group_size;
        const float *qh = q + hq * head_dim;
        float *out_h = attn_out + hq * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));

        float max_score = -1e30f;
        for (uint32_t t = 0; t <= pos; ++t) {
            const float *kt = kv_cache->k_cache + ((size_t)t * n_heads_kv + hkv) * head_dim;
            float dot = lfm_vec_dot_f32(qh, kt, head_dim);
            float s = dot * scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (uint32_t t = 0; t <= pos; ++t) {
            float exp_val = expf(scores[t] - max_score);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t t = 0; t <= pos; ++t) {
            scores[t] *= inv_sum;
        }

        // Weighted sum over V
        for (uint32_t t = 0; t <= pos; ++t) {
            float weight = scores[t];
            const float *vt = kv_cache->v_cache + ((size_t)t * n_heads_kv + hkv) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out_h[d] += weight * vt[d];
            }
        }
    }

    // 5. Output Projection: [n_heads_q * head_dim] -> [hidden_dim]
    lfm_matvec_simd(out, layer->attn_out_proj, layer->attn_out_proj_type, attn_out, hidden_dim, hidden_dim);
}

void lfm_swiglu_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    MemoryArena *scratch
) {
    float *gate = (float*)arena_alloc(scratch, ffn_dim * sizeof(float));
    float *up   = (float*)arena_alloc(scratch, ffn_dim * sizeof(float));
    float *act  = (float*)arena_alloc(scratch, ffn_dim * sizeof(float));

    lfm_matvec_simd(gate, layer->ffn_gate, layer->ffn_gate_type, x_norm, ffn_dim, hidden_dim);
    lfm_matvec_simd(up,   layer->ffn_up,   layer->ffn_up_type,   x_norm, ffn_dim, hidden_dim);

    // SwiGLU: silu(gate) * up
    lfm_vec_silu_f32(gate, gate, ffn_dim);
    lfm_vec_mul_f32(act, gate, up, ffn_dim);

    lfm_matvec_simd(out, layer->ffn_down, layer->ffn_down_type, act, hidden_dim, ffn_dim);
}

void lfm_forward_token(
    float *logits,
    uint32_t token_id,
    LFMModel *model,
    LFMState *state,
    ThreadPool *pool,
    MemoryArena *scratch
) {
    const LFMConfig *config = &model->config;
    uint32_t hidden_dim = config->hidden_dim;
    uint32_t pos = state->current_pos;

    // 1. Embedding lookup: [hidden_dim]
    float *h = (float*)arena_alloc(scratch, hidden_dim * sizeof(float));
    if (model->tok_embeddings_type == GGML_TYPE_F32) {
        const float *emb = (const float*)model->tok_embeddings;
        memcpy(h, emb + (size_t)token_id * hidden_dim, hidden_dim * sizeof(float));
    } else if (model->tok_embeddings_type == GGML_TYPE_F16) {
        const uint16_t *emb = (const uint16_t*)model->tok_embeddings;
        const uint16_t *row = emb + (size_t)token_id * hidden_dim;
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            h[i] = fp16_to_fp32(row[i]);
        }
    } else {
        memset(h, 0, hidden_dim * sizeof(float));
    }

    // Working buffers
    float *normed = (float*)arena_alloc(scratch, hidden_dim * sizeof(float));
    float *op_out = (float*)arena_alloc(scratch, hidden_dim * sizeof(float));
    float *ffn_out = (float*)arena_alloc(scratch, hidden_dim * sizeof(float));

    // 2. Iterate through hybrid layers
    for (uint32_t i = 0; i < config->n_layers; ++i) {
        const LFMLayerWeights *lw = &model->layers[i];

        // Operator Norm
        lfm_rmsnorm(normed, h, lw->operator_norm, hidden_dim, config->norm_eps);

        if (config->is_conv_layer[i]) {
            // LIV ShortConv Block (uses fused kernel)
            lfm_liv_shortconv_step(op_out, normed, lw, &state->conv_states[i], hidden_dim, scratch);
        } else {
            // Attention Block
            lfm_gqa_step(op_out, normed, lw, &state->kv_caches[i], pos, config, scratch);
        }

        // Residual Add 1: h = h + op_out
        lfm_vec_add_f32(h, h, op_out, hidden_dim);

        // FFN Norm
        lfm_rmsnorm(normed, h, lw->ffn_norm, hidden_dim, config->norm_eps);

        // SwiGLU FFN Block
        lfm_swiglu_step(ffn_out, normed, lw, hidden_dim, config->ffn_dim, scratch);

        // Residual Add 2: h = h + ffn_out
        lfm_vec_add_f32(h, h, ffn_out, hidden_dim);
    }

    // 3. Final Output Norm
    lfm_rmsnorm(h, h, model->output_norm, hidden_dim, config->norm_eps);

    // 4. LM Head (logits projection)
    if (logits) {
        lfm_matvec_parallel(pool, logits, model->lm_head, model->lm_head_type, h, config->vocab_size, hidden_dim);
    }

    // Advance position
    state->current_pos++;
}
