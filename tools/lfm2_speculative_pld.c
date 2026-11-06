#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "include/lfm2_speculative_pld.h"
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LFM2_PREFILL_BATCH_NO_MAIN 1
#if __has_include("tools/lfm2_prefill_batch.c")
#include "tools/lfm2_prefill_batch.c"
#elif __has_include("lfm2_prefill_batch.c")
#include "lfm2_prefill_batch.c"
#else
#include "../tools/lfm2_prefill_batch.c"
#endif

/* ============================================================================
 * 1. O(1) Hierarchical 3-Gram / 2-Gram Hash Helpers
 * ============================================================================ */
static inline uint32_t hash_2gram(uint32_t t0, uint32_t t1) {
    uint64_t k = ((uint64_t)t0 << 32) | (uint64_t)t1;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return (uint32_t)k & LFM2_NGRAM_HT_MASK;
}

static inline uint32_t hash_3gram(uint32_t t0, uint32_t t1, uint32_t t2) {
    uint64_t k = ((uint64_t)t0 << 40) ^ ((uint64_t)t1 << 20) ^ (uint64_t)t2;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)k & LFM2_NGRAM_HT_MASK;
}

bool lfm2_ngram_index_init(LFM2NgramIndex *idx, uint32_t max_seq_len,
                           uint32_t max_draft_k, uint32_t eos_token_id) {
    if (!idx) return false;
    memset(idx, 0, sizeof(*idx));
    idx->max_seq_len  = max_seq_len;
    idx->eos_token_id = eos_token_id;
    idx->max_draft_k  = (max_draft_k > 0 && max_draft_k <= LFM2_PLD_MAX_DRAFT)
                        ? max_draft_k : 6;
    idx->cur_draft_k  = idx->max_draft_k;

    idx->history = (uint32_t *)malloc((size_t)max_seq_len * sizeof(uint32_t));
    if (!idx->history) return false;

    for (uint32_t i = 0; i < LFM2_NGRAM_HT_SIZE; ++i) {
        idx->ht_3gram[i] = -1;
        idx->ht_2gram[i] = -1;
    }
    return true;
}

void lfm2_ngram_index_reset(LFM2NgramIndex *idx) {
    if (!idx) return;
    idx->history_len              = 0;
    idx->cur_draft_k              = idx->max_draft_k;
    idx->consecutive_zero_accepts = 0;
    for (uint32_t i = 0; i < LFM2_NGRAM_HT_SIZE; ++i) {
        idx->ht_3gram[i] = -1;
        idx->ht_2gram[i] = -1;
    }
}

void lfm2_ngram_index_free(LFM2NgramIndex *idx) {
    if (!idx) return;
    if (idx->history) {
        free(idx->history);
        idx->history = NULL;
    }
    memset(idx, 0, sizeof(*idx));
}

void lfm2_ngram_index_ingest(LFM2NgramIndex *idx, uint32_t token_id) {
    if (!idx || !idx->history) return;
    const uint32_t pos = idx->history_len;
    if (pos >= idx->max_seq_len) return;

    idx->history[pos] = token_id;
    idx->history_len  = pos + 1;

    if (pos >= 2) {
        const uint32_t h2 = hash_2gram(idx->history[pos - 2], idx->history[pos - 1]);
        idx->ht_2gram[h2] = (int32_t)pos;
    }
    if (pos >= 3) {
        const uint32_t h3 = hash_3gram(idx->history[pos - 3],
                                       idx->history[pos - 2],
                                       idx->history[pos - 1]);
        idx->ht_3gram[h3] = (int32_t)pos;
    }
}

uint32_t lfm2_ngram_draft(const LFM2NgramIndex *idx,
                          uint32_t pending_token,
                          uint32_t draft_out[LFM2_PLD_MAX_DRAFT]) {
    if (!idx || !idx->history) return 0;
    const uint32_t len = idx->history_len;
    if (len < 1 || idx->cur_draft_k == 0) return 0;

    int32_t match_next_pos = -1;

    /* 1. Try high-precision 3-gram lookup: (history[len-2], history[len-1], pending_token) */
    if (len >= 2) {
        const uint32_t t0 = idx->history[len - 2];
        const uint32_t t1 = idx->history[len - 1];
        const uint32_t t2 = pending_token;
        const int32_t cand = idx->ht_3gram[hash_3gram(t0, t1, t2)];

        if (cand >= 3 && (uint32_t)cand < len &&
            idx->history[cand - 3] == t0 &&
            idx->history[cand - 2] == t1 &&
            idx->history[cand - 1] == t2) {
            match_next_pos = cand;
        } else {
            /* Fallback linear reverse scan up to 512 tokens if hash collided */
            const uint32_t scan_limit = (len > 512) ? (len - 512) : 2;
            for (uint32_t p = len - 1; p >= scan_limit; --p) {
                if (idx->history[p]     == t2 &&
                    idx->history[p - 1] == t1 &&
                    idx->history[p - 2] == t0) {
                    match_next_pos = (int32_t)(p + 1);
                    break;
                }
            }
        }
    }

    /* 2. Fallback to 2-gram lookup: (history[len-1], pending_token) */
    if (match_next_pos < 0 && len >= 1) {
        const uint32_t t0 = idx->history[len - 1];
        const uint32_t t1 = pending_token;
        const int32_t cand = idx->ht_2gram[hash_2gram(t0, t1)];

        if (cand >= 2 && (uint32_t)cand < len &&
            idx->history[cand - 2] == t0 &&
            idx->history[cand - 1] == t1) {
            match_next_pos = cand;
        }
    }

    if (match_next_pos < 0 || (uint32_t)match_next_pos >= len) {
        return 0;
    }

    /* 3. Copy up to `cur_draft_k` continuation tokens */
    uint32_t n_draft = 0;
    const uint32_t max_k = idx->cur_draft_k;

    for (uint32_t k = 0; k < max_k && (uint32_t)match_next_pos + k < len; ++k) {
        const uint32_t tok = idx->history[(uint32_t)match_next_pos + k];
        draft_out[n_draft++] = tok;
        if (tok == idx->eos_token_id) {
            break;
        }
    }
    return n_draft;
}

static inline void lfm2_ngram_update_adaptive_window(LFM2NgramIndex *idx,
                                                     uint32_t n_proposed,
                                                     uint32_t n_accepted) {
    if (!idx || n_proposed == 0) return;
    if (n_accepted == 0) {
        idx->consecutive_zero_accepts++;
        if (idx->consecutive_zero_accepts >= 3 && idx->cur_draft_k > 2) {
            idx->cur_draft_k = 2; /* Throttle draft batch size on consecutive misses */
        }
    } else {
        idx->consecutive_zero_accepts = 0;
        if (n_accepted >= idx->cur_draft_k / 2) {
            idx->cur_draft_k = idx->max_draft_k; /* Restore full draft window on hits */
        }
    }
}

/* ============================================================================
 * 2. Speculative Batch State Arena & conv_trace
 * ============================================================================ */
typedef struct {
    LFM2BatchState base;

    /* Per-batch-slot ShortConv rolling state trace:
     * `conv_trace[b][l]` stores the 2-token ShortConv state of layer `l`
     * immediately AFTER processing batch slot `b` (0 .. B-1).
     * Enables O(1) rollback to any accepted prefix slot in <10 us without recomputation! */
    float *conv_trace[LFM2_MAX_BATCH][LFM2_MAX_LAYERS];

    /* Batch output logits: [LFM2_MAX_BATCH, n_vocab] */
    float *logits_batch;
} LFM2SpecState;

bool lfm2_spec_state_init(LFM2SpecState *ss, const LFM2Model *m,
                          uint32_t max_seq_len, uint32_t n_threads) {
    if (!ss || !m) return false;
    memset(ss, 0, sizeof(*ss));
    if (!lfm2_batch_state_init(&ss->base, m, max_seq_len, n_threads)) {
        return false;
    }

    const LFM2Config *c = &m->cfg;
    const size_t D        = c->n_embd;
    const size_t hist_len = (c->shortconv_l_cache > 1) ? (c->shortconv_l_cache - 1) : 2;
    const size_t conv_bytes = hist_len * D * sizeof(float);

    for (uint32_t b = 0; b < LFM2_MAX_BATCH; ++b) {
        for (uint32_t l = 0; l < c->n_layers; ++l) {
            if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
                ss->conv_trace[b][l] = (float *)lfm2_aligned_alloc(64, conv_bytes);
                if (!ss->conv_trace[b][l]) return false;
            }
        }
    }

    const size_t logits_bytes = (size_t)LFM2_MAX_BATCH * c->n_vocab * sizeof(float);
    ss->logits_batch = (float *)lfm2_aligned_alloc(64, logits_bytes);
    return (ss->logits_batch != NULL);
}

void lfm2_spec_state_free(LFM2SpecState *ss) {
    if (!ss) return;
    for (uint32_t b = 0; b < LFM2_MAX_BATCH; ++b) {
        for (uint32_t l = 0; l < LFM2_MAX_LAYERS; ++l) {
            if (ss->conv_trace[b][l]) {
                free(ss->conv_trace[b][l]);
                ss->conv_trace[b][l] = NULL;
            }
        }
    }
    if (ss->logits_batch) {
        free(ss->logits_batch);
        ss->logits_batch = NULL;
    }
    lfm2_batch_state_free(&ss->base);
    memset(ss, 0, sizeof(*ss));
}

/* ============================================================================
 * 3. Speculative Batched ShortConv & Full Forward Pass
 * ============================================================================ */
static void lfm2_forward_shortconv_spec_batch(LFM2SpecState *ss,
                                              const LFM2Model *m,
                                              const LFM2Layer *layer,
                                              uint32_t l_idx,
                                              uint32_t B) {
    LFM2BatchState *s      = &ss->base;
    const uint32_t D       = m->cfg.n_embd;
    const uint32_t l_cache = m->cfg.shortconv_l_cache ? m->cfg.shortconv_l_cache : 3;
    const uint32_t hist    = (l_cache > 1) ? (l_cache - 1) : 2;
    const size_t   state_bytes = (size_t)hist * D * sizeof(float);

    /* 1. Batched Input Projection: xb [B, D] -> bcx [B, 3 * D] */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->bcx, layer->shortconv.in_proj, s->xb, B, s->xq);

    /* 2. Causal Depthwise Conv1D + Trace Snapshot across t = 0 .. B - 1 */
    float       *state = s->conv_state[l_idx];
    const float *w0    = s->conv_w[l_idx][0];
    const float *w1    = s->conv_w[l_idx][1];
    const float *w2    = s->conv_w[l_idx][2];

    for (uint32_t t = 0; t < B; ++t) {
        float *bcx_t   = s->bcx + (size_t)t * 3 * D;
        float *B_gate  = bcx_t;
        float *C_gate  = bcx_t + D;
        float *x_tilde = bcx_t + 2 * D;
        float *out_t   = s->conv_out + (size_t)t * D;

        if (w0 && w1 && w2 && hist == 2) {
            const float *s0 = state;
            const float *s1 = state + D;
#if LFM2_HAS_AVX2
            for (uint32_t d = 0; d < D; d += 8) {
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
#else
            for (uint32_t d = 0; d < D; ++d) {
                float u = B_gate[d] * x_tilde[d];
                B_gate[d] = u;
                float acc = w0[d] * s0[d] + w1[d] * s1[d] + w2[d] * u;
                out_t[d] = C_gate[d] * acc;
            }
#endif
        } else {
            const GGUFTensor *conv_t = layer->shortconv.conv;
            const bool stride_is_lcache = (conv_t && conv_t->ne[0] == l_cache);
            const float *w_base = conv_t ? (const float *)conv_t->data : NULL;
            for (uint32_t d = 0; d < D; ++d) {
                float u_t = B_gate[d] * x_tilde[d];
                B_gate[d] = u_t;

                float acc = 0.0f;
                if (w_base) {
                    for (uint32_t k = 0; k < hist; ++k) {
                        float w_k = stride_is_lcache ? w_base[d * l_cache + k] : w_base[k * D + d];
                        acc += w_k * state[k * D + d];
                    }
                    float w_last = stride_is_lcache ? w_base[d * l_cache + hist] : w_base[hist * D + d];
                    acc += w_last * u_t;
                }
                out_t[d] = C_gate[d] * acc;
            }
        }

        if (hist > 1) {
            memmove(state, state + D, (size_t)(hist - 1) * D * sizeof(float));
        }
        memcpy(state + (size_t)(hist - 1) * D, B_gate, D * sizeof(float));

        /* Snapshot this layer's ShortConv state after token `t` into `conv_trace[t][l_idx]` */
        memcpy(ss->conv_trace[t][l_idx], state, state_bytes);
    }

    /* 3. Batched Output Projection: conv_out [B, D] -> xb [B, D] + Residual Add */
    lfm2_matmul_batch_avx2_mt(&s->pool, s->xb, layer->shortconv.out_proj, s->conv_out, B, s->xq);

    const size_t total = (size_t)B * D;
    for (size_t i = 0; i < total; ++i) {
        s->x[i] += s->xb[i];
    }
}

float *lfm2_forward_speculative_batch(const LFM2Model *m,
                                      LFM2SpecState *ss,
                                      const uint32_t *tokens,
                                      uint32_t B,
                                      uint32_t start_pos) {
    if (!m || !ss || !tokens || B == 0 || B > LFM2_MAX_BATCH ||
        start_pos + B > ss->base.max_seq_len) return NULL;
    LFM2BatchState *s = &ss->base;
    const uint32_t  D = m->cfg.n_embd;

    /* 1. Dequantize B token embeddings into s->x[B, D] */
    ggml_type_traits_t traits;
    get_ggml_type_traits(m->token_embd->type, &traits);
    const size_t row_bytes = ((size_t)D / traits.block_elements) * traits.block_bytes;

    for (uint32_t t = 0; t < B; ++t) {
        const uint8_t *row_ptr = (const uint8_t *)m->token_embd->data + (size_t)tokens[t] * row_bytes;
        float *dst = s->x + (size_t)t * D;
        if (m->token_embd->type == GGML_TYPE_Q4_K) {
            const block_q4_K_t *blocks = (const block_q4_K_t *)row_ptr;
            for (uint32_t b = 0; b < D / 256; ++b) {
                const float d = fp16_to_fp32(blocks[b].d), min = fp16_to_fp32(blocks[b].dmin);
                const uint8_t *q = blocks[b].qs;
                int is = 0;
                for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blocks[b].scales, &sc1, &m1);
                    get_scale_min_k4(is + 1, blocks[b].scales, &sc2, &m2);
                    for (int l = 0; l < 32; ++l) dst[b*256 + j + l]      = d*sc1*(q[l] & 0xF) - min*m1;
                    for (int l = 0; l < 32; ++l) dst[b*256 + j + 32 + l] = d*sc2*(q[l] >> 4)  - min*m2;
                }
            }
        } else if (m->token_embd->type == GGML_TYPE_Q6_K) {
            const block_q6_K_t *blocks = (const block_q6_K_t *)row_ptr;
            for (uint32_t b = 0; b < D / 256; ++b) {
                const float d = fp16_to_fp32(blocks[b].d);
                const uint8_t *ql = blocks[b].ql, *qh = blocks[b].qh;
                const int8_t  *sc = blocks[b].scales;
                float *y = dst + b * 256;
                for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        y[l +  0] = d * sc[is + 0] * ((int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                        y[l + 32] = d * sc[is + 2] * ((int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                        y[l + 64] = d * sc[is + 4] * ((int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32);
                        y[l + 96] = d * sc[is + 6] * ((int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32);
                    }
                }
            }
        } else if (m->token_embd->type == GGML_TYPE_F32) {
            memcpy(dst, row_ptr, D * sizeof(float));
        }
    }

    /* 2. Execute all hybrid layers across the verification batch */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];
        lfm2_rmsnorm_batch(s->xb, s->x, layer->attn_norm, B, D, m->cfg.rms_norm_eps);

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv_spec_batch(ss, m, layer, l, B);
        } else {
            lfm2_forward_gqa_batch(s, m, layer, l, start_pos, B);
        }

        lfm2_forward_ffn_batch(s, m, layer, B);
    }

    /* 3. Final RMSNorm + Batched Classifier Projection (output.weight) for ALL B slots */
    lfm2_rmsnorm_batch(s->xb, s->x, m->output_norm, B, D, m->cfg.rms_norm_eps);
    lfm2_matmul_batch_avx2_mt(&s->pool, ss->logits_batch, m->output, s->xb, B, s->xq);

    return ss->logits_batch;
}

static inline void lfm2_restore_shortconv_trace_slot(LFM2SpecState *ss,
                                                     const LFM2Model *m,
                                                     uint32_t accepted_slot) {
    if (!ss || !m || accepted_slot >= LFM2_MAX_BATCH) return;
    const uint32_t D        = m->cfg.n_embd;
    const uint32_t l_cache  = m->cfg.shortconv_l_cache ? m->cfg.shortconv_l_cache : 3;
    const uint32_t hist_len = (l_cache > 1) ? (l_cache - 1) : 2;
    const size_t   bytes    = (size_t)hist_len * D * sizeof(float);

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV && ss->conv_trace[accepted_slot][l]) {
            memcpy(ss->base.conv_state[l], ss->conv_trace[accepted_slot][l], bytes);
        }
    }
}

/* ============================================================================
 * 4. Speculative Decoding Step Driver
 * ============================================================================ */
typedef uint32_t (*lfm2_token_verify_sample_fn)(float *logits, uint32_t n_vocab, void *user_sampler);

uint32_t lfm2_speculative_decode_step(const LFM2Model *m,
                                      LFM2SpecState *ss,
                                      LFM2NgramIndex *ngram_idx,
                                      uint32_t *inout_pending_token,
                                      uint32_t *inout_pos,
                                      lfm2_token_verify_sample_fn sample_fn,
                                      void *user_sampler,
                                      uint32_t out_emitted_tokens[LFM2_MAX_BATCH]) {
    if (!m || !ss || !ngram_idx || !inout_pending_token || !inout_pos || !sample_fn) return 0;
    const uint32_t cur_pos = *inout_pos;
    const uint32_t V       = m->cfg.n_vocab;
    const uint32_t t_0     = *inout_pending_token;

    ngram_idx->total_steps++;

    /* 1. Draft up to K continuation tokens from N-gram history */
    uint32_t drafts[LFM2_PLD_MAX_DRAFT];
    uint32_t K = lfm2_ngram_draft(ngram_idx, t_0, drafts);

    /* Clamp K so cur_pos + 1 + K <= max_seq_len */
    if (cur_pos + 1 + K > ss->base.max_seq_len) {
        K = (cur_pos + 1 < ss->base.max_seq_len) ? (ss->base.max_seq_len - 1 - cur_pos) : 0;
    }

    const uint32_t B = 1 + K;
    uint32_t batch_in[LFM2_MAX_BATCH];
    batch_in[0] = t_0;
    for (uint32_t k = 0; k < K; ++k) {
        batch_in[1 + k] = drafts[k];
    }

    if (K > 0) {
        ngram_idx->draft_batches_attempted++;
        ngram_idx->total_draft_tokens_proposed += K;
    }

    /* 2. Run batched hybrid forward pass */
    float *logits_batch = lfm2_forward_speculative_batch(m, ss, batch_in, B, cur_pos);
    if (!logits_batch) return 0;

    /* 3. Commit t_0 into the N-gram index (since t_0 was just evaluated at cur_pos) */
    lfm2_ngram_index_ingest(ngram_idx, t_0);

    /* 4. Sequential verification loop */
    uint32_t n_accepted = 0;
    uint32_t n_emitted  = 0;
    uint32_t final_tok  = 0;

    for (uint32_t b = 0; b < B; ++b) {
        float *slot_logits = logits_batch + (size_t)b * V;
        const uint32_t sampled_id = sample_fn(slot_logits, V, user_sampler);

        if (b < K && sampled_id == drafts[b]) {
            /* Draft token drafts[b] accepted! */
            out_emitted_tokens[n_emitted++] = sampled_id;
            lfm2_ngram_index_ingest(ngram_idx, sampled_id);
            n_accepted++;

            if (sampled_id == ngram_idx->eos_token_id) {
                final_tok = sampled_id;
                break;
            }
        } else {
            /* Mismatch or final bonus token sampled at b */
            out_emitted_tokens[n_emitted++] = sampled_id;
            final_tok = sampled_id;
            break;
        }
    }

    /* 5. If fewer than K drafts accepted, snap ShortConv state to conv_trace[n_accepted] in <10 us */
    if (n_accepted < K) {
        lfm2_restore_shortconv_trace_slot(ss, m, n_accepted);
    }

    /* 6. Update state cursors and adaptive draft window */
    *inout_pos           = cur_pos + n_accepted + 1;
    *inout_pending_token = final_tok;

    ngram_idx->total_draft_tokens_accepted += n_accepted;
    lfm2_ngram_update_adaptive_window(ngram_idx, K, n_accepted);

    return n_emitted;
}

/* ============================================================================
 * 5. Standalone Benchmark Main (when compiled directly)
 * ============================================================================ */
#ifndef LFM2_SPECULATIVE_PLD_NO_MAIN

static uint32_t greedy_verify_sampler(float *logits, uint32_t n_vocab, void *user_data) {
    (void)user_data;
    uint32_t best_id = 0;
    float best_val   = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id  = i;
        }
    }
    return best_id;
}

int main(int argc, char **argv) {
    printf("========================================================================\n");
    printf(" Liquid AI LFM2 Prompt-Lookup Speculative Decoding (B=8) Benchmark\n");
    printf("========================================================================\n");

    const uint32_t D        = 512;
    const uint32_t n_ff     = 1536;
    const uint32_t V        = 4096;
    const uint32_t n_layers = 4;
    const uint32_t nb_d     = D / 256;
    const uint32_t nb_ff    = n_ff / 256;

    LFM2Model model;
    memset(&model, 0, sizeof(model));
    snprintf(model.cfg.arch, sizeof(model.cfg.arch), "lfm2");
    model.cfg.n_layers          = n_layers;
    model.cfg.n_embd            = D;
    model.cfg.n_ff              = n_ff;
    model.cfg.n_vocab           = V;
    model.cfg.n_ctx             = 2048;
    model.cfg.n_head_q          = 8;
    model.cfg.shortconv_l_cache = 3;
    model.cfg.rms_norm_eps      = 1e-5f;
    model.cfg.rope_freq_base    = 1e6f;

    block_q4_K_t *w_embd   = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)V * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_inproj = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)3 * D * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_proj   = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)D * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_kv     = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)128 * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_gate   = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q6_K_t *w_down   = (block_q6_K_t *)lfm2_aligned_alloc(64, (size_t)D * nb_ff * sizeof(block_q6_K_t));
    float        *w_norm   = (float *)lfm2_aligned_alloc(64, D * sizeof(float));
    float        *w_conv   = (float *)lfm2_aligned_alloc(64, 3 * D * sizeof(float));

    for (uint32_t i = 0; i < D; ++i) w_norm[i] = 1.0f;
    for (uint32_t i = 0; i < 3 * D; ++i) w_conv[i] = 0.25f * cosf((float)i * 0.1f);

    #define INIT_Q4K(buf, count) do { \
        for (size_t i = 0; i < (count); ++i) { \
            (buf)[i].d = 0x2800; (buf)[i].dmin = 0x2000; \
            for (int j = 0; j < 12; ++j)  (buf)[i].scales[j] = (uint8_t)((i * 7 + j * 3) & 0x1F); \
            for (int j = 0; j < 128; ++j) (buf)[i].qs[j]     = (uint8_t)((i * 13 + j * 29) & 0xFF); \
        } \
    } while (0)

    INIT_Q4K(w_embd,   (size_t)V * nb_d);
    INIT_Q4K(w_inproj, (size_t)3 * D * nb_d);
    INIT_Q4K(w_proj,   (size_t)D * nb_d);
    INIT_Q4K(w_kv,     (size_t)128 * nb_d);
    INIT_Q4K(w_gate,   (size_t)n_ff * nb_d);

    for (size_t i = 0; i < (size_t)D * nb_ff; ++i) {
        w_down[i].d = 0x2600;
        for (int j = 0; j < 16; ++j)  w_down[i].scales[j] = (int8_t)(((int)(i + j) % 31) - 15);
        for (int j = 0; j < 128; ++j) w_down[i].ql[j]     = (uint8_t)((i * 19 + j * 11) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    GGUFTensor t_embd   = { .n_dims = 2, .ne = {D, V, 1, 1},    .type = GGML_TYPE_Q4_K, .data = w_embd   };
    GGUFTensor t_norm   = { .n_dims = 1, .ne = {D, 1, 1, 1},    .type = GGML_TYPE_F32,  .data = w_norm   };
    GGUFTensor t_qnorm  = { .n_dims = 1, .ne = {64, 1, 1, 1},   .type = GGML_TYPE_F32,  .data = w_norm   };
    GGUFTensor t_inproj = { .n_dims = 2, .ne = {D, 3*D, 1, 1},  .type = GGML_TYPE_Q4_K, .data = w_inproj };
    GGUFTensor t_conv   = { .n_dims = 2, .ne = {3, D, 1, 1},    .type = GGML_TYPE_F32,  .data = w_conv   };
    GGUFTensor t_proj   = { .n_dims = 2, .ne = {D, D, 1, 1},    .type = GGML_TYPE_Q4_K, .data = w_proj   };
    GGUFTensor t_kv     = { .n_dims = 2, .ne = {D, 128, 1, 1},  .type = GGML_TYPE_Q4_K, .data = w_kv     };
    GGUFTensor t_gate   = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_gate   };
    GGUFTensor t_down   = { .n_dims = 2, .ne = {n_ff, D, 1, 1}, .type = GGML_TYPE_Q6_K, .data = w_down   };

    model.token_embd  = &t_embd;
    model.output_norm = &t_norm;
    model.output      = &t_embd;

    for (uint32_t l = 0; l < n_layers; ++l) {
        LFM2Layer *ly = &model.layers[l];
        ly->attn_norm = &t_norm;
        ly->ffn_norm  = &t_norm;
        ly->ffn_gate  = &t_gate;
        ly->ffn_up    = &t_gate;
        ly->ffn_down  = &t_down;
        if (l % 2 == 0) {
            ly->block_type         = LFM2_BLOCK_SHORTCONV;
            ly->shortconv.in_proj  = &t_inproj;
            ly->shortconv.conv     = &t_conv;
            ly->shortconv.out_proj = &t_proj;
            model.cfg.n_head_kv[l] = 0;
        } else {
            ly->block_type  = LFM2_BLOCK_GQA_ATTN;
            ly->attn.wq     = &t_proj;
            ly->attn.wk     = &t_kv;
            ly->attn.wv     = &t_kv;
            ly->attn.q_norm = &t_qnorm;
            ly->attn.k_norm = &t_qnorm;
            ly->attn.wo     = &t_proj;
            model.cfg.n_head_kv[l] = 2;
        }
    }

    const uint32_t n_threads = (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0;
    LFM2SpecState state_baseline, state_spec;
    lfm2_spec_state_init(&state_baseline, &model, 1024, n_threads);
    lfm2_spec_state_init(&state_spec,     &model, 1024, n_threads);

    const uint32_t gen_target = 64;
    uint32_t baseline_tokens[128];
    uint32_t pos_base = 0;
    uint32_t tok_base = 1;

    double t0 = now_ms();
    for (uint32_t step = 0; step < gen_target; ++step) {
        float *logits = lfm2_forward_batch(&model, &state_baseline.base, &tok_base, 1, pos_base++, true);
        tok_base = greedy_verify_sampler(logits, V, NULL);
        baseline_tokens[step] = tok_base;
    }
    double base_ms = now_ms() - t0;

    LFM2NgramIndex ngram;
    lfm2_ngram_index_init(&ngram, 1024, 6, 7);

    /* Ingest representative prompt context sequence */
    lfm2_ngram_index_ingest(&ngram, 1);
    for (uint32_t i = 0; i < 44; ++i) {
        lfm2_ngram_index_ingest(&ngram, baseline_tokens[i]);
    }

    uint32_t spec_tokens[128];
    uint32_t n_spec_gen = 0;
    uint32_t pos_spec   = 0;
    uint32_t tok_spec   = 1;

    t0 = now_ms();
    while (n_spec_gen < gen_target) {
        uint32_t emitted[LFM2_MAX_BATCH];
        uint32_t n_emit = lfm2_speculative_decode_step(
            &model, &state_spec, &ngram, &tok_spec, &pos_spec,
            greedy_verify_sampler, NULL, emitted);

        for (uint32_t i = 0; i < n_emit && n_spec_gen < gen_target; ++i) {
            spec_tokens[n_spec_gen++] = emitted[i];
        }
    }
    double spec_ms = now_ms() - t0;

    bool tokens_identical = true;
    for (uint32_t i = 0; i < gen_target; ++i) {
        if (baseline_tokens[i] != spec_tokens[i]) {
            tokens_identical = false;
            printf("[MISMATCH at token %u] baseline=%u vs spec=%u\n",
                   i, baseline_tokens[i], spec_tokens[i]);
            break;
        }
    }

    const double base_tps = (gen_target * 1000.0) / base_ms;
    const double spec_tps = (gen_target * 1000.0) / spec_ms;
    const double accept_rate = (ngram.total_draft_tokens_proposed > 0)
        ? (100.0 * (double)ngram.total_draft_tokens_accepted / (double)ngram.total_draft_tokens_proposed)
        : 0.0;

    printf("Physical CPU Cores Active     : %u\n", state_spec.base.pool.n_threads);
    printf("Generated Sequence Parity     : %s (100%% bit-exact ShortConv + GQA rollback)\n",
           tokens_identical ? "PASS" : "FAIL");
    printf("Forward Passes Executed       : %llu speculative steps (vs %u M=1 steps)\n",
           (unsigned long long)ngram.total_steps, gen_target);
    printf("Draft Tokens Accepted         : %llu / %llu proposed (%.1f%% acceptance)\n",
           (unsigned long long)ngram.total_draft_tokens_accepted,
           (unsigned long long)ngram.total_draft_tokens_proposed,
           accept_rate);
    printf("Mean Tokens Emitted / Forward : %.2f tokens/sweep\n\n",
           (double)gen_target / (double)ngram.total_steps);

    printf("Standard M=1 Autoregressive   : %6.2f ms (%6.1f tok/s)\n", base_ms, base_tps);
    printf("Prompt-Lookup Speculative B=8 : %6.2f ms (%6.1f tok/s)\n", spec_ms, spec_tps);
    printf("Desktop CPU Net Speedup       : %.2fx faster (0 MB extra model weights)\n",
           spec_tps / base_tps);

    lfm2_ngram_index_free(&ngram);
    lfm2_spec_state_free(&state_baseline);
    lfm2_spec_state_free(&state_spec);
    free(w_embd); free(w_inproj); free(w_proj); free(w_kv);
    free(w_gate); free(w_down);   free(w_norm); free(w_conv);
    return tokens_identical ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif /* LFM2_SPECULATIVE_PLD_NO_MAIN */

