#define _POSIX_C_SOURCE 200809L
#include "lfm_engine.h"
#include "lfm_defaults.h"
#include "lfm_sys_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t id;
    float val;
} TokenCandidate;

static int cmp_candidates_desc(const void *a, const void *b) {
    float va = ((const TokenCandidate*)a)->val;
    float vb = ((const TokenCandidate*)b)->val;
    return (va < vb) - (va > vb);
}

static uint32_t sample_token(
    float *logits,
    uint32_t vocab_size,
    float temperature,
    float min_p,
    uint32_t top_k,
    float rep_penalty,
    const uint32_t *history,
    uint32_t history_len,
    uint32_t *rng_state
) {
    // 1. Sliding-window repetition penalty
    if (rep_penalty > 1.0001f && history && history_len > 0) {
        for (uint32_t i = 0; i < history_len; ++i) {
            uint32_t tid = history[i];
            if (tid >= vocab_size) continue;
            bool already = false;
            for (uint32_t j = 0; j < i; ++j) {
                if (history[j] == tid) { already = true; break; }
            }
            if (!already) {
                if (logits[tid] > 0.0f) logits[tid] /= rep_penalty;
                else logits[tid] *= rep_penalty;
            }
        }
    }

    // 2. Global max logit & greedy check
    uint32_t best_id = 0;
    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best_id = i;
        }
    }

    if (temperature <= 0.001f || (top_k <= 1 && min_p <= 0.0f)) {
        return best_id;
    }

    // 3. Logit-space Min-P cutoff: z_cutoff = max_logit + temperature * ln(min_p)
    float logit_cutoff = -1e30f;
    if (min_p > 0.0f && min_p < 1.0f) {
        logit_cutoff = max_logit + temperature * logf(min_p);
    }

    TokenCandidate *cands = (TokenCandidate*)malloc(vocab_size * sizeof(TokenCandidate));
    if (!cands) return best_id;

    uint32_t n_cands = 0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (logits[i] >= logit_cutoff) {
            cands[n_cands].id = i;
            cands[n_cands].val = logits[i];
            n_cands++;
        }
    }

    if (n_cands <= 1) {
        free(cands);
        return best_id;
    }

    qsort(cands, n_cands, sizeof(TokenCandidate), cmp_candidates_desc);
    if (top_k > 0 && n_cands > top_k) n_cands = top_k;

    // 4. Softmax
    float inv_temp = 1.0f / temperature;
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < n_cands; ++i) {
        float p = expf((cands[i].val - max_logit) * inv_temp);
        cands[i].val = p;
        sum_exp += p;
    }

    // 5. Sampling via PCG / xorshift
    *rng_state ^= (*rng_state << 13);
    *rng_state ^= (*rng_state >> 17);
    *rng_state ^= (*rng_state << 5);
    float r = ((float)(*rng_state & 0x00FFFFFF) / (float)0x01000000) * sum_exp;

    float cdf = 0.0f;
    uint32_t selected = cands[n_cands - 1].id;
    for (uint32_t i = 0; i < n_cands; ++i) {
        cdf += cands[i].val;
        if (r <= cdf) {
            selected = cands[i].id;
            break;
        }
    }

    free(cands);
    return selected;
}


bool lfm_engine_init(LFMEngine *engine, const char *model_path, uint32_t num_threads) {
    if (!engine || !model_path) return false;
    memset(engine, 0, sizeof(LFMEngine));

    // 1. Load model weights and config
    if (!lfm_model_load(&engine->model, model_path)) {
        return false;
    }

    // 2. Load tokenizer
    lfm_tokenizer_load(&engine->tokenizer, &engine->model.gguf);

    // 3. Initialize ThreadPool for parallel SIMD execution
    if (!threadpool_init(&engine->threadpool, num_threads)) {
        lfm_model_free(&engine->model);
        return false;
    }

    // 4. Compute required state memory
    const LFMConfig *cfg = &engine->model.config;
    size_t state_mem_needed = 1024 * 1024; // 1 MB base overhead
    for (uint32_t i = 0; i < cfg->n_layers; ++i) {
        if (cfg->is_conv_layer[i]) {
            state_mem_needed += 2 * cfg->hidden_dim * sizeof(float);
        } else {
            state_mem_needed += 2 * (size_t)cfg->max_seq_len * cfg->n_heads_kv * cfg->head_dim * sizeof(float);
        }
    }

    if (!arena_init(&engine->state_arena, state_mem_needed)) {
        threadpool_free(&engine->threadpool);
        lfm_model_free(&engine->model);
        return false;
    }

    // Scratch arena: 32 MB for activation buffers, SwiGLU, logits
    size_t scratch_size = 32 * 1024 * 1024;
    if (!arena_init(&engine->scratch_arena, scratch_size)) {
        arena_free(&engine->state_arena);
        threadpool_free(&engine->threadpool);
        lfm_model_free(&engine->model);
        return false;
    }

    // 5. Initialize forward pass recurrent and KV states
    if (!lfm_state_init(&engine->state, cfg, &engine->state_arena)) {
        lfm_engine_free(engine);
        return false;
    }

    return true;
}

void lfm_engine_free(LFMEngine *engine) {
    if (!engine) return;
    threadpool_free(&engine->threadpool);
    arena_free(&engine->state_arena);
    arena_free(&engine->scratch_arena);
    lfm_tokenizer_free(&engine->tokenizer);
    lfm_model_free(&engine->model);
}

size_t lfm_engine_generate(
    LFMEngine *engine,
    const char *prompt,
    const LFMSamplerConfig *sampler,
    LFMTokenCallback callback,
    void *user_data
) {
    if (!engine || !prompt) return 0;

    const LFMConfig *cfg = &engine->model.config;
    lfm_state_reset(&engine->state, cfg);

    // Tokenize prompt
    uint32_t max_prompt_tokens = 2048;
    uint32_t *tokens = (uint32_t*)malloc(max_prompt_tokens * sizeof(uint32_t));
    size_t n_prompt = lfm_tokenizer_encode(&engine->tokenizer, prompt, tokens, max_prompt_tokens, true);

    if (n_prompt == 0) {
        tokens[0] = engine->tokenizer.bos_id;
        n_prompt = 1;
    }

    uint32_t rng = sampler && sampler->seed ? sampler->seed : 1337;
    float temp = sampler ? sampler->temperature : 0.30f;
    float min_p = sampler ? sampler->min_p : 0.0f;
    uint32_t top_k = sampler ? sampler->top_k : 50;
    float rep_penalty = sampler && sampler->rep_penalty > 0.0f ? sampler->rep_penalty : 1.0f;
    uint32_t rep_win = sampler && sampler->rep_window > 0 ? sampler->rep_window : 64;
    uint32_t max_gen = sampler && sampler->max_new_tokens ? sampler->max_new_tokens : (uint32_t)LFM_DEFAULT_MAX_TOKENS;

    uint32_t *history = (uint32_t*)calloc(rep_win, sizeof(uint32_t));
    uint32_t hist_count = 0;

    float *logits = (float*)malloc(cfg->vocab_size * sizeof(float));
    size_t tokens_generated = 0;

    // Setup adapter and prompt cache if requested
    LFM2Model adapter_m;
    memset(&adapter_m, 0, sizeof(adapter_m));
    snprintf(adapter_m.cfg.arch, sizeof(adapter_m.cfg.arch), "%s", cfg->architecture);
    adapter_m.cfg.n_layers = cfg->n_layers;
    adapter_m.cfg.n_embd = cfg->hidden_dim;
    adapter_m.cfg.n_vocab = cfg->vocab_size;
    adapter_m.cfg.n_ctx = cfg->max_seq_len;
    adapter_m.cfg.n_head_q = cfg->n_heads_q;
    adapter_m.cfg.shortconv_l_cache = cfg->conv_L_cache;
    for (uint32_t l = 0; l < cfg->n_layers; ++l) {
        adapter_m.layers[l].block_type = cfg->is_conv_layer[l] ? LFM2_BLOCK_SHORTCONV : LFM2_BLOCK_GQA_ATTN;
        adapter_m.cfg.n_head_kv[l] = cfg->is_conv_layer[l] ? 0 : cfg->layer_heads_kv[l];
    }

    float *conv_slabs[LFM_MAX_LAYERS] = {0};
    float *k_ptrs[LFM_MAX_LAYERS] = {0};
    float *v_ptrs[LFM_MAX_LAYERS] = {0};
    for (uint32_t l = 0; l < cfg->n_layers; ++l) {
        if (cfg->is_conv_layer[l]) {
            conv_slabs[l] = (float *)calloc(2 * cfg->hidden_dim, sizeof(float));
        } else {
            k_ptrs[l] = engine->state.kv_caches[l].k_cache;
            v_ptrs[l] = engine->state.kv_caches[l].v_cache;
        }
    }

    uint32_t matched_pos = 0;
    bool cache_loaded = false;
    if (sampler && sampler->prompt_cache_path) {
        cache_loaded = lfm2_prompt_cache_try_load_prefix_f32(
            sampler->prompt_cache_path, &adapter_m, tokens, (uint32_t)n_prompt,
            cfg->max_seq_len, conv_slabs, k_ptrs, v_ptrs, &matched_pos);
        if (cache_loaded && matched_pos > 0) {
            for (uint32_t l = 0; l < cfg->n_layers; ++l) {
                if (cfg->is_conv_layer[l] && conv_slabs[l]) {
                    memcpy(engine->state.conv_states[l].s0, conv_slabs[l], cfg->hidden_dim * sizeof(float));
                    memcpy(engine->state.conv_states[l].s1, conv_slabs[l] + cfg->hidden_dim, cfg->hidden_dim * sizeof(float));
                }
            }
            if (matched_pos == n_prompt) {
                engine->state.current_pos = matched_pos - 1;
            } else {
                engine->state.current_pos = matched_pos;
            }
            for (uint32_t i = 0; i < (matched_pos == n_prompt ? matched_pos - 1 : matched_pos); ++i) {
                if (history) {
                    history[hist_count % rep_win] = tokens[i];
                    hist_count++;
                }
            }
        }
    }

    // 1. Ingest prompt (Prefill)
    size_t start_i = (cache_loaded && matched_pos == n_prompt) ? (n_prompt - 1) : matched_pos;
    for (size_t i = start_i; i < n_prompt; ++i) {
        if (history) {
            history[hist_count % rep_win] = tokens[i];
            hist_count++;
        }
        arena_reset(&engine->scratch_arena);
        lfm_forward_token(logits, tokens[i], &engine->model, &engine->state, &engine->threadpool, &engine->scratch_arena);

        if (sampler && sampler->echo_prompt && callback) {
            const char *piece = lfm_tokenizer_decode_id(&engine->tokenizer, tokens[i]);
            callback(tokens[i], piece, user_data);
        }
    }

    // Save prompt cache if requested and was not loaded from disk
    if (sampler && sampler->prompt_cache_path && !cache_loaded) {
        for (uint32_t l = 0; l < cfg->n_layers; ++l) {
            if (cfg->is_conv_layer[l] && conv_slabs[l]) {
                memcpy(conv_slabs[l], engine->state.conv_states[l].s0, cfg->hidden_dim * sizeof(float));
                memcpy(conv_slabs[l] + cfg->hidden_dim, engine->state.conv_states[l].s1, cfg->hidden_dim * sizeof(float));
            }
        }
        lfm2_prompt_cache_save_f32(sampler->prompt_cache_path, &adapter_m, tokens, (uint32_t)n_prompt,
                                   conv_slabs, k_ptrs, v_ptrs);
    }

    for (uint32_t l = 0; l < cfg->n_layers; ++l) {
        free(conv_slabs[l]);
    }

    // 2. Autoregressive Generation
    uint32_t cur_hist_len = hist_count < rep_win ? hist_count : rep_win;
    uint32_t next_token = sample_token(logits, cfg->vocab_size, temp, min_p, top_k, rep_penalty, history, cur_hist_len, &rng);

    while ((max_gen == 0 || tokens_generated < max_gen) && engine->state.current_pos < cfg->max_seq_len) {
        if (sampler && sampler->stop_signal && *(sampler->stop_signal)) {
            break;
        }
        if (next_token == engine->tokenizer.eos_id) {
            break;
        }

        const char *piece = lfm_tokenizer_decode_id(&engine->tokenizer, next_token);
        if (callback) {
            callback(next_token, piece, user_data);
        }
        tokens_generated++;

        if (sampler && sampler->stop_signal && *(sampler->stop_signal)) {
            break;
        }

        if (history) {
            history[hist_count % rep_win] = next_token;
            hist_count++;
        }

        arena_reset(&engine->scratch_arena);
        lfm_forward_token(logits, next_token, &engine->model, &engine->state, &engine->threadpool, &engine->scratch_arena);
        cur_hist_len = hist_count < rep_win ? hist_count : rep_win;
        next_token = sample_token(logits, cfg->vocab_size, temp, min_p, top_k, rep_penalty, history, cur_hist_len, &rng);
    }

    free(history);
    free(tokens);
    free(logits);
    return tokens_generated;
}

