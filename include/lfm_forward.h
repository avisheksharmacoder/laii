#ifndef LFM_FORWARD_H
#define LFM_FORWARD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "lfm_model.h"
#include "arena.h"

// State buffer for a single LIV ShortConv layer (k=3 means 2 previous tokens cached)
typedef struct {
    float *s0; // [hidden_dim] state at t-2
    float *s1; // [hidden_dim] state at t-1
} LFMConvState;

// KV cache for a single Attention layer
typedef struct {
    float *k_cache; // [max_seq_len, n_heads_kv, head_dim]
    float *v_cache; // [max_seq_len, n_heads_kv, head_dim]
} LFMKVCache;

// Context holding all recurrent and KV state across layers
typedef struct {
    uint32_t current_pos;
    uint32_t max_seq_len;
    LFMConvState conv_states[LFM_MAX_LAYERS];
    LFMKVCache kv_caches[LFM_MAX_LAYERS];
} LFMState;

// Allocate and zero-initialize state for a given model
bool lfm_state_init(LFMState *state, const LFMConfig *config, MemoryArena *arena);

// Reset state back to position 0 (clearing conv buffers and KV cache pointers)
void lfm_state_reset(LFMState *state, const LFMConfig *config);

// Math primitives
void lfm_rmsnorm(float *out, const float *x, const float *weight, size_t dim, float eps);
void lfm_silu_vec(float *out, const float *x, size_t dim);
void lfm_rope_neox(float *vec, uint32_t head_dim, uint32_t pos, float theta);
void lfm_matvec(float *out, const void *weight, GGMLType w_type, const float *x, size_t out_dim, size_t in_dim);

// LIV ShortConv step for a single token
void lfm_liv_shortconv_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    LFMConvState *conv_state,
    uint32_t hidden_dim,
    MemoryArena *scratch
);

// LIV ShortConv prefill over T tokens
void lfm_liv_shortconv_prefill(
    float *out_seq,               // [seq_len, hidden_dim]
    const float *x_norm_seq,       // [seq_len, hidden_dim]
    const LFMLayerWeights *layer,
    LFMConvState *conv_state,
    uint32_t hidden_dim,
    uint32_t seq_len,
    MemoryArena *scratch
);

// Grouped Query Attention step for a single token
void lfm_gqa_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    LFMKVCache *kv_cache,
    uint32_t pos,
    const LFMConfig *config,
    MemoryArena *scratch
);

// SwiGLU FFN step
void lfm_swiglu_step(
    float *out,
    const float *x_norm,
    const LFMLayerWeights *layer,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    MemoryArena *scratch
);

struct ThreadPool;

// Full model forward pass for a single token
void lfm_forward_token(
    float *logits,              // [vocab_size]
    uint32_t token_id,
    LFMModel *model,
    LFMState *state,
    struct ThreadPool *pool,
    MemoryArena *scratch
);

#endif // LFM_FORWARD_H
