#ifndef LFM_MODEL_H
#define LFM_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "gguf.h"

#define LFM_MAX_LAYERS 64

typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t ffn_dim;
    uint32_t n_layers;
    uint32_t n_heads_q;
    uint32_t n_heads_kv;
    uint32_t head_dim;
    uint32_t conv_L_cache; // typically 3 (ring-buffer length = 2)
    uint32_t max_seq_len;
    float norm_eps;
    float rope_theta;

    char architecture[32];
    bool is_conv_layer[LFM_MAX_LAYERS];
    uint32_t layer_heads_kv[LFM_MAX_LAYERS];
} LFMConfig;

typedef struct {
    // Shared layer norms
    const float *operator_norm;
    const float *ffn_norm;

    // ShortConv layer weights (if is_conv_layer == true)
    const void *shortconv_in_proj;      // [3 * hidden_dim, hidden_dim]
    GGMLType shortconv_in_proj_type;
    const float *shortconv_conv;        // [hidden_dim, conv_L_cache] (or k=3 weights)
    const float *shortconv_conv_bias;   // [hidden_dim] (optional)
    const void *shortconv_out_proj;     // [hidden_dim, hidden_dim]
    GGMLType shortconv_out_proj_type;

    // Attention layer weights (if is_conv_layer == false)
    const void *q_proj;                 // [n_heads_q * head_dim, hidden_dim]
    GGMLType q_proj_type;
    const void *k_proj;                 // [n_heads_kv * head_dim, hidden_dim]
    GGMLType k_proj_type;
    const void *v_proj;                 // [n_heads_kv * head_dim, hidden_dim]
    GGMLType v_proj_type;
    const float *q_norm;                // [head_dim]
    const float *k_norm;                // [head_dim]
    const void *attn_out_proj;          // [hidden_dim, hidden_dim]
    GGMLType attn_out_proj_type;

    // FFN weights (SwiGLU)
    const void *ffn_gate;               // w1: [ffn_dim, hidden_dim]
    GGMLType ffn_gate_type;
    const void *ffn_up;                 // w3: [ffn_dim, hidden_dim]
    GGMLType ffn_up_type;
    const void *ffn_down;               // w2: [hidden_dim, ffn_dim]
    GGMLType ffn_down_type;
} LFMLayerWeights;

typedef struct {
    LFMConfig config;

    // Global weights
    const void *tok_embeddings;
    GGMLType tok_embeddings_type;
    const float *output_norm;
    const void *lm_head;
    GGMLType lm_head_type;

    // Layer weights
    LFMLayerWeights layers[LFM_MAX_LAYERS];

    // Backing GGUF file
    GGUFFile gguf;
} LFMModel;

// Load and initialize LFM2 model weights and config from GGUF file
bool lfm_model_load(LFMModel *model, const char *filepath);

// Free model resources
void lfm_model_free(LFMModel *model);

#endif // LFM_MODEL_H
