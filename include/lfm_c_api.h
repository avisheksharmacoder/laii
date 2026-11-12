/*
 * include/lfm_c_api.h - Legacy Public C ABI for LFMEngine
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Avishek and Contributors
 *
 * Independent open-source inference engine compatible with Liquid AI's LFM2 architecture.
 * Not affiliated with, endorsed by, or sponsored by Liquid AI, Inc.
 */

#ifndef LFM_C_API_H
#define LFM_C_API_H

#include "lfm_defaults.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(_WIN32)
  #ifdef LFM_EXPORTS
    #define LFM_API __declspec(dllexport)
  #else
    #define LFM_API __declspec(dllimport)
  #endif
#else
  #define LFM_API __attribute__((visibility("default")))
#endif

// Opaque engine handle
typedef struct LFMEngine lfm_engine_t;

// Generation sampling parameters
typedef struct {
    float temperature;
    uint32_t top_k;
    uint32_t max_new_tokens; /* 0 = LFM_DEFAULT_MAX_TOKENS (2000), UINT32_MAX = unlimited */
    uint32_t seed;
    bool echo_prompt;
} lfm_sampler_params_t;

// Token streaming callback
typedef void (*lfm_token_callback_t)(uint32_t token_id, const char *piece, void *user_data);

// Engine lifecycle
LFM_API lfm_engine_t* lfm_create(const char *model_path, int num_threads);
LFM_API void lfm_destroy(lfm_engine_t *engine);

// High-level text generation
// Returns the number of tokens generated, or negative on error
LFM_API int lfm_generate(
    lfm_engine_t *engine,
    const char *prompt,
    char *out_buf,
    size_t out_buf_size,
    const lfm_sampler_params_t *params
);

// Streaming text generation
LFM_API int lfm_generate_stream(
    lfm_engine_t *engine,
    const char *prompt,
    const lfm_sampler_params_t *params,
    lfm_token_callback_t callback,
    void *user_data
);

// Tokenizer functions
// Returns number of tokens written to tokens_out, or negative on error
LFM_API int lfm_tokenize(
    lfm_engine_t *engine,
    const char *text,
    int32_t *tokens_out,
    int max_tokens
);

// Detokenize a single token into a string buffer
// Returns bytes written, or negative on error
LFM_API int lfm_detokenize(
    lfm_engine_t *engine,
    int32_t token,
    char *buf_out,
    size_t buf_size
);

// Model metadata getters
LFM_API uint32_t lfm_get_hidden_dim(const lfm_engine_t *engine);
LFM_API uint32_t lfm_get_num_layers(const lfm_engine_t *engine);
LFM_API uint32_t lfm_get_vocab_size(const lfm_engine_t *engine);
LFM_API const char* lfm_get_architecture(const lfm_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif // LFM_C_API_H
