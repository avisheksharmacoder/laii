#ifndef LFM_ENGINE_H
#define LFM_ENGINE_H

#include "lfm_defaults.h"

#include <stdint.h>
#include <stdbool.h>
#include "lfm_model.h"
#include "lfm_tokenizer.h"
#include "lfm_forward.h"
#include "lfm_threadpool.h"
#include "arena.h"

typedef struct {
    float temperature;       // 0.0f = greedy; default 0.30f for LFM2
    float min_p;             // Min-P cutoff (default 0.15f for LFM2; 0.0 = disabled)
    uint32_t top_k;          // Top-K cutoff (default 50 for LFM2)
    float rep_penalty;       // Repetition penalty (default 1.05f for LFM2; 1.0 = disabled)
    uint32_t rep_window;     // Window size for repetition penalty (default 64)
    uint32_t max_new_tokens; /* 0 = LFM_DEFAULT_MAX_TOKENS (2000), UINT32_MAX = unlimited */
    uint32_t seed;
    bool echo_prompt;
    volatile bool *stop_signal; // Optional early stop signal (SIGINT or stop-string match)
    const char *prompt_cache_path; // Optional path to disk-backed hybrid prompt cache (.lfmcache)
} LFMSamplerConfig;


typedef struct {
    LFMModel model;
    LFMTokenizer tokenizer;
    LFMState state;
    ThreadPool threadpool;

    MemoryArena state_arena;   // Persistent memory for KV-cache and conv states
    MemoryArena scratch_arena; // Per-token reset arena for intermediate activations
} LFMEngine;

typedef void (*LFMTokenCallback)(uint32_t token_id, const char *piece, void *user_data);

// Initialize engine from a GGUF model file with specified thread count (0 = auto-detect)
bool lfm_engine_init(LFMEngine *engine, const char *model_path, uint32_t num_threads);

// Release engine memory
void lfm_engine_free(LFMEngine *engine);

// Generate tokens for a given prompt
size_t lfm_engine_generate(
    LFMEngine *engine,
    const char *prompt,
    const LFMSamplerConfig *sampler,
    LFMTokenCallback callback,
    void *user_data
);

#endif // LFM_ENGINE_H
