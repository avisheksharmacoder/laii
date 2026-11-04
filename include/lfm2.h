/*
 * include/lfm2.h - Public C11 ABI for the LFM2 Inference Engine
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Avishek and Contributors
 *
 * Independent open-source inference engine compatible with Liquid AI's LFM2 architecture.
 * Not affiliated with, endorsed by, or sponsored by Liquid AI, Inc.
 */

#ifndef LFM2_H
#define LFM2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define LFM2_API __attribute__((visibility("default")))
#else
    #define LFM2_API
#endif

#define LFM2_ABI_VERSION 100U /* v1.0.0 */

/* ============================================================================
 * 1. Opaque Engine Handles
 * ============================================================================ */
typedef struct lfm2_model      lfm2_model_t;
typedef struct lfm2_context    lfm2_context_t;
typedef struct lfm2_checkpoint lfm2_checkpoint_t;

/* ============================================================================
 * 2. Status Codes & Enums
 * ============================================================================ */
typedef enum {
    LFM2_OK                  =  0, /* Success / UTF-8 chunk ready               */
    LFM2_DONE                =  1, /* Generation finished (<|im_end|> or stop)  */
    LFM2_ERR_INVALID_ARG     = -1,
    LFM2_ERR_IO              = -2,
    LFM2_ERR_CORRUPT_GGUF    = -3,
    LFM2_ERR_OOM             = -4,
    LFM2_ERR_CTX_FULL        = -5,
    LFM2_ERR_CHECKPOINT      = -6
} lfm2_status_t;

typedef enum {
    LFM2_GRAMMAR_NONE        = 0,  /* Unconstrained generation                  */
    LFM2_GRAMMAR_JSON_OBJECT = 1,  /* Strict RFC 8259 JSON object               */
    LFM2_GRAMMAR_TOOL_CALL   = 2   /* LFM2 Pythonic/JSON <|tool_call_start|>    */
} lfm2_grammar_type_t;

typedef enum {
    LFM2_POOL_LAST_TOKEN     = 0,  /* Normalized final token state (causal)     */
    LFM2_POOL_MEAN           = 1   /* Mean-pooled normalized states + L2 norm   */
} lfm2_pooling_type_t;

/* ============================================================================
 * 3. Fixed-Width POD Configuration & Introspection Structs
 * ============================================================================ */
typedef struct {
    char     arch[32];             /* "lfm2" or "lfm2moe"                       */
    char     active_isa[32];       /* "AVX2 + FMA + F16C" or "SCALAR"           */
    uint32_t n_layers;
    uint32_t n_shortconv_layers;   /* e.g. 10 in 1.2B, 18 in 8B-A1B             */
    uint32_t n_gqa_layers;         /* e.g. 6                                    */
    uint32_t n_embd;               /* e.g. 2048                                 */
    uint32_t n_ff;                 /* Dense FFN dim                             */
    uint32_t n_vocab;              /* e.g. 65536                                */
    uint32_t n_ctx_train;          /* e.g. 32768 or 128000                      */
    uint32_t n_experts;            /* 0 for dense, 32 for 8B-A1B, 64 for 24B    */
    uint32_t n_experts_used;       /* 0 for dense, 4 for MoE                    */
    uint64_t file_size_bytes;
} lfm2_model_info_t;

typedef struct {
    uint32_t max_seq_len;          /* 0 = auto-plan via /proc/meminfo (4096)    */
    uint32_t n_threads;            /* 0 = auto-detect physical CPU cores        */
    uint32_t n_sink_tokens;        /* Attention-sink tokens kept on KV shift(64)*/
    int32_t  auto_clamp_ram;       /* 1 = clamp max_seq_len to prevent Linux OOM*/
    int32_t  warmup_mmap;          /* 1 = sequential 4KB page warmup on load    */
} lfm2_context_params_t;

typedef struct {
    float    temperature;          /* Default: 0.30f (0.0f = greedy argmax)     */
    float    min_p;                /* Default: 0.15f                            */
    uint32_t top_k;                /* Default: 50                               */
    float    rep_penalty;          /* Default: 1.05f                            */
    uint32_t rep_window;           /* Default: 64                               */
    uint64_t seed;                 /* 0 = random hardware clock seed            */
    int32_t  grammar_type;         /* lfm2_grammar_type_t                       */
    uint32_t max_think_tokens;     /* 0 = unlimited; >0 forces \n</think>\n     */
} lfm2_sample_params_t;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t current_pos;
    double   prefill_ms;
    double   decode_ms;
    double   prefill_tok_per_sec;
    double   decode_tok_per_sec;
} lfm2_perf_stats_t;

/* Default parameter initializers */
LFM2_API lfm2_context_params_t lfm2_context_default_params(void);
LFM2_API lfm2_sample_params_t  lfm2_sample_default_params(void);

/* ============================================================================
 * 4. Model Lifecycle & Introspection API
 * ============================================================================ */
LFM2_API lfm2_model_t *lfm2_model_load(const char *gguf_path, int32_t warmup_pages);
LFM2_API void          lfm2_model_free(lfm2_model_t *model);
LFM2_API lfm2_status_t lfm2_model_get_info(const lfm2_model_t *model, lfm2_model_info_t *out_info);

/* ============================================================================
 * 5. Stateful Inference Context & Multi-Turn Chat API
 * ============================================================================ */
LFM2_API lfm2_context_t *lfm2_context_create(const lfm2_model_t *model,
                                             const lfm2_context_params_t *params);
LFM2_API void            lfm2_context_free(lfm2_context_t *ctx);
LFM2_API void            lfm2_context_reset(lfm2_context_t *ctx);
LFM2_API uint32_t        lfm2_context_get_pos(const lfm2_context_t *ctx);
LFM2_API void            lfm2_context_get_stats(const lfm2_context_t *ctx,
                                                lfm2_perf_stats_t *out_stats);

/* Configures system prompt, optional JSON tool list, and multi-token stop strings */
LFM2_API lfm2_status_t lfm2_context_set_system_prompt(lfm2_context_t *ctx,
                                                      const char *system_prompt,
                                                      const char *tools_json);
LFM2_API void          lfm2_context_clear_stops(lfm2_context_t *ctx);
LFM2_API lfm2_status_t lfm2_context_add_stop(lfm2_context_t *ctx, const char *stop_str);

/* Tokenizes raw text or formats + prefills a stateful ChatML user turn */
LFM2_API int32_t       lfm2_tokenize_text(const lfm2_model_t *model,
                                          const char *text,
                                          uint32_t *out_ids,
                                          uint32_t max_ids);

LFM2_API lfm2_status_t lfm2_prefill_tokens(lfm2_context_t *ctx,
                                           const uint32_t *tokens,
                                           uint32_t n_tokens,
                                           bool compute_last_logits);

/* Zero-copy raw logit vector accessor for the most recent prefill / decode step.
 * Returns pointer to ctx->state.logits (length n_vocab), or NULL if ctx is NULL. */
LFM2_API const float *lfm2_context_get_logits_ptr(const lfm2_context_t *ctx);

LFM2_API lfm2_status_t lfm2_begin_chat_turn(lfm2_context_t *ctx,
                                            const char *user_message,
                                            const lfm2_sample_params_t *sample_params);

/* Pull-based streaming generator:
 * Steps token generation until at least 1 complete UTF-8 byte is verified safe
 * past all partial stop-string prefixes, or returns LFM2_DONE when the turn ends. */
LFM2_API lfm2_status_t lfm2_context_next_utf8(lfm2_context_t *ctx,
                                              char *out_utf8_buf,
                                              size_t buf_cap,
                                              size_t *out_bytes_written,
                                              int32_t *out_think_phase);

/* ============================================================================
 * 6. L2-Normalized Embeddings & Multimodal Input Hook
 * ============================================================================ */
/* Computes an L2-normalized embedding vector `out_embd[n_embd]` for `text`
 * while skipping the 65,536-row LM classifier head (`output.weight`). */
LFM2_API lfm2_status_t lfm2_embed_text(lfm2_context_t *ctx,
                                       const char *text,
                                       int32_t pooling_type,
                                       float *out_embd);

/* Direct custom embedding hook for LFM2-VL (vision patches) and LFM2-Audio */
LFM2_API lfm2_status_t lfm2_forward_embeddings(lfm2_context_t *ctx,
                                               const float *embd_batch,
                                               uint32_t n_tokens,
                                               bool compute_last_logits);

/* ============================================================================
 * 7. Hybrid State Checkpoints (In-RAM Rollbacks & Disk Prompt Cache)
 * ============================================================================ */
LFM2_API lfm2_checkpoint_t *lfm2_checkpoint_create(const lfm2_model_t *model);
LFM2_API void               lfm2_checkpoint_free(lfm2_checkpoint_t *ckpt);

/* Saves/restores all ShortConv 2-token rolling buffers + GQA KV cursor in ~15 us */
LFM2_API lfm2_status_t lfm2_context_save_checkpoint(const lfm2_context_t *ctx,
                                                    lfm2_checkpoint_t *ckpt);
LFM2_API lfm2_status_t lfm2_context_restore_checkpoint(lfm2_context_t *ctx,
                                                       const lfm2_checkpoint_t *ckpt);

/* Disk-backed `.lfmcache` serializer & prefix loader */
LFM2_API lfm2_status_t lfm2_context_save_prompt_cache(const lfm2_context_t *ctx,
                                                      const char *cache_filepath);
LFM2_API lfm2_status_t lfm2_context_load_prompt_cache(lfm2_context_t *ctx,
                                                      const char *cache_filepath,
                                                      const uint32_t *prompt_tokens,
                                                      uint32_t n_prompt,
                                                      uint32_t *out_matched_len);

/* ============================================================================
 * 8. Zero-Dependency Linux epoll HTTP/SSE OpenAI Server
 * ============================================================================ */
LFM2_API int lfm2_run_http_server(lfm2_model_t *model,
                                  lfm2_context_t *ctx,
                                  const char *bind_ip,
                                  uint16_t port);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_H */

