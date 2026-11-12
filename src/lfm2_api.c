#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "lfm2.h"
#include "lfm_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_PREFILL_BATCH_NO_MAIN 1
#define LFM2_CLI_STREAM_NO_MAIN 1
#define LFM2_SYS_CACHE_NO_MAIN 1
#define LFM2_RUNTIME_FEATURES_NO_MAIN 1
#define LFM2_MOE_CHECKPOINT_NO_MAIN 1
#define LFM2_SERVER_NO_MAIN 1
#define LFM2_SPECULATIVE_PLD_NO_MAIN 1
#define LFM3_ENGINE_NO_MAIN 1

#include "../tools/lfm2_prefill_batch.c"
#include "../tools/lfm2_bpe.c"
#include "../tools/lfm2_runtime_features.c"
#include "../tools/lfm2_cli_stream.c"
#include "../tools/lfm2_moe_checkpoint.c"
#include "../tools/lfm2_sys_cache.c"
#include "../tools/lfm2_speculative_pld.c"
#include "../tools/lfm3_engine.c"

/* ============================================================================
 * 1. Sampler & Token Candidate Buffer (Matches LFM2 Official Spec)
 * ============================================================================ */
typedef struct {
    uint32_t id;
    float    val;
} TokenCand;

typedef struct {
    float     temperature;
    float     min_p;
    uint32_t  top_k;
    float     rep_penalty;
    uint32_t  rep_window;
    uint64_t  rng_state;

    uint32_t  n_vocab;
    uint32_t *history_ring;
    uint32_t  history_count;
    TokenCand *cands;
} LFM2Sampler;

static inline uint32_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static inline float random_f32_uniform(uint64_t *state) {
    return (xorshift64star(state) >> 8) / 16777216.0f;
}

static int compare_cands_desc(const void *a, const void *b) {
    const float va = ((const TokenCand *)a)->val;
    const float vb = ((const TokenCand *)b)->val;
    return (va < vb) - (va > vb);
}

static void lfm2_sampler_free(LFM2Sampler *s) {
    if (!s) return;
    free(s->history_ring);
    free(s->cands);
    s->history_ring = NULL;
    s->cands        = NULL;
    memset(s, 0, sizeof(*s));
}

static bool lfm2_sampler_init(LFM2Sampler *s, uint32_t n_vocab, float temp, float min_p,
                             uint32_t top_k, float rep_penalty, uint32_t rep_window, uint64_t seed) {
    if (!s) return false;
    lfm2_sampler_free(s);
    s->temperature = temp;
    s->min_p       = min_p;
    s->top_k       = (top_k == 0 || top_k > n_vocab) ? n_vocab : top_k;
    s->rep_penalty = rep_penalty;
    s->rep_window  = (rep_window > 0) ? rep_window : 64;
    s->rng_state   = seed ? seed : (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ULL;
    s->n_vocab     = n_vocab;

    s->history_ring = (uint32_t *)calloc(s->rep_window, sizeof(uint32_t));
    s->cands        = (TokenCand *)malloc((size_t)n_vocab * sizeof(TokenCand));
    return (s->history_ring != NULL && s->cands != NULL);
}

static void lfm2_sampler_reset(LFM2Sampler *s) {
    s->history_count = 0;
}

static void lfm2_sampler_accept(LFM2Sampler *s, uint32_t token_id) {
    if (s->rep_window == 0 || !s->history_ring) return;
    s->history_ring[s->history_count % s->rep_window] = token_id;
    s->history_count++;
}

static uint32_t lfm2_sample_token(LFM2Sampler *s, float *logits) {
    const uint32_t V = s->n_vocab;

    if (s->rep_penalty > 1.0001f && s->history_count > 0 && s->history_ring) {
        const uint32_t win_len = (s->history_count < s->rep_window)
                                 ? s->history_count : s->rep_window;
        for (uint32_t i = 0; i < win_len; ++i) {
            const uint32_t tid = s->history_ring[i];
            if (tid >= V) continue;

            bool already_penalized = false;
            for (uint32_t j = 0; j < i; ++j) {
                if (s->history_ring[j] == tid) {
                    already_penalized = true;
                    break;
                }
            }
            if (!already_penalized) {
                if (logits[tid] > 0.0f) {
                    logits[tid] /= s->rep_penalty;
                } else {
                    logits[tid] *= s->rep_penalty;
                }
            }
        }
    }

    uint32_t best_id  = 0;
    float    max_logit = logits[0];
    for (uint32_t i = 1; i < V; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best_id   = i;
        }
    }

    if (s->temperature <= 1e-5f) {
        lfm2_sampler_accept(s, best_id);
        return best_id;
    }

    float logit_cutoff = -1e30f;
    if (s->min_p > 0.0f && s->min_p < 1.0f) {
        logit_cutoff = max_logit + s->temperature * logf(s->min_p);
    }

    uint32_t n_surviving = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (logits[i] >= logit_cutoff) {
            s->cands[n_surviving].id  = i;
            s->cands[n_surviving].val = logits[i];
            n_surviving++;
        }
    }

    if (n_surviving <= 1) {
        lfm2_sampler_accept(s, best_id);
        return best_id;
    }

    qsort(s->cands, n_surviving, sizeof(TokenCand), compare_cands_desc);
    if (s->top_k > 0 && n_surviving > s->top_k) {
        n_surviving = s->top_k;
    }

    const float inv_temp = 1.0f / s->temperature;
    const float top_logit = s->cands[0].val;
    float sum_prob = 0.0f;

    for (uint32_t i = 0; i < n_surviving; ++i) {
        const float p = expf((s->cands[i].val - top_logit) * inv_temp);
        s->cands[i].val = p;
        sum_prob += p;
    }

    const float r = random_f32_uniform(&s->rng_state) * sum_prob;
    float cdf = 0.0f;
    uint32_t sampled_id = s->cands[n_surviving - 1].id;

    for (uint32_t i = 0; i < n_surviving; ++i) {
        cdf += s->cands[i].val;
        if (r <= cdf) {
            sampled_id = s->cands[i].id;
            break;
        }
    }

    lfm2_sampler_accept(s, sampled_id);
    return sampled_id;
}

/* ============================================================================
 * 2. Stateful ChatML Prompt Formatter
 * ============================================================================ */
typedef struct {
    char system_prompt[1024];
    char tool_json_list[2048];
    bool has_tools;
    bool is_lfm3;
} LFM2ChatFormatter;

static void lfm2_chat_formatter_init(LFM2ChatFormatter *fmt, const char *system_prompt,
                                    const char *optional_tools_json) {
    const bool keep_lfm3 = fmt->is_lfm3;
    memset(fmt, 0, sizeof(*fmt));
    fmt->is_lfm3 = keep_lfm3;
    snprintf(fmt->system_prompt, sizeof(fmt->system_prompt), "%s",
             (system_prompt && system_prompt[0])
                 ? system_prompt
                 : "You are a helpful assistant trained by Liquid AI.");
    if (optional_tools_json && optional_tools_json[0]) {
        snprintf(fmt->tool_json_list, sizeof(fmt->tool_json_list), "%s", optional_tools_json);
        fmt->has_tools = true;
    }
}

static int lfm2_format_chatml_turn(const LFM2ChatFormatter *fmt, const char *user_msg,
                                   bool is_first_turn, char *out_buf, size_t out_cap) {
    const char *think_prefix = fmt->is_lfm3 ? "<think>" : "";
    if (is_first_turn) {
        if (fmt->has_tools) {
            return snprintf(out_buf, out_cap,
                "<|startoftext|><|im_start|>system\n%s\n"
                "<|tool_list_start|>%s<|tool_list_end|><|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n%s",
                fmt->system_prompt, fmt->tool_json_list, user_msg, think_prefix);
        } else {
            return snprintf(out_buf, out_cap,
                "<|startoftext|><|im_start|>system\n%s<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n%s",
                fmt->system_prompt, user_msg, think_prefix);
        }
    } else {
        return snprintf(out_buf, out_cap,
            "\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n%s",
            user_msg, think_prefix);
    }
}

/* ============================================================================
 * 3. Internal Engine Handle Structs
 * ============================================================================ */
struct lfm2_model {
    LFM2Model core;
    LFM2Vocab vocab;
    LFM2DecodedVocab decoded_vocab;
    lfm2_model_info_t info;
    uint32_t close_think_ids[8];
    uint32_t n_close_think_ids;
    char filepath[512];
};

struct lfm2_checkpoint {
    LFM2Checkpoint ckpt;
    const lfm2_model_t *model;
};

typedef struct {
    uint32_t token_id;
    uint32_t batch_slot;    /* Slot b (0..B-1) in ss->conv_trace[b][l] */
    int32_t  think_phase;   /* <think> controller phase after this token */
} LFM2SpecFifoItem;

struct lfm2_context {
    const lfm2_model_t *model;
    lfm2_context_params_t params;
    LFM2SpecState       spec_state;     /* Holds LFM2BatchState base + conv_trace[8][L] + logits_batch */
    LFM3State          *lfm3;           /* Non-NULL ONLY for LFM3 models (isolated SWA + Conv7 + Sandwich arena) */
    LFM2RoPELUT         rope_lut;
    LFM2Sampler         sampler;
    LFM2Grammar         grammar;
    LFM2ThinkController think_ctrl;
    LFM2StreamFilter    stream_filter;
    LFM2NgramIndex      ngram_idx;
    LFM2ChatFormatter   chat_fmt;

    uint32_t            pos;            /* GQA KV-cache position of committed state */
    uint32_t            pending_token;  /* Last sampled token waiting to enter graph */
    bool                has_pending;
    uint32_t            n_sink_tokens;
    uint32_t           *history_tokens;
    size_t              history_cap;
    uint32_t            n_history;
    lfm2_perf_stats_t   stats;

    /* 8-Token Speculative Verification FIFO */
    LFM2SpecFifoItem    spec_fifo[LFM2_MAX_BATCH];
    uint32_t            fifo_head;
    uint32_t            fifo_tail;
};

/* Internal accumulator used by LFM2StreamFilter inside lfm2_context_next_utf8 */
typedef struct {
    char  *dst;
    size_t cap;
    size_t written;
} ApiEmitChunkBuf;

static void api_chunk_emit_cb(const char *utf8_bytes, size_t len, void *user_data) {
    ApiEmitChunkBuf *b = (ApiEmitChunkBuf *)user_data;
    if (b->written + len < b->cap) {
        memcpy(b->dst + b->written, utf8_bytes, len);
        b->written += len;
        b->dst[b->written] = '\0';
    }
}

/* ============================================================================
 * 4. Default Parameter Initializers
 * ============================================================================ */
LFM2_API lfm2_context_params_t lfm2_context_default_params(void) {
    return (lfm2_context_params_t){
        .max_seq_len    = LFM_DEFAULT_CTX_LEN,
        .n_threads      = 0,
        .n_sink_tokens  = 64,
        .auto_clamp_ram = 1,
        .warmup_mmap    = 1
    };
}

LFM2_API lfm2_sample_params_t lfm2_sample_default_params(void) {
    return (lfm2_sample_params_t){
        .temperature      = 0.30f,
        .min_p            = 0.15f,
        .top_k            = 50,
        .rep_penalty      = 1.05f,
        .rep_window       = 64,
        .seed             = 0,
        .grammar_type     = LFM2_GRAMMAR_NONE,
        .max_think_tokens = 0
    };
}

/* ============================================================================
 * 5. Model Lifecycle & Introspection API
 * ============================================================================ */
LFM2_API lfm2_model_t *lfm2_model_load_from_fd(int fd, uint64_t file_size, int32_t warmup_pages) {
    lfm2_set_denormals_zero_mode();
    if (fd < 0) return NULL;

    int dup_fd = dup(fd);
    if (dup_fd < 0) return NULL;

    lfm2_model_t *m = (lfm2_model_t *)calloc(1, sizeof(lfm2_model_t));
    if (!m) {
        close(dup_fd);
        return NULL;
    }

    if (!lfm2_load_gguf_from_fd(dup_fd, (size_t)file_size, &m->core)) {
        free(m);
        return NULL;
    }

    if (!lfm2_vocab_load(&m->vocab, &m->core)) {
        lfm2_free_model(&m->core);
        free(m);
        return NULL;
    }

    if (!lfm2_decoded_vocab_init(&m->decoded_vocab, m->vocab.tokens, m->vocab.n_vocab)) {
        lfm2_vocab_free(&m->vocab);
        lfm2_free_model(&m->core);
        free(m);
        return NULL;
    }

    int32_t n_close = lfm2_tokenize(&m->vocab, "\n</think>\n", m->close_think_ids, 8);
    m->n_close_think_ids = (n_close > 0) ? (uint32_t)n_close : 0;

    /* Populate model info */
    const LFM2Config *c = &m->core.cfg;
    snprintf(m->info.arch, sizeof(m->info.arch), "%s",
             c->is_lfm3 ? "lfm3" : ((c->n_experts > 0) ? "lfm2moe" : "lfm2"));
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    const char *isa_str = "SCALAR";
    if (kt && kt->isa_name) {
        if (strstr(kt->isa_name, "AVX-VNNI") != NULL) {
            isa_str = "AVX-VNNI + FMA + F16C";
        } else if (strstr(kt->isa_name, "AVX2") != NULL) {
            isa_str = "AVX2 + FMA + F16C";
        } else if (strstr(kt->isa_name, "DOTPROD") != NULL) {
            isa_str = (strstr(kt->isa_name, "I8MM") != NULL)
                      ? "ARM64 NEON + DotProd + I8MM"
                      : "ARM64 NEON + DotProd + FP16";
        } else if (kt->quantize_q8_K != lfm2_quantize_row_q8_K_scalar) {
            isa_str = "ARM64 NEON";
        }
    }
    snprintf(m->info.active_isa, sizeof(m->info.active_isa), "%s", isa_str);
    m->info.n_layers = c->n_layers;
    uint32_t n_sc = 0;
    uint32_t n_gqa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->core.layers[l].block_type == LFM2_BLOCK_SHORTCONV) n_sc++;
        else n_gqa++;
    }
    m->info.n_shortconv_layers = n_sc;
    m->info.n_gqa_layers       = n_gqa;
    m->info.n_embd             = c->n_embd;
    m->info.n_ff               = c->n_ff;
    m->info.n_vocab            = c->n_vocab;
    m->info.n_ctx_train        = c->n_ctx;
    m->info.n_experts          = c->n_experts;
    m->info.n_experts_used     = c->n_experts_used;
    m->info.file_size_bytes    = (uint64_t)m->core.file_size;

    snprintf(m->filepath, sizeof(m->filepath), "<fd:%d>", fd);

    if (warmup_pages && m->core.mmap_base && m->core.file_size > 0) {
        lfm2_warmup_mmap_pages(&m->core, NULL, false);
    }

    return m;
}

LFM2_API lfm2_model_t *lfm2_model_load(const char *gguf_path, int32_t warmup_pages) {
    if (!gguf_path) return NULL;
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        close(fd);
        return NULL;
    }
    lfm2_model_t *m = lfm2_model_load_from_fd(fd, (uint64_t)st.st_size, warmup_pages);
    close(fd);
    if (m) {
        snprintf(m->filepath, sizeof(m->filepath), "%s", gguf_path);
    }
    return m;
}

LFM2_API void lfm2_model_release_page_cache(lfm2_model_t *model) {
    if (!model || !model->core.mmap_base || model->core.file_size == 0) return;
    madvise((void *)model->core.mmap_base, model->core.file_size, MADV_DONTNEED);
#if defined(__linux__) || defined(__ANDROID__)
    if (model->core.fd >= 0) {
        posix_fadvise(model->core.fd, 0, (off_t)model->core.file_size, POSIX_FADV_DONTNEED);
    }
#endif
}

LFM2_API void lfm2_model_warmup_page_cache(lfm2_model_t *model) {
    if (!model || !model->core.mmap_base || model->core.file_size == 0) return;
    madvise((void *)model->core.mmap_base, model->core.file_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    long ps = sysconf(_SC_PAGESIZE);
    size_t stride = (ps > 0) ? (size_t)ps : 4096;
    volatile uint64_t acc = 0;
    const uint8_t *ptr = (const uint8_t *)model->core.mmap_base;
    for (size_t off = 0; off < model->core.file_size; off += stride) {
        acc += ptr[off];
    }
    (void)acc;
}

LFM2_API uint64_t lfm2_get_process_rss_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long vm_pages = 0, rss_pages = 0;
    if (fscanf(f, "%lu %lu", &vm_pages, &rss_pages) != 2) {
        rss_pages = 0;
    }
    fclose(f);
    long ps = sysconf(_SC_PAGESIZE);
    return (uint64_t)rss_pages * (uint64_t)((ps > 0) ? ps : 4096);
}

LFM2_API void lfm2_model_free(lfm2_model_t *model) {
    if (!model) return;
    lfm2_decoded_vocab_free(&model->decoded_vocab);
    lfm2_vocab_free(&model->vocab);
    lfm2_free_model(&model->core);
    free(model);
}

LFM2_API lfm2_status_t lfm2_model_get_info(const lfm2_model_t *model, lfm2_model_info_t *out_info) {
    if (!model || !out_info) return LFM2_ERR_INVALID_ARG;
    *out_info = model->info;
    return LFM2_OK;
}

/* ============================================================================
 * 6. Stateful Inference Context API
 * ============================================================================ */
LFM2_API lfm2_context_t *lfm2_context_create(const lfm2_model_t *model,
                                             const lfm2_context_params_t *params) {
    lfm2_set_denormals_zero_mode();
    if (!model) return NULL;

    lfm2_context_t *ctx = NULL;
    if (posix_memalign((void **)&ctx, 64, sizeof(lfm2_context_t)) != 0 || !ctx) {
        return NULL;
    }
    memset(ctx, 0, sizeof(lfm2_context_t));

    ctx->model  = model;
    ctx->params = params ? *params : lfm2_context_default_params();
    ctx->params.max_seq_len = lfm_resolve_ctx_len(ctx->params.max_seq_len, model->core.cfg.n_ctx);

    if (ctx->params.auto_clamp_ram) {
        LFM2MemoryPlan plan = lfm2_plan_memory_and_guard_oom(&model->core, ctx->params.max_seq_len, 512);
        if (plan.was_clamped) {
            ctx->params.max_seq_len = plan.safe_clamped_ctx;
        }
    }

    if (model->core.cfg.is_lfm3) {
        ctx->lfm3 = (LFM3State *)calloc(1, sizeof(LFM3State));
        if (!ctx->lfm3 || !lfm3_state_init(ctx->lfm3, &model->core, ctx->params.max_seq_len, ctx->params.n_threads)) {
            lfm3_state_free(ctx->lfm3);
            free(ctx->lfm3);
            free(ctx);
            return NULL;
        }
        ctx->params.max_seq_len = ctx->lfm3->max_seq_len;
    } else {
        if (!lfm2_spec_state_init(&ctx->spec_state, &model->core, ctx->params.max_seq_len, ctx->params.n_threads)) {
            free(ctx);
            return NULL;
        }
        ctx->params.max_seq_len = ctx->spec_state.base.max_seq_len;
    }

    if (!lfm2_ngram_index_init(&ctx->ngram_idx, ctx->params.max_seq_len, 6, model->vocab.id_im_end)) {
        if (ctx->lfm3) { lfm3_state_free(ctx->lfm3); free(ctx->lfm3); }
        else { lfm2_spec_state_free(&ctx->spec_state); }
        free(ctx);
        return NULL;
    }

    if (!lfm2_sampler_init(&ctx->sampler, model->core.cfg.n_vocab, 0.30f, 0.15f, 50, 1.05f, 64, 0)) {
        lfm2_ngram_index_free(&ctx->ngram_idx);
        if (ctx->lfm3) { lfm3_state_free(ctx->lfm3); free(ctx->lfm3); }
        else { lfm2_spec_state_free(&ctx->spec_state); }
        free(ctx);
        return NULL;
    }

    lfm2_grammar_init(&ctx->grammar, LFM2_GRAMMAR_NONE);
    lfm2_think_controller_init(&ctx->think_ctrl, 0, NULL, 0);
    lfm2_stream_filter_init(&ctx->stream_filter, NULL, NULL);

    if (!ctx->lfm3) {
        const float freq_base = (model->core.cfg.rope_freq_base > 0.0f) ? model->core.cfg.rope_freq_base : 10000.0f;
        lfm2_rope_lut_init(&ctx->rope_lut, ctx->params.max_seq_len, 64, freq_base);
    }
    ctx->chat_fmt.is_lfm3 = model->core.cfg.is_lfm3;
    lfm2_chat_formatter_init(&ctx->chat_fmt, NULL, NULL);

    ctx->n_sink_tokens = ctx->params.n_sink_tokens ? ctx->params.n_sink_tokens : 64;
    ctx->history_cap   = ctx->params.max_seq_len;
    ctx->history_tokens = (uint32_t *)calloc(ctx->history_cap, sizeof(uint32_t));
    if (!ctx->history_tokens) {
        lfm2_rope_lut_free(&ctx->rope_lut);
        lfm2_sampler_free(&ctx->sampler);
        lfm2_ngram_index_free(&ctx->ngram_idx);
        if (ctx->lfm3) { lfm3_state_free(ctx->lfm3); free(ctx->lfm3); }
        else { lfm2_spec_state_free(&ctx->spec_state); }
        free(ctx);
        return NULL;
    }

    ctx->pos = 0;
    ctx->n_history = 0;
    ctx->fifo_head = 0;
    ctx->fifo_tail = 0;
    ctx->has_pending = false;
    ctx->pending_token = 0;
    return ctx;
}

LFM2_API void lfm2_context_free(lfm2_context_t *ctx) {
    if (!ctx) return;
    free(ctx->history_tokens);
    lfm2_rope_lut_free(&ctx->rope_lut);
    lfm2_sampler_free(&ctx->sampler);
    lfm2_ngram_index_free(&ctx->ngram_idx);
    if (ctx->lfm3) {
        lfm3_state_free(ctx->lfm3);
        free(ctx->lfm3);
        ctx->lfm3 = NULL;
    } else {
        lfm2_spec_state_free(&ctx->spec_state);
    }
    free(ctx);
}

LFM2_API void lfm2_context_reset(lfm2_context_t *ctx) {
    if (!ctx) return;
    ctx->pos = 0;
    ctx->n_history = 0;
    ctx->fifo_head = 0;
    ctx->fifo_tail = 0;
    ctx->has_pending = false;
    ctx->pending_token = 0;
    lfm2_ngram_index_reset(&ctx->ngram_idx);
    lfm2_sampler_reset(&ctx->sampler);
    lfm2_stream_filter_reset(&ctx->stream_filter);
    lfm2_grammar_init(&ctx->grammar, LFM2_GRAMMAR_NONE);
    lfm2_think_controller_init(&ctx->think_ctrl, 0, NULL, 0);

    const LFM2Model *m = &ctx->model->core;
    if (ctx->lfm3) {
        lfm3_state_reset(ctx->lfm3, m);
    } else {
        const uint32_t hist_len = (m->cfg.shortconv_l_cache > 1) ? (m->cfg.shortconv_l_cache - 1) : 2;
        for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
            if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV && ctx->spec_state.base.conv_state[l]) {
                memset(ctx->spec_state.base.conv_state[l], 0, hist_len * m->cfg.n_embd * sizeof(float));
            }
        }
    }
    memset(&ctx->stats, 0, sizeof(ctx->stats));
}

LFM2_API uint32_t lfm2_context_get_pos(const lfm2_context_t *ctx) {
    return ctx ? ctx->pos : 0;
}

LFM2_API void lfm2_context_get_stats(const lfm2_context_t *ctx,
                                     lfm2_perf_stats_t *out_stats) {
    if (!ctx || !out_stats) return;
    *out_stats = ctx->stats;
    out_stats->current_pos = ctx->pos;
}

LFM2_API lfm2_status_t lfm2_context_set_system_prompt(lfm2_context_t *ctx,
                                                      const char *system_prompt,
                                                      const char *tools_json) {
    if (!ctx) return LFM2_ERR_INVALID_ARG;
    lfm2_chat_formatter_init(&ctx->chat_fmt, system_prompt, tools_json);
    return LFM2_OK;
}

LFM2_API void lfm2_context_clear_stops(lfm2_context_t *ctx) {
    if (!ctx) return;
    ctx->stream_filter.n_stops = 0;
    lfm2_stream_filter_reset(&ctx->stream_filter);
}

LFM2_API lfm2_status_t lfm2_context_add_stop(lfm2_context_t *ctx, const char *stop_str) {
    if (!ctx || !stop_str) return LFM2_ERR_INVALID_ARG;
    return lfm2_stream_filter_add_stop(&ctx->stream_filter, stop_str) ? LFM2_OK : LFM2_ERR_INVALID_ARG;
}

LFM2_API int32_t lfm2_tokenize_text(const lfm2_model_t *model,
                                    const char *text,
                                    uint32_t *out_ids,
                                    uint32_t max_ids) {
    if (!model || !text || !out_ids || max_ids == 0) return -1;
    return (int32_t)lfm2_tokenize(&model->vocab, text, out_ids, max_ids);
}

static inline uint32_t api_ctx_max_seq_len(const lfm2_context_t *ctx) {
    return ctx->lfm3 ? ctx->lfm3->max_seq_len : ctx->spec_state.base.max_seq_len;
}

LFM2_API lfm2_status_t lfm2_prefill_tokens(lfm2_context_t *ctx,
                                           const uint32_t *tokens,
                                           uint32_t n_tokens,
                                           bool compute_last_logits) {
    if (!ctx || !tokens || n_tokens == 0) return LFM2_ERR_INVALID_ARG;
    if (ctx->pos + n_tokens > api_ctx_max_seq_len(ctx)) return LFM2_ERR_CTX_FULL;

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    const LFM2Model *m = &ctx->model->core;
    uint32_t i = 0;
    while (i < n_tokens) {
        uint32_t chunk = n_tokens - i;
        if (chunk > LFM2_MAX_BATCH) chunk = LFM2_MAX_BATCH;
        bool is_last = (i + chunk == n_tokens) && compute_last_logits;

        if (ctx->lfm3) {
            lfm3_forward_batch(m, ctx->lfm3, tokens + i, chunk, ctx->pos, is_last);
        } else {
            lfm2_forward_batch(m, &ctx->spec_state.base, tokens + i, chunk, ctx->pos, is_last);
        }

        for (uint32_t t = 0; t < chunk; ++t) {
            lfm2_sampler_accept(&ctx->sampler, tokens[i + t]);
            lfm2_ngram_index_ingest(&ctx->ngram_idx, tokens[i + t]);
            if (ctx->n_history < ctx->history_cap) {
                ctx->history_tokens[ctx->n_history++] = tokens[i + t];
            }
        }
        ctx->pos += chunk;
        i += chunk;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double dt_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    ctx->stats.prefill_ms += dt_ms;
    ctx->stats.prompt_tokens += n_tokens;
    if (ctx->stats.prefill_ms > 0.0) {
        ctx->stats.prefill_tok_per_sec = (ctx->stats.prompt_tokens * 1000.0) / ctx->stats.prefill_ms;
    }

    return LFM2_OK;
}

LFM2_API const float *lfm2_context_get_logits_ptr(const lfm2_context_t *ctx) {
    if (!ctx) return NULL;
    return ctx->lfm3 ? ctx->lfm3->logits : ctx->spec_state.base.logits;
}

/* ============================================================================
 * Causal Verification Sampler Callback (Grammar PDA + <think> Budget)
 * ============================================================================
 * Invoked for each slot b = 0 .. K inside `lfm2_speculative_decode_step()`.
 * Because `lfm2_speculative_decode_step` halts immediately on the first draft
 * mismatch, every token returned by this callback is guaranteed to be emitted
 * in causal order, so committing to `think_ctrl` and `grammar` here keeps
 * slot `b + 1`'s PDA state 100% synchronized with slot `b`! */
static uint32_t api_spec_verify_sample_cb(float *slot_logits,
                                          uint32_t n_vocab,
                                          void *user_data) {
    lfm2_context_t *ctx = (lfm2_context_t *)user_data;

    /* 1. Apply <think> token budget mask & Grammar PDA logit mask for slot `b` */
    lfm2_think_apply_budget_mask(&ctx->think_ctrl, slot_logits, n_vocab);
    lfm2_grammar_mask_logits(&ctx->grammar, &ctx->model->decoded_vocab, slot_logits);

    /* 2. Sample token from masked distribution */
    const uint32_t sampled_id = lfm2_sample_token(&ctx->sampler, slot_logits);

    /* 3. Advance <think> controller & Grammar PDA state for the next slot `b + 1` */
    lfm2_think_commit_token(&ctx->think_ctrl, &ctx->model->decoded_vocab, sampled_id);
    lfm2_grammar_commit_token(&ctx->grammar, &ctx->model->decoded_vocab, sampled_id);

    return sampled_id;
}

static uint32_t lfm2_format_and_tokenize_turn(lfm2_context_t *ctx,
                                              const char *user_message,
                                              uint32_t *out_ids,
                                              size_t max_ids) {
    char turn_buf[16384];
    bool is_first = (ctx->pos == 0);
    int written = lfm2_format_chatml_turn(&ctx->chat_fmt, user_message, is_first,
                                          turn_buf, sizeof(turn_buf));
    if (written <= 0 || (size_t)written >= sizeof(turn_buf)) {
        return 0;
    }
    int32_t n = lfm2_tokenize_text(ctx->model, turn_buf, out_ids, (uint32_t)max_ids);
    if (n <= 0) {
        n = lfm2_tokenize_text(ctx->model, user_message, out_ids, (uint32_t)max_ids);
    }
    if (n <= 0) {
        out_ids[0] = (ctx->model->vocab.id_bos != 0) ? ctx->model->vocab.id_bos : 1;
        n = 1;
    }
    return (uint32_t)n;
}

LFM2_API lfm2_status_t lfm2_begin_chat_turn(lfm2_context_t *ctx,
                                            const char *user_message,
                                            const lfm2_sample_params_t *sp) {
    if (!ctx || !user_message || !sp) return LFM2_ERR_INVALID_ARG;

    /* Clear speculative FIFO & stream holdback filter for the new turn */
    ctx->fifo_head = 0;
    ctx->fifo_tail = 0;
    lfm2_stream_filter_reset(&ctx->stream_filter);

    /* Configure sampler, grammar PDA, and <think> budget */
    lfm2_sampler_init(&ctx->sampler, ctx->model->core.cfg.n_vocab,
                      sp->temperature, sp->min_p, sp->top_k,
                      sp->rep_penalty, sp->rep_window, sp->seed);
    lfm2_grammar_init(&ctx->grammar, (lfm2_grammar_mode_t)sp->grammar_type);
    lfm2_think_controller_init(&ctx->think_ctrl, sp->max_think_tokens,
                               (ctx->model->n_close_think_ids > 0) ? ctx->model->close_think_ids : NULL,
                               ctx->model->n_close_think_ids);
    if (ctx->chat_fmt.is_lfm3) {
        ctx->think_ctrl.phase = LFM2_THINK_ACTIVE;
    }

    /* Format ChatML turn & tokenize into `turn_ids[]` */
    uint32_t turn_ids[8192];
    uint32_t n_turn = lfm2_format_and_tokenize_turn(ctx, user_message, turn_ids, 8192);
    if (n_turn == 0) return LFM2_ERR_INVALID_ARG;

    const double t0 = now_ms();
    lfm2_status_t st = lfm2_prefill_tokens(ctx, turn_ids, n_turn, true);
    if (st != LFM2_OK) return st;

    ctx->stats.prefill_ms    = now_ms() - t0;
    ctx->stats.prompt_tokens = n_turn;
    ctx->stats.generated_tokens = 0;
    ctx->has_pending         = false;
    return LFM2_OK;
}

/* Pull-based streaming step with 8-Token Speculative Verification FIFO:
 * Loops internally if a token is held back by an incomplete 4-byte UTF-8 sequence
 * or a partial stop-string prefix, guaranteeing every LFM2_OK return delivers
 * complete, valid UTF-8 bytes to Python or C callers! */
LFM2_API lfm2_status_t lfm2_context_next_utf8(lfm2_context_t *ctx,
                                              char *out_utf8_buf,
                                              size_t buf_cap,
                                              size_t *out_bytes_written,
                                              int32_t *out_think_phase) {
    if (!ctx || !out_utf8_buf || buf_cap < 8) return LFM2_ERR_INVALID_ARG;
    *out_bytes_written = 0;
    out_utf8_buf[0]    = '\0';

    ApiEmitChunkBuf chunk = { .dst = out_utf8_buf, .cap = buf_cap, .written = 0 };
    ctx->stream_filter.emit_cb   = api_chunk_emit_cb;
    ctx->stream_filter.user_data = &chunk;

    while (chunk.written == 0) {
        /* 1. If the 8-token speculative FIFO is empty, run one B <= 8 speculative batch */
        if (ctx->fifo_head >= ctx->fifo_tail) {
            float *cur_logits = ctx->lfm3 ? ctx->lfm3->logits : ctx->spec_state.base.logits;
            const uint32_t max_seq = api_ctx_max_seq_len(ctx);
            if (!ctx->has_pending) {
                if (ctx->pos > 0 && cur_logits) {
                    lfm2_think_apply_budget_mask(&ctx->think_ctrl, cur_logits, ctx->model->core.cfg.n_vocab);
                    lfm2_grammar_mask_logits(&ctx->grammar, &ctx->model->decoded_vocab, cur_logits);
                    const uint32_t sampled_first = lfm2_sample_token(&ctx->sampler, cur_logits);
                    lfm2_think_commit_token(&ctx->think_ctrl, &ctx->model->decoded_vocab, sampled_first);
                    lfm2_grammar_commit_token(&ctx->grammar, &ctx->model->decoded_vocab, sampled_first);

                    ctx->pending_token = sampled_first;
                    ctx->has_pending   = true;

                    /* Put first generated token into the FIFO */
                    ctx->spec_fifo[0].token_id    = sampled_first;
                    ctx->spec_fifo[0].batch_slot  = 0;
                    ctx->spec_fifo[0].think_phase = (int32_t)ctx->think_ctrl.phase;
                    ctx->fifo_head = 0;
                    ctx->fifo_tail = 1;
                } else {
                    lfm2_stream_filter_flush(&ctx->stream_filter);
                    *out_bytes_written = chunk.written;
                    ctx->stream_filter.user_data = NULL;
                    return LFM2_DONE;
                }
            } else {
                /* Auto-shift GQA KV-cache with Attention Sinks if within 8 tokens of max_seq_len */
                if (ctx->pos + LFM2_MAX_BATCH >= max_seq) {
                    if (ctx->lfm3) {
                        lfm3_kv_cache_shift_sinks(
                            ctx->lfm3->k_cache_f16, ctx->lfm3->v_cache_f16,
                            &ctx->model->core, &ctx->lfm3->rope_lut_full,
                            &ctx->lfm3->rope_lut_swa, ctx->lfm3->swa_cap,
                            &ctx->pos, ctx->n_sink_tokens, max_seq / 2U);
                    } else {
                        const uint32_t head_dim = ctx->model->core.cfg.n_head_q
                                                  ? (ctx->model->core.cfg.n_embd / ctx->model->core.cfg.n_head_q)
                                                  : 64;
                        if (head_dim == 64) {
                            lfm2_kv_cache_shift_sinks(
                                ctx->spec_state.base.k_cache_f16,
                                ctx->spec_state.base.v_cache_f16,
                                &ctx->model->core, &ctx->rope_lut, &ctx->pos,
                                ctx->n_sink_tokens, max_seq / 2);
                        }
                    }
                }
                if (ctx->pos + LFM2_MAX_BATCH >= max_seq) {
                    lfm2_stream_filter_flush(&ctx->stream_filter);
                    *out_bytes_written = chunk.written;
                    ctx->stream_filter.user_data = NULL;
                    return LFM2_DONE;
                }

                uint32_t emitted_ids[LFM2_MAX_BATCH];
                const double t_step0 = now_ms();

                /* Disable N-gram drafting while forcing `\n</think>\n` budget closure */
                const uint32_t saved_k = ctx->ngram_idx.cur_draft_k;
                if (ctx->think_ctrl.phase == LFM2_THINK_FORCING_CLOSE) {
                    ctx->ngram_idx.cur_draft_k = 0;
                }

                const uint32_t n_emitted = ctx->lfm3
                    ? lfm3_speculative_decode_step(
                        &ctx->model->core,
                        ctx->lfm3,
                        &ctx->ngram_idx,
                        &ctx->pending_token,
                        &ctx->pos,
                        api_spec_verify_sample_cb,
                        ctx,
                        emitted_ids)
                    : lfm2_speculative_decode_step(
                        &ctx->model->core,
                        &ctx->spec_state,
                        &ctx->ngram_idx,
                        &ctx->pending_token,
                        &ctx->pos,
                        api_spec_verify_sample_cb,
                        ctx,
                        emitted_ids);

                if (ctx->think_ctrl.phase == LFM2_THINK_FORCING_CLOSE) {
                    ctx->ngram_idx.cur_draft_k = saved_k;
                }

                ctx->stats.decode_ms += (now_ms() - t_step0);
                if (n_emitted == 0) {
                    ctx->stream_filter.user_data = NULL;
                    return LFM2_ERR_CTX_FULL;
                }

                /* Populate the 8-token FIFO with all verified tokens from this sweep */
                ctx->fifo_head = 0;
                ctx->fifo_tail = n_emitted;
                for (uint32_t b = 0; b < n_emitted; ++b) {
                    ctx->spec_fifo[b].token_id    = emitted_ids[b];
                    ctx->spec_fifo[b].batch_slot  = b;
                    ctx->spec_fifo[b].think_phase = (int32_t)ctx->think_ctrl.phase;
                }
            }
        }

        /* 2. Pop the next verified token from the FIFO in O(1) (0.001 ms, zero compute!) */
        const LFM2SpecFifoItem item = ctx->spec_fifo[ctx->fifo_head++];
        ctx->stats.generated_tokens++;
        if (ctx->n_history < ctx->history_cap) {
            ctx->history_tokens[ctx->n_history++] = item.token_id;
        }
        if (ctx->stats.decode_ms > 0.0) {
            ctx->stats.decode_tok_per_sec =
                (ctx->stats.generated_tokens * 1000.0) / ctx->stats.decode_ms;
        }
        if (out_think_phase) *out_think_phase = item.think_phase;

        /* 3. Check for ChatML <|im_end|> termination */
        if (item.token_id == ctx->model->vocab.id_im_end) {
            ctx->fifo_head   = ctx->fifo_tail;
            ctx->has_pending = false;
            lfm2_stream_filter_flush(&ctx->stream_filter);
            *out_bytes_written = chunk.written;
            ctx->stream_filter.user_data = NULL;
            return LFM2_DONE;
        }

        /* 4. Feed token bytes into the UTF-8 boundary + Multi-Token Stop-String filter */
        const char  *tok_str = ctx->model->decoded_vocab.bytes[item.token_id];
        const size_t tok_len = ctx->model->decoded_vocab.lens[item.token_id];

        lfm2_stream_status_t st = lfm2_stream_filter_feed(&ctx->stream_filter, tok_str, tok_len);
        if (st == LFM2_STREAM_STOPPED) {
            /* Mid-batch stop string matched!
             * If unpopped draft tokens remain in `spec_fifo`, rewind `ShortConv`
             * state to `item.batch_slot` and rewind `ctx->pos` to match! */
            const uint32_t unpopped = ctx->fifo_tail - ctx->fifo_head;
            if (unpopped > 0) {
                if (ctx->lfm3) {
                    lfm3_restore_shortconv_trace_slot(ctx->lfm3, &ctx->model->core, item.batch_slot);
                } else {
                    lfm2_restore_shortconv_trace_slot(&ctx->spec_state, &ctx->model->core, item.batch_slot);
                }
                ctx->pos -= unpopped;
            }
            ctx->fifo_head   = ctx->fifo_tail;
            ctx->has_pending = false;

            /* Append <|im_end|> to close the assistant turn cleanly in state */
            const uint32_t im_end = ctx->model->vocab.id_im_end;
            if (ctx->lfm3) {
                lfm3_forward_batch(&ctx->model->core, ctx->lfm3, &im_end, 1, ctx->pos++, false);
            } else {
                lfm2_forward_batch(&ctx->model->core, &ctx->spec_state.base,
                                   &im_end, 1, ctx->pos++, false);
            }
            *out_bytes_written = chunk.written;
            ctx->stream_filter.user_data = NULL;
            return LFM2_DONE;
        }
    }

    *out_bytes_written = chunk.written;
    ctx->stream_filter.user_data = NULL;
    return LFM2_OK;
}

/* ============================================================================
 * 7. Embeddings & Custom Forwarding API
 * ============================================================================ */
LFM2_API lfm2_status_t lfm2_embed_text(lfm2_context_t *ctx,
                                       const char *text,
                                       int32_t pooling_type,
                                       float *out_embd) {
    if (!ctx || !text || !out_embd) return LFM2_ERR_INVALID_ARG;
    const LFM2Model *m = &ctx->model->core;
    const uint32_t   D = m->cfg.n_embd;

    lfm2_context_reset(ctx);

    uint32_t token_ids[4096];
    int32_t n_tok = lfm2_tokenize_text(ctx->model, text, token_ids, 4096);
    if (n_tok <= 0) {
        token_ids[0] = (ctx->model->vocab.id_bos != 0) ? ctx->model->vocab.id_bos : 1;
        n_tok = 1;
    }

    float *x_buf  = ctx->lfm3 ? ctx->lfm3->x  : ctx->spec_state.base.x;
    float *xb_buf = ctx->lfm3 ? ctx->lfm3->xb : ctx->spec_state.base.xb;

    if (pooling_type == LFM2_POOL_MEAN) {
        memset(out_embd, 0, D * sizeof(float));
        uint32_t i = 0;
        while (i < (uint32_t)n_tok) {
            uint32_t B = ((uint32_t)n_tok - i > LFM2_MAX_BATCH) ? LFM2_MAX_BATCH : ((uint32_t)n_tok - i);
            if (ctx->lfm3) lfm3_forward_batch(m, ctx->lfm3, token_ids + i, B, ctx->pos, false);
            else lfm2_forward_batch(m, &ctx->spec_state.base, token_ids + i, B, ctx->pos, false);
            ctx->pos += B;

            /* Normalize each token state in the batch and accumulate into out_embd */
            lfm2_rmsnorm_batch(xb_buf, x_buf, m->output_norm, B, D, m->cfg.rms_norm_eps);
            for (uint32_t b = 0; b < B; ++b) {
                const float *vec = xb_buf + (size_t)b * D;
                for (uint32_t d = 0; d < D; ++d) out_embd[d] += vec[d];
            }
            i += B;
        }
    } else {
        /* LFM2_POOL_LAST_TOKEN: Prefill all chunks with compute_last_logits = false */
        uint32_t i = 0;
        uint32_t last_B = 1;
        while (i < (uint32_t)n_tok) {
            last_B = ((uint32_t)n_tok - i > LFM2_MAX_BATCH) ? LFM2_MAX_BATCH : ((uint32_t)n_tok - i);
            if (ctx->lfm3) lfm3_forward_batch(m, ctx->lfm3, token_ids + i, last_B, ctx->pos, false);
            else lfm2_forward_batch(m, &ctx->spec_state.base, token_ids + i, last_B, ctx->pos, false);
            ctx->pos += last_B;
            i += last_B;
        }
        const float *last_x = x_buf + (size_t)(last_B - 1) * D;
        lfm2_rmsnorm(out_embd, last_x, m->output_norm, D, m->cfg.rms_norm_eps);
    }

    /* L2-Normalize `out_embd[0 .. D-1]` so cosine similarity is a simple dot product */
    double ss = 0.0;
    for (uint32_t d = 0; d < D; ++d) ss += (double)out_embd[d] * (double)out_embd[d];
    const float inv_norm = (ss > 0.0) ? (1.0f / sqrtf((float)ss)) : 0.0f;
    for (uint32_t d = 0; d < D; ++d) out_embd[d] *= inv_norm;

    return LFM2_OK;
}

static void lfm2_forward_embd_chunk(const LFM2Model *m,
                                    LFM2BatchState *s,
                                    const float *embd,
                                    uint32_t B,
                                    uint32_t start_pos,
                                    bool compute_last_logits) {
    const uint32_t D = m->cfg.n_embd;
    memcpy(s->x, embd, (size_t)B * D * sizeof(float));
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];
        lfm2_rmsnorm_batch(s->xb, s->x, layer->attn_norm, B, D, m->cfg.rms_norm_eps);
        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv_batch(s, m, layer, l, B);
        } else {
            lfm2_forward_gqa_batch(s, m, layer, l, start_pos, B);
        }
        lfm2_forward_ffn_batch(s, m, layer, B);
    }
    if (compute_last_logits) {
        float *last_x = s->x + (size_t)(B - 1) * D;
        lfm2_rmsnorm(last_x, last_x, m->output_norm, D, m->cfg.rms_norm_eps);
        lfm2_matmul_vec(s->logits, m->output, last_x, (block_q8_K_t *)s->xq, &s->pool);
    }
}

LFM2_API lfm2_status_t lfm2_forward_embeddings(lfm2_context_t *ctx,
                                               const float *embd_batch,
                                               uint32_t n_tokens,
                                               bool compute_last_logits) {
    if (!ctx || !embd_batch || n_tokens == 0) return LFM2_ERR_INVALID_ARG;
    if (ctx->pos + n_tokens > api_ctx_max_seq_len(ctx)) return LFM2_ERR_CTX_FULL;

    const uint32_t D = ctx->model->core.cfg.n_embd;
    uint32_t i = 0;
    while (i < n_tokens) {
        uint32_t chunk = n_tokens - i;
        if (chunk > LFM2_MAX_BATCH) chunk = LFM2_MAX_BATCH;
        bool is_last = (i + chunk == n_tokens) && compute_last_logits;
        if (ctx->lfm3) {
            lfm3_forward_embd_batch(&ctx->model->core, ctx->lfm3,
                                    embd_batch + (size_t)i * D,
                                    chunk, ctx->pos, is_last);
        } else {
            lfm2_forward_embd_chunk(&ctx->model->core, &ctx->spec_state.base,
                                    embd_batch + (size_t)i * D,
                                    chunk, ctx->pos, is_last);
        }
        ctx->pos += chunk;
        i += chunk;
    }
    return LFM2_OK;
}

/* ============================================================================
 * 8. Hybrid State Checkpoints & Disk Prompt Cache API
 * ============================================================================ */
LFM2_API lfm2_checkpoint_t *lfm2_checkpoint_create(const lfm2_model_t *model) {
    if (!model) return NULL;
    lfm2_checkpoint_t *ckpt = (lfm2_checkpoint_t *)calloc(1, sizeof(lfm2_checkpoint_t));
    if (!ckpt) return NULL;
    ckpt->model = model;
    if (!lfm2_checkpoint_init(&ckpt->ckpt, &model->core, 64)) {
        free(ckpt);
        return NULL;
    }
    return ckpt;
}

LFM2_API void lfm2_checkpoint_free(lfm2_checkpoint_t *ckpt) {
    if (!ckpt) return;
    lfm2_checkpoint_destroy(&ckpt->ckpt);
    free(ckpt);
}

LFM2_API lfm2_status_t lfm2_context_save_checkpoint(const lfm2_context_t *ctx,
                                                    lfm2_checkpoint_t *ckpt) {
    if (!ctx || !ckpt) return LFM2_ERR_INVALID_ARG;
    if (ctx->lfm3) {
        lfm2_checkpoint_save_ex(&ckpt->ckpt, &ctx->model->core,
                                (float * const *)ctx->lfm3->conv_ring,
                                ctx->lfm3->conv_ring_head,
                                (uint16_t * const *)ctx->lfm3->k_cache_f16,
                                (uint16_t * const *)ctx->lfm3->v_cache_f16,
                                ctx->pos,
                                ctx->sampler.rng_state,
                                ctx->sampler.history_ring,
                                ctx->sampler.history_count);
    } else {
        lfm2_checkpoint_save(&ckpt->ckpt, &ctx->model->core,
                             (float * const *)ctx->spec_state.base.conv_state,
                             ctx->pos,
                             ctx->sampler.rng_state,
                             ctx->sampler.history_ring,
                             ctx->sampler.history_count);
    }
    return LFM2_OK;
}

LFM2_API lfm2_status_t lfm2_context_restore_checkpoint(lfm2_context_t *ctx,
                                                       const lfm2_checkpoint_t *ckpt) {
    if (!ctx || !ckpt) return LFM2_ERR_INVALID_ARG;
    bool ok = ctx->lfm3
        ? lfm2_checkpoint_restore_ex(&ckpt->ckpt, &ctx->model->core,
                                     ctx->lfm3->conv_ring,
                                     ctx->lfm3->conv_ring_head,
                                     ctx->lfm3->k_cache_f16,
                                     ctx->lfm3->v_cache_f16,
                                     &ctx->pos,
                                     &ctx->sampler.rng_state,
                                     ctx->sampler.history_ring,
                                     &ctx->sampler.history_count)
        : lfm2_checkpoint_restore(&ckpt->ckpt, &ctx->model->core,
                                  ctx->spec_state.base.conv_state,
                                  &ctx->pos,
                                  &ctx->sampler.rng_state,
                                  ctx->sampler.history_ring,
                                  &ctx->sampler.history_count);
    if (!ok) {
        return LFM2_ERR_CHECKPOINT;
    }
    ctx->n_history = ctx->pos;
    ctx->fifo_head = 0;
    ctx->fifo_tail = 0;
    ctx->has_pending = false;
    lfm2_stream_filter_reset(&ctx->stream_filter);
    return LFM2_OK;
}

LFM2_API lfm2_status_t lfm2_context_save_prompt_cache(const lfm2_context_t *ctx,
                                                      const char *cache_filepath) {
    if (!ctx || !cache_filepath) return LFM2_ERR_INVALID_ARG;
    if (ctx->pos == 0) return LFM2_ERR_INVALID_ARG;

    bool ok = ctx->lfm3
        ? lfm2_prompt_cache_save_ex(cache_filepath,
                                    &ctx->model->core,
                                    ctx->history_tokens,
                                    ctx->pos,
                                    (float * const *)ctx->lfm3->conv_ring,
                                    ctx->lfm3->conv_ring_head,
                                    (uint16_t * const *)ctx->lfm3->k_cache_f16,
                                    (uint16_t * const *)ctx->lfm3->v_cache_f16,
                                    ctx->lfm3->swa_cap)
        : lfm2_prompt_cache_save(cache_filepath,
                                 &ctx->model->core,
                                 ctx->history_tokens,
                                 ctx->pos,
                                 (float * const *)ctx->spec_state.base.conv_state,
                                 ctx->spec_state.base.k_cache_f16,
                                 ctx->spec_state.base.v_cache_f16);
    return ok ? LFM2_OK : LFM2_ERR_IO;
}

LFM2_API lfm2_status_t lfm2_context_load_prompt_cache(lfm2_context_t *ctx,
                                                      const char *cache_filepath,
                                                      const uint32_t *prompt_tokens,
                                                      uint32_t n_prompt,
                                                      uint32_t *out_matched_len) {
    if (!ctx || !cache_filepath || !prompt_tokens) return LFM2_ERR_INVALID_ARG;

    uint32_t matched = 0;
    bool ok = ctx->lfm3
        ? lfm2_prompt_cache_try_load_prefix_ex(cache_filepath,
                                               &ctx->model->core,
                                               prompt_tokens,
                                               n_prompt,
                                               ctx->lfm3->max_seq_len,
                                               ctx->lfm3->conv_ring,
                                               ctx->lfm3->conv_ring_head,
                                               ctx->lfm3->k_cache_f16,
                                               ctx->lfm3->v_cache_f16,
                                               ctx->lfm3->swa_cap,
                                               &matched)
        : lfm2_prompt_cache_try_load_prefix(cache_filepath,
                                            &ctx->model->core,
                                            prompt_tokens,
                                            n_prompt,
                                            ctx->spec_state.base.max_seq_len,
                                            ctx->spec_state.base.conv_state,
                                            ctx->spec_state.base.k_cache_f16,
                                            ctx->spec_state.base.v_cache_f16,
                                            &matched);
    if (!ok) {
        if (out_matched_len) *out_matched_len = 0;
        return LFM2_ERR_IO;
    }

    ctx->pos = matched;
    ctx->n_history = matched;
    for (uint32_t i = 0; i < matched; ++i) {
        ctx->history_tokens[i] = prompt_tokens[i];
        lfm2_sampler_accept(&ctx->sampler, prompt_tokens[i]);
    }
    if (out_matched_len) *out_matched_len = matched;
    return LFM2_OK;
}

#ifndef LFM2_API_NO_SERVER
#include "../tools/lfm2_server.c"
#endif

#if !defined(LFM2_DISABLE_JNI) && (defined(LFM2_ENABLE_JNI) || defined(__ANDROID__) || (defined(__has_include) && __has_include(<jni.h>)))
#include "lfm2_jni.c"
#endif

