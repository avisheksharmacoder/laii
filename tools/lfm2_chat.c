#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Include Step 1 (loader) and Step 2 (forward pass) excluding their demo main() */
#define LFM2_LOADER_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_PREFILL_BATCH_NO_MAIN 1
#define LFM2_CLI_STREAM_NO_MAIN 1
#define LFM2_SYS_CACHE_NO_MAIN 1
#define LFM3_ENGINE_NO_MAIN 1
#include "lfm2_prefill_batch.c"
#include "lfm2_cli_stream.c"
#include "lfm2_sys_cache.c"
#include "lfm3_engine.c"

/* ============================================================================
 * 1. LFM2 Token Sampler (Repetition Penalty + Logit-Space Min-P + Top-K + Temp)
 * ============================================================================ */
typedef struct {
    uint32_t id;
    float    val; /* Holds raw logit during filtering, then probability */
} TokenCand;

typedef struct {
    float     temperature;   /* Default for LFM2.5: 0.30f (0.0f = greedy) */
    float     min_p;         /* Default for LFM2.5: 0.15f                 */
    uint32_t  top_k;         /* Default for LFM2.5: 50                    */
    float     rep_penalty;   /* Default for LFM2.5: 1.05f                 */
    uint32_t  rep_window;    /* Sliding penalty window (e.g., 64 tokens)  */
    uint64_t  rng_state;     /* xorshift64* PRNG state                    */

    /* Pre-allocated buffers (zero malloc during token generation) */
    uint32_t  n_vocab;
    uint32_t *history_ring;  /* [rep_window] recent token IDs             */
    uint32_t  history_count; /* Total tokens seen in current session      */
    TokenCand *cands;        /* [n_vocab] surviving candidate buffer      */
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
    return (xorshift64star(state) >> 8) / 16777216.0f; /* [0.0, 1.0) */
}

static int compare_cands_desc(const void *a, const void *b) {
    const float va = ((const TokenCand *)a)->val;
    const float vb = ((const TokenCand *)b)->val;
    return (va < vb) - (va > vb);
}

bool lfm2_sampler_init(LFM2Sampler *s, uint32_t n_vocab, float temp, float min_p,
                       uint32_t top_k, float rep_penalty, uint32_t rep_window, uint64_t seed) {
    memset(s, 0, sizeof(*s));
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

void lfm2_sampler_reset(LFM2Sampler *s) {
    s->history_count = 0;
}

void lfm2_sampler_free(LFM2Sampler *s) {
    free(s->history_ring);
    free(s->cands);
    memset(s, 0, sizeof(*s));
}

/* Records an ingested or generated token into the repetition penalty ring buffer */
void lfm2_sampler_accept(LFM2Sampler *s, uint32_t token_id) {
    if (s->rep_window == 0) return;
    s->history_ring[s->history_count % s->rep_window] = token_id;
    s->history_count++;
}

/* Executes the full LFM2 sampling pipeline on raw logits[n_vocab] */
uint32_t lfm2_sample_token(LFM2Sampler *s, float *logits) {
    const uint32_t V = s->n_vocab;

    /* ------------------------------------------------------------------------
     * Step 1: Deduplicated Sliding-Window Repetition Penalty
     * Standard HF/llama.cpp rule: if logit > 0, divide by penalty; else multiply.
     * ------------------------------------------------------------------------ */
    if (s->rep_penalty > 1.0001f && s->history_count > 0) {
        const uint32_t win_len = (s->history_count < s->rep_window)
                                 ? s->history_count : s->rep_window;
        for (uint32_t i = 0; i < win_len; ++i) {
            const uint32_t tid = s->history_ring[i];
            if (tid >= V) continue;

            /* Ensure each unique token in the window is penalized only once */
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

    /* ------------------------------------------------------------------------
     * Step 2: Find Global Max Logit (and Greedy Fast-Path if temp <= 0)
     * ------------------------------------------------------------------------ */
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

    /* ------------------------------------------------------------------------
     * Step 3: O(V) Logit-Space Min-P Pre-Filter
     * Instead of sorting 65,536 tokens, discard any token whose temperature-scaled
     * probability cannot possibly reach (min_p * max_prob).
     * ------------------------------------------------------------------------ */
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

    /* Fallback safety if numerical underflow filtered everything */
    if (n_surviving <= 1) {
        lfm2_sampler_accept(s, best_id);
        return best_id;
    }

    /* ------------------------------------------------------------------------
     * Step 4: Sort Surviving Candidates & Apply Top-K Truncation
     * Because Min-P typically reduces 65,536 -> <32 candidates, qsort is instant.
     * ------------------------------------------------------------------------ */
    qsort(s->cands, n_surviving, sizeof(TokenCand), compare_cands_desc);
    if (s->top_k > 0 && n_surviving > s->top_k) {
        n_surviving = s->top_k;
    }

    /* ------------------------------------------------------------------------
     * Step 5: Temperature Scaling + Softmax over Surviving Candidates
     * ------------------------------------------------------------------------ */
    const float inv_temp = 1.0f / s->temperature;
    const float top_logit = s->cands[0].val;
    float sum_prob = 0.0f;

    for (uint32_t i = 0; i < n_surviving; ++i) {
        const float p = expf((s->cands[i].val - top_logit) * inv_temp);
        s->cands[i].val = p;
        sum_prob += p;
    }

    /* ------------------------------------------------------------------------
     * Step 6: Multinomial Sampling via xorshift64*
     * ------------------------------------------------------------------------ */
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
 * 2. LFM2 / LFM2.5 Stateful ChatML Prompt Formatter
 * ============================================================================
 * Exact LFM2 template:
 * Turn 0:
 *   <|startoftext|><|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
 * Turn N (appended directly to active KV + ShortConv state):
 *   \n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
 * ============================================================================ */
typedef struct {
    char system_prompt[1024];
    char tool_json_list[2048]; /* Optional JSON tool definitions */
    bool has_tools;
    bool is_lfm3;
} LFM2ChatFormatter;

void lfm2_chat_formatter_init(LFM2ChatFormatter *fmt, const char *system_prompt,
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

/* Formats a user turn into `out_buf`.
 * If `is_first_turn` is true (pos == 0), emits <|startoftext|> and the system block.
 * If `is_first_turn` is false (pos > 0), emits only the incremental user/assistant turn delta. */
int lfm2_format_chatml_turn(const LFM2ChatFormatter *fmt, const char *user_msg,
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
        /* Incremental continuation right after the assistant's previous <|im_end|> token */
        return snprintf(out_buf, out_cap,
            "\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n%s",
            user_msg, think_prefix);
    }
}

/* ============================================================================
 * 3. Exact GGUF BPE Rank Merging Tokenizer & Vocabulary Engine
 * ============================================================================ */
#include "lfm2_bpe.c"

/* ============================================================================
 * 4. Interactive Terminal Chat REPL with Stateful KV + ShortConv Continuation
 * ============================================================================ */
static double time_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float *chat_prefill_prompt(const LFM2Model *m, LFM2BatchState *st, LFM3State *lfm3,
                                  const uint32_t *tokens, uint32_t n_prompt, uint32_t *inout_pos) {
    if (!lfm3) return lfm2_prefill_prompt(m, st, tokens, n_prompt, inout_pos);
    float *logits = NULL;
    uint32_t i = 0;
    while (i < n_prompt) {
        uint32_t chunk = n_prompt - i;
        if (chunk > LFM2_MAX_BATCH) chunk = LFM2_MAX_BATCH;
        const bool is_last = (i + chunk == n_prompt);
        logits = lfm3_forward_batch(m, lfm3, tokens + i, chunk, *inout_pos, is_last);
        *inout_pos += chunk;
        i += chunk;
    }
    return logits;
}

static inline float *chat_forward_one(const LFM2Model *m, LFM2BatchState *st, LFM3State *lfm3,
                                      uint32_t tok, uint32_t pos, bool compute_logits) {
    return lfm3 ? lfm3_forward_batch(m, lfm3, &tok, 1U, pos, compute_logits)
                : lfm2_forward_batch(m, st,   &tok, 1U, pos, compute_logits);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s -m <path-to-lfm2.gguf> [system_prompt] [--warmup] [--preload] [--prompt-cache <file>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    lfm2_install_sigint_guard();
    LFM2TerminalEnv term;
    lfm2_detect_terminal_env(&term);

    const char *model_path = NULL;
    const char *system_prompt = NULL;
    const char *prompt_cache_path = NULL;
    const char *single_prompt = NULL;
    uint32_t max_turn_tokens = 256;
    bool do_warmup = false;
    const char *extra_stops[LFM2_MAX_STOP_STRINGS];
    uint32_t n_extra_stops = 0;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            max_turn_tokens = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
            if (n_extra_stops < LFM2_MAX_STOP_STRINGS) {
                extra_stops[n_extra_stops++] = argv[++i];
            }
        } else if (strcmp(argv[i], "--prompt-cache") == 0 && i + 1 < argc) {
            prompt_cache_path = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 || strcmp(argv[i], "--preload") == 0) {
            do_warmup = true;
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (argv[i][0] != '-') {
            if (!model_path) {
                model_path = argv[i];
            } else if (!system_prompt) {
                system_prompt = argv[i];
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m <path-to-lfm2.gguf> [-p <prompt>] [-n <max_tokens>] [system_prompt] [--warmup] [--preload] [--prompt-cache <file>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) return EXIT_FAILURE;

    LFM2Vocab vocab;
    if (!lfm2_vocab_load(&vocab, &model)) {
        fprintf(stderr, "[ERROR] Failed to load GGUF vocabulary.\n");
        lfm2_free_model(&model);
        return EXIT_FAILURE;
    }

    LFM2BatchState state;
    memset(&state, 0, sizeof(state));
    LFM3State *lfm3 = NULL;
    uint32_t state_max_seq = 4096U;

    /* Linux /proc/meminfo OOM guard and context verification */
    LFM2MemoryPlan plan = lfm2_plan_memory_and_guard_oom(&model, state_max_seq, 512);
    if (plan.was_clamped) {
        state_max_seq = plan.safe_clamped_ctx;
        fprintf(term.log_stream, "[OOM Guard: Clamped context from %u to %u to prevent out-of-memory kill]\n",
                plan.requested_ctx, plan.safe_clamped_ctx);
    }

    if (model.cfg.is_lfm3) {
        lfm3 = (LFM3State *)calloc(1, sizeof(LFM3State));
        if (!lfm3 || !lfm3_state_init(lfm3, &model, state_max_seq, 0)) {
            fprintf(stderr, "[ERROR] Failed to allocate LFM3 state arena.\n");
            lfm3_state_free(lfm3); free(lfm3);
            lfm2_vocab_free(&vocab); lfm2_free_model(&model);
            return EXIT_FAILURE;
        }
        state_max_seq = lfm3->max_seq_len;
    } else {
        if (!lfm2_batch_state_init(&state, &model, state_max_seq, 0)) {
            fprintf(stderr, "[ERROR] Failed to allocate LFM2 batch state arena.\n");
            lfm2_vocab_free(&vocab); lfm2_free_model(&model);
            return EXIT_FAILURE;
        }
        state_max_seq = state.max_seq_len;
    }

    /* Sequential mmap page warmup if requested */
    if (do_warmup) {
        lfm2_warmup_mmap_pages(&model, term.log_stream, true);
    }

    /* Initialize Sampler with Official Liquid AI LFM2.5 Defaults:
     * temperature = 0.30, min_p = 0.15, top_k = 50, rep_penalty = 1.05 */
    LFM2Sampler sampler;
    lfm2_sampler_init(&sampler, model.cfg.n_vocab, 0.30f, 0.15f, 50, 1.05f, 64, 0);

    LFM2ChatFormatter formatter;
    memset(&formatter, 0, sizeof(formatter));
    formatter.is_lfm3 = model.cfg.is_lfm3;
    lfm2_chat_formatter_init(&formatter, system_prompt, NULL);

    LFM2StreamFilter stream_filter;
    lfm2_stream_filter_init(&stream_filter, NULL, stdout);
    lfm2_stream_filter_add_stop(&stream_filter, "\nUser:");
    lfm2_stream_filter_add_stop(&stream_filter, "<|im_end|>");
    lfm2_stream_filter_add_stop(&stream_filter, "<|endoftext|>");
    for (uint32_t i = 0; i < n_extra_stops; ++i) {
        lfm2_stream_filter_add_stop(&stream_filter, extra_stops[i]);
    }

    if (!single_prompt) {
        FILE *out = term.log_stream;
        fprintf(out, "================================================================\n");
        fprintf(out, " Liquid AI LFM2 / LFM2.5 / LFM3 Zero-Dependency C11 Terminal Chat\n");
        fprintf(out, " Model   : %s (%u layers, %u vocab, max_seq=%u)\n",
                model.cfg.arch, model.cfg.n_layers, model.cfg.n_vocab, state_max_seq);
        fprintf(out, " Tokens  : BOS=%u (<|startoftext|>) | IM_START=%u | IM_END=%u\n",
                vocab.id_bos, vocab.id_im_start, vocab.id_im_end);
        fprintf(out, " Sampler : temp=%.2f | min_p=%.2f | top_k=%u | rep_penalty=%.2f\n",
                sampler.temperature, sampler.min_p, sampler.top_k, sampler.rep_penalty);
        fprintf(out, " Filter  : UTF-8 boundary buffer active | %u stop strings registered\n",
                stream_filter.n_stops);
        fprintf(out, " Signals : Two-stage SIGINT active (Ctrl+C cancels turn, 2nd exits)\n");
        fprintf(out, " Commands: /reset (clear state), /temp <f>, /minp <f>, /max <n>, /exit\n");
        fprintf(out, "================================================================\n");
    }

    uint32_t pos = 0;
    bool cache_restored = false;
    if (prompt_cache_path) {
        char sys_text[4096];
        uint32_t sys_tokens[2048];
        snprintf(sys_text, sizeof(sys_text), "<|startoftext|><|im_start|>system\n%s<|im_end|>\n",
                 system_prompt ? system_prompt : "You are a helpful assistant.");
        uint32_t n_sys = lfm2_tokenize(&vocab, sys_text, sys_tokens, 2048);
        if (n_sys > 0) {
            uint32_t matched = 0;
            bool loaded = lfm3
                ? lfm2_prompt_cache_try_load_prefix_ex(prompt_cache_path, &model, sys_tokens, n_sys,
                                                       lfm3->max_seq_len, lfm3->conv_ring, lfm3->conv_ring_head,
                                                       lfm3->k_cache_f16, lfm3->v_cache_f16, lfm3->swa_cap, &matched)
                : lfm2_prompt_cache_try_load_prefix(prompt_cache_path, &model, sys_tokens, n_sys,
                                                    state.max_seq_len, state.conv_state,
                                                    state.k_cache_f16, state.v_cache_f16, &matched);
            if (loaded) {
                pos = matched;
                for (uint32_t i = 0; i < pos; ++i) {
                    lfm2_sampler_accept(&sampler, sys_tokens[i]);
                }
                cache_restored = true;
                fprintf(term.log_stream, "[Prompt Cache: Restored %u system prefix tokens from %s]\n",
                        matched, prompt_cache_path);
            } else {
                for (uint32_t i = 0; i < n_sys; ++i) {
                    lfm2_sampler_accept(&sampler, sys_tokens[i]);
                }
                chat_prefill_prompt(&model, &state, lfm3, sys_tokens, n_sys, &pos);
                bool saved = lfm3
                    ? lfm2_prompt_cache_save_ex(prompt_cache_path, &model, sys_tokens, n_sys,
                                                (float * const *)lfm3->conv_ring, lfm3->conv_ring_head,
                                                (uint16_t * const *)lfm3->k_cache_f16,
                                                (uint16_t * const *)lfm3->v_cache_f16, lfm3->swa_cap)
                    : lfm2_prompt_cache_save(prompt_cache_path, &model, sys_tokens, n_sys,
                                             state.conv_state, state.k_cache_f16, state.v_cache_f16);
                if (saved) {
                    fprintf(term.log_stream, "[Prompt Cache: Cached %u system prefix tokens to %s]\n",
                            n_sys, prompt_cache_path);
                    cache_restored = true;
                }
            }
        }
    }

    char user_input[2048];
    char formatted_prompt[4096];
    uint32_t prompt_ids[2048];

    if (single_prompt != NULL) {
        /* Single-shot non-interactive or UNIX pipe mode */
        lfm2_format_chatml_turn(&formatter, single_prompt, (pos == 0),
                                formatted_prompt, sizeof(formatted_prompt));
        uint32_t n_prompt = lfm2_tokenize(&vocab, formatted_prompt, prompt_ids, 2048);
        if (n_prompt == 0 || pos + n_prompt >= state_max_seq) {
            fprintf(stderr, "[ERROR] Prompt empty or exceeds max context length (%u >= %u)\n",
                    pos + n_prompt, state_max_seq);
            lfm2_sampler_free(&sampler);
            if (lfm3) { lfm3_state_free(lfm3); free(lfm3); } else { lfm2_batch_state_free(&state); }
            lfm2_vocab_free(&vocab);
            lfm2_free_model(&model);
            return EXIT_FAILURE;
        }

        double t_prefill_start = time_now_ms();
        for (uint32_t i = 0; i < n_prompt; ++i) {
            lfm2_sampler_accept(&sampler, prompt_ids[i]);
        }
        float *logits = chat_prefill_prompt(&model, &state, lfm3, prompt_ids, n_prompt, &pos);
        double prefill_ms = time_now_ms() - t_prefill_start;

        double t_decode_start = time_now_ms();
        uint32_t gen_tokens = 0;
        lfm2_stream_filter_reset(&stream_filter);

        while (pos < state_max_seq && gen_tokens < max_turn_tokens) {
            uint32_t next_id = lfm2_sample_token(&sampler, logits);
            gen_tokens++;

            if (next_id == vocab.id_im_end || next_id == vocab.id_bos) {
                break;
            }

            char tok_buf[512];
            size_t tok_len = lfm2_detokenize_token(&vocab, next_id, tok_buf, sizeof(tok_buf));
            lfm2_stream_status_t s_st = lfm2_stream_filter_feed(&stream_filter, tok_buf, tok_len);
            if (s_st == LFM2_STREAM_STOPPED) {
                break;
            }

            logits = chat_forward_one(&model, &state, lfm3, next_id, pos++, true);
        }

        lfm2_stream_filter_flush(&stream_filter);
        printf("\n");
        fflush(stdout);
        double decode_ms = time_now_ms() - t_decode_start;

        fprintf(term.log_stream, "[prefill: %u tok in %.1f ms (%.1f tok/s) | "
                                 "decode: %u tok in %.1f ms (%.2f tok/s) | ctx: %u]\n",
                n_prompt, prefill_ms, (n_prompt * 1000.0) / (prefill_ms + 1e-6),
                gen_tokens, decode_ms, (gen_tokens * 1000.0) / (decode_ms + 1e-6), pos);

        lfm2_sampler_free(&sampler);
        if (lfm3) { lfm3_state_free(lfm3); free(lfm3); } else { lfm2_batch_state_free(&state); }
        lfm2_vocab_free(&vocab);
        lfm2_free_model(&model);
        return EXIT_SUCCESS;
    }

    while (1) {
        if (lfm2_sigint_should_exit()) break;

        if (term.use_ansi_colors) {
            printf("\n\033[1;32mUser >\033[0m ");
        } else {
            printf("\nUser > ");
        }
        fflush(stdout);

        if (!fgets(user_input, sizeof(user_input), stdin)) {
            if (lfm2_sigint_should_exit()) break;
            if (errno == EINTR) {
                lfm2_sigint_clear_exit();
                printf("\n");
                continue;
            }
            break;
        }

        /* Strip trailing newline */
        size_t len = strlen(user_input);
        while (len > 0 && (user_input[len - 1] == '\n' || user_input[len - 1] == '\r')) {
            user_input[--len] = '\0';
        }
        if (len == 0) continue;

        /* Handle REPL slash commands */
        if (strcmp(user_input, "/exit") == 0 || strcmp(user_input, "/quit") == 0) break;
        if (strcmp(user_input, "/reset") == 0) {
            if (lfm3) {
                lfm3_state_reset(lfm3, &model);
            } else {
                lfm2_batch_state_free(&state);
                lfm2_batch_state_init(&state, &model, state_max_seq, 0);
            }
            lfm2_sampler_reset(&sampler);
            pos = 0;
            if (prompt_cache_path && cache_restored) {
                char sys_text[4096];
                uint32_t sys_tokens[2048];
                snprintf(sys_text, sizeof(sys_text), "<|startoftext|><|im_start|>system\n%s<|im_end|>\n",
                         system_prompt ? system_prompt : "You are a helpful assistant.");
                uint32_t n_sys = lfm2_tokenize(&vocab, sys_text, sys_tokens, 2048);
                uint32_t matched = 0;
                bool ok = lfm3
                    ? lfm2_prompt_cache_try_load_prefix_ex(prompt_cache_path, &model, sys_tokens, n_sys,
                                                           lfm3->max_seq_len, lfm3->conv_ring, lfm3->conv_ring_head,
                                                           lfm3->k_cache_f16, lfm3->v_cache_f16, lfm3->swa_cap, &matched)
                    : lfm2_prompt_cache_try_load_prefix(prompt_cache_path, &model, sys_tokens, n_sys,
                                                        state.max_seq_len, state.conv_state,
                                                        state.k_cache_f16, state.v_cache_f16, &matched);
                if (ok) {
                    pos = matched;
                    for (uint32_t i = 0; i < pos; ++i) lfm2_sampler_accept(&sampler, sys_tokens[i]);
                    printf("[State reset: Reloaded system prefix from cache (%u tokens).]\n", pos);
                    continue;
                }
            }
            printf("[State reset: KV caches and ShortConv ring buffers cleared.]\n");
            continue;
        }
        if (strncmp(user_input, "/temp ", 6) == 0) {
            sampler.temperature = strtof(user_input + 6, NULL);
            printf("[Sampler temperature updated to %.2f]\n", sampler.temperature);
            continue;
        }
        if (strncmp(user_input, "/minp ", 6) == 0) {
            sampler.min_p = strtof(user_input + 6, NULL);
            printf("[Sampler min_p updated to %.2f]\n", sampler.min_p);
            continue;
        }
        if (strncmp(user_input, "/max ", 5) == 0) {
            max_turn_tokens = (uint32_t)atoi(user_input + 5);
            printf("[Max tokens per turn updated to %u]\n", max_turn_tokens);
            continue;
        }

        /* 1. Format Turn in LFM2 ChatML (First turn vs Stateful Continuation) */
        lfm2_format_chatml_turn(&formatter, user_input, (pos == 0),
                                formatted_prompt, sizeof(formatted_prompt));

        /* 2. Tokenize Formatted Prompt */
        uint32_t n_prompt = lfm2_tokenize(&vocab, formatted_prompt, prompt_ids, 2048);
        if (n_prompt == 0 || pos + n_prompt >= state_max_seq) {
            fprintf(stderr, "[WARN] Context window full or empty prompt. Type /reset.\n");
            continue;
        }

        /* 3. Batched Prompt Prefill (B=8) through ShortConv + GQA Layers */
        double t_prefill_start = time_now_ms();
        for (uint32_t i = 0; i < n_prompt; ++i) {
            lfm2_sampler_accept(&sampler, prompt_ids[i]);
        }
        float *logits = chat_prefill_prompt(&model, &state, lfm3, prompt_ids, n_prompt, &pos);
        double prefill_ms = time_now_ms() - t_prefill_start;

        /* 4. Autoregressive Decode Loop until <|im_end|> or max_turn_tokens */
        if (term.use_ansi_colors) {
            printf("\033[1;36m%s >\033[0m ", lfm3 ? "LFM3" : "LFM2");
        } else {
            printf("%s > ", lfm3 ? "LFM3" : "LFM2");
        }
        fflush(stdout);

        double t_decode_start = time_now_ms();
        uint32_t gen_tokens = 0;
        bool cancelled = false;

        lfm2_stream_filter_reset(&stream_filter);
        lfm2_sigint_begin_generation();

        while (pos < state_max_seq && gen_tokens < max_turn_tokens) {
            /* Check if user pressed Ctrl+C during token decode */
            if (lfm2_sigint_should_abort()) {
                cancelled = true;
                lfm2_stream_filter_flush(&stream_filter);
                if (term.use_ansi_colors) {
                    fprintf(term.log_stream, "\n\033[33m[Generation interrupted by Ctrl+C]\033[0m\n");
                } else {
                    fprintf(term.log_stream, "\n[Generation interrupted by Ctrl+C]\n");
                }
                /* Close turn in state so state remains consistent */
                chat_forward_one(&model, &state, lfm3, vocab.id_im_end, pos++, false);
                break;
            }

            uint32_t next_id = lfm2_sample_token(&sampler, logits);
            gen_tokens++;

            /* Stop generation cleanly when LFM2/LFM3 emits <|im_end|> or EOS */
            if (next_id == vocab.id_im_end || next_id == vocab.id_bos) {
                break;
            }

            /* Feed token bytes through unified UTF-8 & Stop-String filter */
            char tok_buf[512];
            size_t tok_len = lfm2_detokenize_token(&vocab, next_id, tok_buf, sizeof(tok_buf));
            lfm2_stream_status_t s_st = lfm2_stream_filter_feed(&stream_filter, tok_buf, tok_len);
            if (s_st == LFM2_STREAM_STOPPED) {
                /* Multi-token stop string matched! Close turn in state */
                chat_forward_one(&model, &state, lfm3, vocab.id_im_end, pos++, false);
                break;
            }

            logits = chat_forward_one(&model, &state, lfm3, next_id, pos++, true);
        }

        lfm2_stream_filter_flush(&stream_filter);
        lfm2_sigint_end_generation();
        if (cancelled) {
            lfm2_sigint_clear_abort();
        }

        double decode_ms = time_now_ms() - t_decode_start;

        if (term.use_ansi_colors) {
            fprintf(term.log_stream, "\n\033[90m[prefill: %u tok in %.1f ms (%.1f tok/s) | "
                   "decode: %u tok in %.1f ms (%.2f tok/s) | ctx_pos: %u]\033[0m\n",
                   n_prompt, prefill_ms, (n_prompt * 1000.0) / (prefill_ms + 1e-6),
                   gen_tokens, decode_ms, (gen_tokens * 1000.0) / (decode_ms + 1e-6), pos);
        } else {
            fprintf(term.log_stream, "\n[prefill: %u tok in %.1f ms (%.1f tok/s) | "
                   "decode: %u tok in %.1f ms (%.2f tok/s) | ctx_pos: %u]\n",
                   n_prompt, prefill_ms, (n_prompt * 1000.0) / (prefill_ms + 1e-6),
                   gen_tokens, decode_ms, (gen_tokens * 1000.0) / (decode_ms + 1e-6), pos);
        }
    }

    lfm2_sampler_free(&sampler);
    if (lfm3) { lfm3_state_free(lfm3); free(lfm3); } else { lfm2_batch_state_free(&state); }
    lfm2_vocab_free(&vocab);
    lfm2_free_model(&model);
    return EXIT_SUCCESS;
}
