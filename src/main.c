#define _POSIX_C_SOURCE 200809L
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <stdbool.h>

#include "lfm_engine.h"
#include "lfm_defaults.h"
#include "lfm_simd.h"
#include "lfm_c_api.h"
#include "lfm_server.h"
#include "lfm_cli_stream.h"
#include "lfm_sys_cache.h"
#include "lfm2_simd_avx2.h"

typedef struct {
    LFM2StreamFilter filter;
    volatile bool stop_signal;
} MainStreamCtx;

static void main_token_stream_cb(uint32_t token_id, const char *piece, void *user_data) {
    (void)token_id;
    MainStreamCtx *ctx = (MainStreamCtx *)user_data;
    if (!piece || !ctx) return;

    if (lfm2_sigint_should_abort()) {
        ctx->stop_signal = true;
        return;
    }

    lfm2_stream_status_t status = lfm2_stream_filter_feed(&ctx->filter, piece, strlen(piece));
    if (status == LFM2_STREAM_STOPPED) {
        ctx->stop_signal = true;
    }
}

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_banner(FILE *out, const LFMEngine *engine) {
    if (!out) out = stdout;
    const LFMConfig *cfg = &engine->model.config;
    uint32_t conv_layers = 0;
    uint32_t attn_layers = 0;
    for (uint32_t i = 0; i < cfg->n_layers; ++i) {
        if (cfg->is_conv_layer[i]) conv_layers++;
        else attn_layers++;
    }

    fprintf(out, "=================================================================\n");
    fprintf(out, "   Liquid AI (LFM2) Specialized C Inference Engine (v1.0)       \n");
    fprintf(out, "=================================================================\n");
    fprintf(out, " Acceleration:       %s | %s\n",
#if LFM_HAS_AVX2
           "AVX2 + FMA SIMD Kernels Active",
           "Fused LIV ShortConv Kernel Active"
#else
           "Portable Scalar Fallback",
           "Standard LIV ShortConv"
#endif
    );
    fprintf(out, " Execution Threads:  %u CPU Worker Threads\n", engine->threadpool.num_threads);
    fprintf(out, " Model Architecture: LFM2 Hybrid Backbone\n");
    fprintf(out, " Total Layers:       %u (%u LIV Conv1D + %u GQA Attention)\n",
           cfg->n_layers, conv_layers, attn_layers);
    fprintf(out, " Hidden Dim:         %u | FFN Dim: %u\n", cfg->hidden_dim, cfg->ffn_dim);
    fprintf(out, " Heads (Q / KV):     %u / %u (Head Dim: %u)\n",
           cfg->n_heads_q, cfg->n_heads_kv, cfg->head_dim);
    fprintf(out, " Recurrent Conv:     k=%u (Ring Buffer State: %u tokens)\n",
           cfg->conv_L_cache, cfg->conv_L_cache - 1);
    fprintf(out, " KV Cache Footprint: ~%.2f MB (reduced by ~%.1f%% vs pure Transformer)\n",
           (double)(attn_layers * 2 * cfg->max_seq_len * cfg->n_heads_kv * cfg->head_dim * 4) / (1024.0 * 1024.0),
           (double)conv_layers / (double)cfg->n_layers * 100.0);
    fprintf(out, "=================================================================\n\n");
}

int main(int argc, char **argv) {
    lfm2_set_denormals_zero_mode();
    const char *model_path = NULL;
    const char *prompt = "Explain why Liquid Foundation Models are faster on edge CPUs:";
    bool prompt_specified = false;
    uint32_t max_tokens = (LFM_DEFAULT_MAX_TOKENS == 0) ? UINT32_MAX : (uint32_t)LFM_DEFAULT_MAX_TOKENS;
    float temp = 0.7f;
    uint32_t top_k = 40;
    uint32_t seed = (uint32_t)time(NULL);
    uint32_t threads = 0; // 0 = auto-detect hardware threads
    bool interactive = false;
    bool run_server = false;
    int server_port = 8080;
    const char *prompt_cache_path = NULL;
    bool warmup_pages = false;

    const char *custom_stops[LFM2_MAX_STOP_STRINGS];
    int n_custom_stops = 0;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
            prompt_specified = true;
        } else if (strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
            if (n_custom_stops < LFM2_MAX_STOP_STRINGS) {
                custom_stops[n_custom_stops++] = argv[++i];
            } else {
                i++;
            }
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            uint32_t parsed = lfm_parse_max_tokens_arg(argv[++i]);
            max_tokens = (parsed == 0) ? UINT32_MAX : parsed;
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            threads = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--prompt-cache") == 0 && i + 1 < argc) {
            prompt_cache_path = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0) {
            warmup_pages = true;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--server") == 0) {
            run_server = true;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            server_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s -m <model.gguf> [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -m, --model <path>       Path to LFM GGUF model file (required)\n");
            printf("  -p, --prompt <string>    Prompt text to generate from\n");
            printf("      --stop <string>      Register stop string (can be specified multiple times)\n");
            printf("  -n, --n-predict <int>    Maximum number of tokens to generate (default: %u, 0 = unlimited)\n", (unsigned)LFM_DEFAULT_MAX_TOKENS);
            printf("  -t, --temp <float>       Temperature (default: 0.7, 0 = greedy)\n");
            printf("      --top-k <int>        Top-K filtering (default: 40)\n");
            printf("  -j, --threads <int>      Number of CPU worker threads (default: auto, N-1 physical cores)\n");
            printf("  -i, --interactive        Interactive chat mode\n");
            printf("      --warmup             Sequentially warm up mmap pages into RAM before generation\n");
            printf("      --prompt-cache <f>   File to load/save hybrid prompt prefix cache (.lfmcache)\n");
            printf("  -s, --server             Run as HTTP REST server and Web Studio\n");
            printf("      --port <int>         HTTP server port (default: 8080)\n");
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: Model path is required (-m <model.gguf>)\n");
        fprintf(stderr, "Run with -h for help or run test suite with 'make test'\n");
        return 1;
    }

    /* Detect UNIX terminal environment and pipe redirection */
    LFM2TerminalEnv term;
    lfm2_detect_terminal_env(&term);

    /* Ingest piped stdin until EOF when stdin is not a TTY and not running server/interactive */
    char *stdin_text = NULL;
    char *allocated_prompt = NULL;
    if (!term.stdin_is_tty && !run_server && !interactive) {
        size_t cap = 8192;
        size_t len = 0;
        stdin_text = (char *)malloc(cap);
        if (stdin_text) {
            size_t nread = 0;
            while ((nread = fread(stdin_text + len, 1, cap - len - 1, stdin)) > 0) {
                len += nread;
                if (len + 1024 >= cap) {
                    cap *= 2;
                    char *next_buf = (char *)realloc(stdin_text, cap);
                    if (!next_buf) break;
                    stdin_text = next_buf;
                }
            }
            stdin_text[len] = '\0';

            if (len > 0) {
                if (prompt_specified && prompt && prompt[0] != '\0') {
                    size_t combined_len = strlen(prompt) + len + 2;
                    allocated_prompt = (char *)malloc(combined_len);
                    if (allocated_prompt) {
                        snprintf(allocated_prompt, combined_len, "%s\n%s", prompt, stdin_text);
                        prompt = allocated_prompt;
                    }
                } else {
                    prompt = stdin_text;
                }
            }
        }
    }

    /* Install two-stage SIGINT handler */
    lfm2_install_sigint_guard();

    /* Diagnostic banner sent to term.log_stream (stderr if redirected, stdout if TTY) */
    fprintf(term.log_stream, "Loading LFM model from %s ...\n", model_path);
    LFMEngine engine;
    if (!lfm_engine_init(&engine, model_path, threads)) {
        fprintf(stderr, "Error: Failed to initialize LFM engine.\n");
        if (allocated_prompt) free(allocated_prompt);
        if (stdin_text) free(stdin_text);
        return 1;
    }

    print_banner(term.log_stream, &engine);

    /* Linux /proc/meminfo OOM guard and context verification */
    LFM2LinuxMemInfo sys_mem;
    if (lfm2_read_linux_meminfo(&sys_mem)) {
        LFM2Model dummy_m;
        memset(&dummy_m, 0, sizeof(dummy_m));
        snprintf(dummy_m.cfg.arch, sizeof(dummy_m.cfg.arch), "%s", engine.model.config.architecture);
        dummy_m.cfg.n_layers = engine.model.config.n_layers;
        dummy_m.cfg.n_embd = engine.model.config.hidden_dim;
        dummy_m.cfg.n_vocab = engine.model.config.vocab_size;
        dummy_m.cfg.n_ctx = engine.model.config.max_seq_len;
        dummy_m.cfg.n_head_q = engine.model.config.n_heads_q;
        dummy_m.cfg.shortconv_l_cache = engine.model.config.conv_L_cache;
        for (uint32_t l = 0; l < dummy_m.cfg.n_layers; ++l) {
            dummy_m.layers[l].block_type = engine.model.config.is_conv_layer[l] ? LFM2_BLOCK_SHORTCONV : LFM2_BLOCK_GQA_ATTN;
            dummy_m.cfg.n_head_kv[l] = engine.model.config.is_conv_layer[l] ? 0 : engine.model.config.layer_heads_kv[l];
        }
        dummy_m.file_size = engine.model.gguf.file_size;

        LFM2MemoryPlan plan = lfm2_plan_memory_and_guard_oom(&dummy_m, engine.model.config.max_seq_len, 512);
        if (plan.was_clamped) {
            fprintf(term.log_stream, "[OOM Guard: Clamped safe context from %u to %u to avoid out-of-memory kill]\n",
                    plan.requested_ctx, plan.safe_clamped_ctx);
        }
    }

    /* Sequential mmap page warmup if requested */
    if (warmup_pages && engine.model.gguf.mmap_addr) {
        LFM2Model dummy_m;
        memset(&dummy_m, 0, sizeof(dummy_m));
        dummy_m.fd = engine.model.gguf.fd;
        dummy_m.file_size = engine.model.gguf.file_size;
        dummy_m.mmap_base = (const uint8_t *)engine.model.gguf.mmap_addr;
        lfm2_warmup_mmap_pages(&dummy_m, term.log_stream, true);
    }

    /* Configure unified stream filter and holdback buffer */
    MainStreamCtx stream_ctx;
    memset(&stream_ctx, 0, sizeof(stream_ctx));
    lfm2_stream_filter_init(&stream_ctx.filter, lfm2_stdout_emit_cb, stdout);

    /* Register default stop strings */
    lfm2_stream_filter_add_stop(&stream_ctx.filter, "\nUser:");
    lfm2_stream_filter_add_stop(&stream_ctx.filter, "<|im_end|>");
    lfm2_stream_filter_add_stop(&stream_ctx.filter, "<|endoftext|>");

    /* Register custom stop strings */
    for (int i = 0; i < n_custom_stops; ++i) {
        lfm2_stream_filter_add_stop(&stream_ctx.filter, custom_stops[i]);
    }

    LFMSamplerConfig sampler = {
        .temperature = temp,
        .top_k = top_k,
        .max_new_tokens = max_tokens,
        .seed = seed,
        .echo_prompt = false,
        .stop_signal = &stream_ctx.stop_signal,
        .prompt_cache_path = prompt_cache_path
    };

    if (run_server) {
        LFMServerConfig s_cfg = {
            .host = "0.0.0.0",
            .port = server_port
        };
        lfm_server_start((lfm_engine_t*)&engine, &s_cfg);
    } else if (interactive) {
        if (term.use_ansi_colors) {
            fprintf(term.log_stream, "\033[1;32mEntering interactive mode.\033[0m Type your prompt and press Enter (Ctrl+C to interrupt turn or exit):\n\n");
        } else {
            fprintf(term.log_stream, "Entering interactive mode. Type your prompt and press Enter (Ctrl+C to interrupt turn or exit):\n\n");
        }

        char line[4096];
        while (1) {
            if (lfm2_sigint_should_exit()) {
                fprintf(term.log_stream, "\nExiting chat session.\n");
                break;
            }

            if (term.use_ansi_colors) {
                fprintf(stdout, "\n\033[1;32mUser >\033[0m ");
            } else {
                fprintf(stdout, "\nUser > ");
            }
            fflush(stdout);

            char *ptr = fgets(line, sizeof(line), stdin);
            if (!ptr) {
                if (lfm2_sigint_should_exit() || feof(stdin)) {
                    fprintf(term.log_stream, "\nExiting chat session.\n");
                    break;
                }
                if (errno == EINTR) {
                    clearerr(stdin);
                    if (lfm2_sigint_should_exit()) {
                        fprintf(term.log_stream, "\nExiting chat session.\n");
                        break;
                    }
                    continue;
                }
                break;
            }

            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;

            if (term.use_ansi_colors) {
                fprintf(stdout, "\033[1;36mLFM >\033[0m ");
            } else {
                fprintf(stdout, "LFM > ");
            }
            fflush(stdout);

            lfm2_stream_filter_reset(&stream_ctx.filter);
            stream_ctx.stop_signal = false;
            lfm2_sigint_begin_generation();

            double t0 = get_time_sec();
            size_t gen = lfm_engine_generate(&engine, line, &sampler, main_token_stream_cb, &stream_ctx);
            lfm2_stream_filter_flush(&stream_ctx.filter);
            lfm2_sigint_end_generation();

            bool interrupted = lfm2_sigint_should_abort();
            if (interrupted) {
                lfm2_sigint_clear_abort();
                if (term.use_ansi_colors) {
                    fprintf(term.log_stream, "\n\033[33m[Generation interrupted by Ctrl+C]\033[0m\n");
                } else {
                    fprintf(term.log_stream, "\n[Generation interrupted by Ctrl+C]\n");
                }
            }

            double dt = get_time_sec() - t0;
            if (term.use_ansi_colors) {
                fprintf(term.log_stream, "\n\033[90m[%.2f tokens/sec (%zu tokens in %.3fs)]\033[0m\n",
                       dt > 0 ? (double)gen / dt : 0.0, gen, dt);
            } else {
                fprintf(term.log_stream, "\n[%.2f tokens/sec (%zu tokens in %.3fs)]\n",
                       dt > 0 ? (double)gen / dt : 0.0, gen, dt);
            }
        }
    } else {
        if (term.stdout_is_tty) {
            fprintf(term.log_stream, "Prompt: \"%s\"\n\nResponse:\n", prompt);
        }
        lfm2_stream_filter_reset(&stream_ctx.filter);
        stream_ctx.stop_signal = false;
        lfm2_sigint_begin_generation();

        double t0 = get_time_sec();
        size_t gen = lfm_engine_generate(&engine, prompt, &sampler, main_token_stream_cb, &stream_ctx);
        lfm2_stream_filter_flush(&stream_ctx.filter);
        lfm2_sigint_end_generation();

        bool interrupted = lfm2_sigint_should_abort();
        if (interrupted) {
            lfm2_sigint_clear_abort();
            fprintf(term.log_stream, "\n[Generation interrupted by Ctrl+C]\n");
        }

        double dt = get_time_sec() - t0;
        if (term.stdout_is_tty) {
            fprintf(term.log_stream, "\n\n-----------------------------------------------------------------\n");
            fprintf(term.log_stream, "Generation finished: %zu tokens in %.3fs (%.2f tokens/sec)\n",
                   gen, dt, dt > 0 ? (double)gen / dt : 0.0);
            fprintf(term.log_stream, "-----------------------------------------------------------------\n");
        } else {
            fprintf(term.log_stream, "\n[info: %zu tokens in %.3fs (%.2f tok/s)]\n",
                   gen, dt, dt > 0 ? (double)gen / dt : 0.0);
        }
    }

    if (allocated_prompt) free(allocated_prompt);
    if (stdin_text) free(stdin_text);
    lfm_engine_free(&engine);
    return 0;
}
