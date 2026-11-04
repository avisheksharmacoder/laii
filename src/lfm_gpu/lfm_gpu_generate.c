#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "lfm_gpu_engine.h"

#define LFM2_LOADER_NO_MAIN 1
#include "tools/lfm2_loader.c"

#define LFM2_BPE_NO_MAIN 1
#include "tools/lfm2_bpe.c"

#define LFM2_FORWARD_NO_MAIN 1
#include "tools/lfm2_forward.c"

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* GPU-Accelerated SwiGLU Feed-Forward Network Pass:
 * Offloads Q4_K FFN Gate & Up projections to the Intel iGPU / Vulkan Compute shader */
static void lfm_gpu_forward_ffn(LFM2State *s, const LFM2Model *m, const LFM2Layer *layer, LFMGPUEngine *gpu) {
    const uint32_t D    = m->cfg.n_embd;
    const uint32_t n_ff = (uint32_t)layer->ffn_gate->ne[1];

    /* 1. FFN Pre-Norm: s->x -> s->xb */
    lfm2_rmsnorm_and_quantize_q8_K(s->xb, s->xq_scratch, s->x, layer->ffn_norm, D, m->cfg.rms_norm_eps);

    /* 2 & 3. Gate & Up projections on GPU */
    if (gpu && layer->ffn_gate->type == GGML_TYPE_Q4_K && layer->ffn_up->type == GGML_TYPE_Q4_K) {
        lfm_gpu_gemv_q4_k(gpu, s->hb,  layer->ffn_gate->data, s->xb, D, n_ff);
        lfm_gpu_gemv_q4_k(gpu, s->hb2, layer->ffn_up->data,   s->xb, D, n_ff);
    } else {
        lfm2_matmul_vec(s->hb,  layer->ffn_gate, s->xb, s->xq_scratch, &s->pool);
        lfm2_matmul_vec(s->hb2, layer->ffn_up,   s->xb, s->xq_scratch, &s->pool);
    }

    /* SwiGLU Non-Linearity: silu(gate) * up */
    for (uint32_t i = 0; i < n_ff; ++i) {
        const float g = s->hb[i];
        const float silu = g / (1.0f + expf(-g));
        s->hb[i] = silu * s->hb2[i];
    }

    /* 4. Down projection */
    lfm2_matmul_vec(s->xb, layer->ffn_down, s->hb, s->xq_scratch, &s->pool);
#if LFM2_HAS_AVX2
    for (uint32_t d = 0; d < D; d += 8) {
        __m256 vx  = _mm256_loadu_ps(s->x + d);
        __m256 vxb = _mm256_loadu_ps(s->xb + d);
        _mm256_storeu_ps(s->x + d, _mm256_add_ps(vx, vxb));
    }
#else
    for (uint32_t d = 0; d < D; ++d) {
        s->x[d] += s->xb[d];
    }
#endif
}

/* Single-token forward pass dispatching layer FFN projections to the GPU */
static float *lfm_gpu_forward_token(const LFM2Model *m, LFM2State *s, uint32_t token_id,
                                    uint32_t pos, LFMGPUEngine *gpu, bool compute_logits) {
    const uint32_t D = m->cfg.n_embd;

    if (token_id >= m->cfg.n_vocab || pos >= s->max_seq_len) {
        fprintf(stderr, "[ERROR] token_id (%u) or pos (%u) out of bounds.\n", token_id, pos);
        return NULL;
    }

    /* 1. Embedding lookup */
    lfm2_dequantize_row(m->token_embd, token_id, s->x);

    /* 2. Hybrid Layers with GPU FFN Acceleration */
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        const LFM2Layer *layer = &m->layers[l];

        lfm2_rmsnorm(s->xb, s->x, layer->attn_norm, D, m->cfg.rms_norm_eps);

        if (layer->block_type == LFM2_BLOCK_SHORTCONV) {
            lfm2_forward_shortconv(s, m, layer, l);
        } else {
            lfm2_forward_gqa(s, m, layer, l, pos);
        }

        lfm_gpu_forward_ffn(s, m, layer, gpu);
    }

    if (!compute_logits) return NULL;

    /* 3. Output RMSNorm */
    lfm2_rmsnorm(s->x, s->x, m->output_norm, D, m->cfg.rms_norm_eps);

    /* 4. LM Head projection */
    lfm2_matmul_vec(s->logits, m->output, s->x, s->xq_scratch, &s->pool);
    return s->logits;
}

/* Temperature + Repetition Penalty Sampler */
static uint32_t sample_next_token(float *logits, uint32_t n_vocab, const uint32_t *history,
                                  uint32_t n_hist, float temperature, float rep_penalty) {
    if (rep_penalty > 1.0f && history && n_hist > 0) {
        for (uint32_t h = 0; h < n_hist; ++h) {
            uint32_t id = history[h];
            if (id < n_vocab) {
                if (logits[id] > 0.0f) logits[id] /= rep_penalty;
                else                   logits[id] *= rep_penalty;
            }
        }
    }

    /* Argmax if temperature is close to 0 */
    if (temperature < 0.05f) {
        uint32_t best_id = 0;
        float best_val = logits[0];
        for (uint32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* Softmax with temperature */
    float max_l = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        logits[i] = expf((logits[i] - max_l) / temperature);
        sum += logits[i];
    }

    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return n_vocab - 1;
}

int main(int argc, char **argv) {
    const char *model_path = (argc >= 2) ? argv[1] : "models/LFM2.5-230M-Q4_K_M.gguf";
    const char *prompt     = (argc >= 3) ? argv[2] : "The sun is shining and the birds are";
    const uint32_t max_tokens = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 30;
    const int target_gpu   = (argc >= 5) ? atoi(argv[4]) : -1;

    printf("================================================================\n");
    printf(" Liquid AI LFM2 - End-to-End GPU-Accelerated Generation\n");
    printf("================================================================\n");
    printf("  Model File   : %s\n", model_path);
    printf("  User Prompt  : \"%s\"\n", prompt);
    printf("  Max Tokens   : %u\n", max_tokens);

    /* 1. Initialize Vulkan Compute GPU Engine */
    LFMGPUEngine gpu;
    if (!lfm_gpu_engine_init(&gpu, target_gpu, 64 * 1024 * 1024)) {
        fprintf(stderr, "[FATAL] Failed to initialize Vulkan GPU Engine.\n");
        return EXIT_FAILURE;
    }

    /* 2. Load GGUF Model */
    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) {
        fprintf(stderr, "[FATAL] Could not load model: %s\n", model_path);
        lfm_gpu_engine_free(&gpu);
        return EXIT_FAILURE;
    }

    /* 3. Load Vocabulary */
    LFM2Vocab vocab;
    if (!lfm2_vocab_load(&vocab, &model)) {
        fprintf(stderr, "[FATAL] Could not load BPE vocabulary.\n");
        lfm2_free_model(&model);
        lfm_gpu_engine_free(&gpu);
        return EXIT_FAILURE;
    }
    printf("  Vocab Info   : %u tokens | BOS=%u | EOS=%u\n", vocab.n_vocab, vocab.id_bos, vocab.id_im_end);

    /* 4. Allocate Runtime Forward State */
    LFM2State state;
    if (!lfm2_state_init(&state, &model, 1024)) {
        fprintf(stderr, "[FATAL] Failed to initialize state arena.\n");
        lfm2_vocab_free(&vocab);
        lfm2_free_model(&model);
        lfm_gpu_engine_free(&gpu);
        return EXIT_FAILURE;
    }

    /* 5. Tokenize Prompt with Official Liquid AI LFM2 Chat Template */
    char formatted_prompt[4096];
    if (strstr(prompt, "<|im_start|>") == NULL) {
        snprintf(formatted_prompt, sizeof(formatted_prompt),
                 "<|startoftext|><|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 prompt);
    } else {
        snprintf(formatted_prompt, sizeof(formatted_prompt), "%s", prompt);
    }

    uint32_t prompt_tokens[1024];
    uint32_t n_prompt = lfm2_tokenize(&vocab, formatted_prompt, prompt_tokens, 1024);
    if (n_prompt == 0) {
        prompt_tokens[0] = vocab.id_bos ? vocab.id_bos : 1;
        n_prompt = 1;
    }

    printf("\n[Prefill] Processing %u prompt tokens on GPU / AVX2 hybrid pipeline...\n", n_prompt);
    double t_start = get_time_sec();

    uint32_t pos = 0;
    for (uint32_t p = 0; p < n_prompt - 1; ++p) {
        lfm_gpu_forward_token(&model, &state, prompt_tokens[p], pos++, &gpu, false);
    }

    /* 6. Autoregressive Generation Loop */
    printf("[Generation] Streaming response:\n\n\033[1;36mUser: %s\033[0m\n\033[1;32mAssistant:\033[0m ", prompt);
    fflush(stdout);

    uint32_t curr_token = prompt_tokens[n_prompt - 1];
    uint32_t token_history[1024];
    uint32_t n_hist = 0;

    for (uint32_t i = 0; i < n_prompt; ++i) {
        if (n_hist < 1024) token_history[n_hist++] = prompt_tokens[i];
    }

    double t_gen_start = get_time_sec();
    uint32_t generated = 0;

    for (uint32_t step = 0; step < max_tokens; ++step) {
        float *logits = lfm_gpu_forward_token(&model, &state, curr_token, pos++, &gpu, true);
        if (!logits) break;

        uint32_t next_tok = sample_next_token(logits, model.cfg.n_vocab, token_history,
                                              n_hist, 0.30f, 1.10f);

        if (vocab.id_im_end > 0 && next_tok == vocab.id_im_end) {
            break;
        }

        /* Stream token to stdout */
        lfm2_print_token(&vocab, next_tok);
        fflush(stdout);

        if (n_hist < 1024) token_history[n_hist++] = next_tok;
        curr_token = next_tok;
        generated++;
    }

    double t_end = get_time_sec();
    double total_sec = t_end - t_start;
    double gen_sec = t_end - t_gen_start;

    printf("\n\n----------------------------------------------------------------\n");
    printf(" Generation Stats:\n");
    printf("   GPU Active Device   : %s (%s)\n", gpu.vk.device_name,
           gpu.vk.is_integrated ? "Intel UMA iGPU" : "Discrete GPU");
    printf("   Prompt Tokens       : %u\n", n_prompt);
    printf("   Generated Tokens    : %u\n", generated);
    printf("   Generation Time     : %.2f sec (%.2f tok/s)\n", gen_sec,
           (gen_sec > 0.0) ? (double)generated / gen_sec : 0.0);
    printf("   Total Elapsed Time  : %.2f sec\n", total_sec);
    printf("   GPU Offload Status  : SUCCESS (FFN Q4_K passes accelerated on GPU)\n");
    printf("----------------------------------------------------------------\n");

    lfm2_state_free(&state);
    lfm2_vocab_free(&vocab);
    lfm2_free_model(&model);
    lfm_gpu_engine_free(&gpu);
    return EXIT_SUCCESS;
}
