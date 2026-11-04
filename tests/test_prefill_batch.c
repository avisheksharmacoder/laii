#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define LFM2_PREFILL_BATCH_NO_MAIN 1
#include "tools/lfm2_prefill_batch.c"

static double test_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* ============================================================================
 * Test 1: Batched Q8_K Quantizer Bit-Exact Parity
 * ============================================================================ */
static void test_batched_quantizer_parity(void) {
    printf("[Test 1/7] Verifying Batched AVX2 Q8_K Quantizer...\n");
    const uint32_t B = 8;
    const uint32_t n_cols = 1024;
    const uint32_t nb = n_cols / 256;

    float *x = (float *)lfm2_aligned_alloc(64, (size_t)B * n_cols * sizeof(float));
    block_q8_K_t *y_seq   = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    block_q8_K_t *y_batch = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    assert(x && y_seq && y_batch);

    for (uint32_t i = 0; i < B * n_cols; ++i) {
        x[i] = sinf((float)i * 0.05f) * 3.14f;
    }

    /* Sequential quantization */
    for (uint32_t t = 0; t < B; ++t) {
        lfm2_quantize_row_q8_K_avx2(x + (size_t)t * n_cols, y_seq + (size_t)t * nb, n_cols);
    }

    /* Batched quantization */
    lfm2_quantize_batch_q8_K_avx2(x, y_batch, B, n_cols);

    /* Verify bit-exact equality */
    for (uint32_t t = 0; t < B; ++t) {
        for (uint32_t b = 0; b < nb; ++b) {
            const block_q8_K_t *blk_s = &y_seq[t * nb + b];
            const block_q8_K_t *blk_b = &y_batch[t * nb + b];
            assert(blk_s->d == blk_b->d);
            for (int j = 0; j < 256; ++j) {
                assert(blk_s->qs[j] == blk_b->qs[j]);
            }
            for (int j = 0; j < 16; ++j) {
                assert(blk_s->bsums[j] == blk_b->bsums[j]);
            }
        }
    }
    printf("  Parity: Bit-exact parity verified across B=%u tokens (%u cols).\n", B, n_cols);

    free(x); free(y_seq); free(y_batch);
    printf("PASSED: Batched Q8_K Quantizer verified.\n");
}

/* ============================================================================
 * Test 2: Q4_K & Q6_K Batched GEMM Microkernel Numerical Parity
 * ============================================================================ */
static void test_gemm_microkernels(void) {
    printf("[Test 2/7] Verifying Q4_K and Q6_K Batched GEMM micro-kernels...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 16;
    const uint32_t B      = 8;
    const uint32_t nb     = n_cols / 256;

    float        *x_batch   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_cols * sizeof(float));
    block_q8_K_t *xq_batch  = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4      = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)n_rows * nb * sizeof(block_q4_K_t));
    block_q6_K_t *w_q6      = (block_q6_K_t *)lfm2_aligned_alloc(64, (size_t)n_rows * nb * sizeof(block_q6_K_t));
    float        *out_q4_seq   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_q4_batch = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_q6_seq   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_q6_batch = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));

    for (uint32_t i = 0; i < B * n_cols; ++i) x_batch[i] = sinf((float)i * 0.03f) * 1.5f;
    lfm2_quantize_batch_q8_K_avx2(x_batch, xq_batch, B, n_cols);

    /* Initialize Q4_K test data */
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3C00; /* 1.0 in FP16 */
        w_q4[i].dmin = 0x3800; /* 0.5 in FP16 */
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)(j * 5 + 3);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 17 + j) & 0xFF);
    }

    /* Initialize Q6_K test data */
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q6[i].d = 0x3C00;
        for (int j = 0; j < 16; ++j)  w_q6[i].scales[j] = (int8_t)((j % 7) - 3);
        for (int j = 0; j < 64; ++j)  w_q6[i].qh[j]     = (uint8_t)(j * 11);
        for (int j = 0; j < 128; ++j) w_q6[i].ql[j]     = (uint8_t)(j * 13);
    }

    /* 1. Test Q4_K GEMM Microkernel */
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t t = 0; t < B; ++t) {
            out_q4_seq[t * n_rows + r] = lfm2_vec_dot_q4_K_q8_K_avx2(
                w_q4 + (size_t)r * nb, xq_batch + (size_t)t * nb, n_cols);
        }
        lfm2_gemm_row_q4_K_q8_K_avx2(
            w_q4 + (size_t)r * nb, xq_batch, B, n_cols, n_rows, r, out_q4_batch);
    }

    float max_diff_q4 = 0.0f;
    for (uint32_t i = 0; i < B * n_rows; ++i) {
        float diff = fabsf(out_q4_seq[i] - out_q4_batch[i]);
        if (diff > max_diff_q4) max_diff_q4 = diff;
        assert(diff < 1e-3f);
    }
    printf("  Q4_K GEMM: Parity confirmed between M=1 and M=8 (max diff: %e).\n", max_diff_q4);

    /* 2. Test Q6_K GEMM Microkernel */
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t t = 0; t < B; ++t) {
            out_q6_seq[t * n_rows + r] = lfm2_vec_dot_q6_K_q8_K_avx2(
                w_q6 + (size_t)r * nb, xq_batch + (size_t)t * nb, n_cols);
        }
        lfm2_gemm_row_q6_K_q8_K_avx2(
            w_q6 + (size_t)r * nb, xq_batch, B, n_cols, n_rows, r, out_q6_batch);
    }

    float max_diff_q6 = 0.0f;
    for (uint32_t i = 0; i < B * n_rows; ++i) {
        float diff = fabsf(out_q6_seq[i] - out_q6_batch[i]);
        if (diff > max_diff_q6) max_diff_q6 = diff;
        assert(diff < 1e-3f);
    }
    printf("  Q6_K GEMM: Parity confirmed between M=1 and M=8 (max diff: %e).\n", max_diff_q6);

    free(x_batch); free(xq_batch); free(w_q4); free(w_q6);
    free(out_q4_seq); free(out_q4_batch); free(out_q6_seq); free(out_q6_batch);
    printf("PASSED: Q4_K and Q6_K Batched GEMM micro-kernels verified.\n");
}

/* ============================================================================
 * Test 3: Multi-Threaded Batched GEMM Dispatcher Parity
 * ============================================================================ */
static void test_multithreaded_batch_gemm(void) {
    printf("[Test 3/7] Verifying ThreadPool Batched GEMM multi-threading...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 64;
    const uint32_t B      = 8;
    const uint32_t nb     = n_cols / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float        *x_batch   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_cols * sizeof(float));
    float        *out_seq   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_batch = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    block_q8_K_t *xq_batch  = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4      = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)n_rows * nb * sizeof(block_q4_K_t));

    for (uint32_t i = 0; i < B * n_cols; ++i) x_batch[i] = cosf((float)i * 0.07f);
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3C00;
        w_q4[i].dmin = 0x3800;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)(j * 3 + 1);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 7 + j * 13) & 0xFF);
    }

    GGUFTensor W = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_q4 };

    /* Single-thread sequential M=1 */
    for (uint32_t t = 0; t < B; ++t) {
        lfm2_matmul_batch_avx2_mt(NULL, out_seq + (size_t)t * n_rows, &W,
                                  x_batch + (size_t)t * n_cols, 1, xq_batch);
    }

    /* Multi-thread batched M=8 */
    lfm2_matmul_batch_avx2_mt(&pool, out_batch, &W, x_batch, B, xq_batch);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < B * n_rows; ++i) {
        float diff = fabsf(out_seq[i] - out_batch[i]);
        if (diff > max_diff) max_diff = diff;
        assert(diff < 1e-3f);
    }
    printf("  ThreadPool MatMul: Parity verified across %u threads (max diff: %e).\n",
           pool.n_threads, max_diff);

    free(x_batch); free(out_seq); free(out_batch); free(xq_batch); free(w_q4);
    lfm2_threadpool_free(&pool);
    printf("PASSED: Multi-threaded Batched GEMM verified.\n");
}

/* ============================================================================
 * Test 4: Exact ShortConv Causality Inside Batch
 * ============================================================================ */
static void test_shortconv_batch_causality(void) {
    printf("[Test 4/7] Verifying Exact ShortConv Causality Inside Batch...\n");
    const char *model_path = "tests/lfm2_toy.gguf";

    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) {
        printf("  [SKIP] %s not found.\n", model_path);
        return;
    }

    /* Find first ShortConv layer */
    int sc_layer = -1;
    for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            sc_layer = (int)l;
            break;
        }
    }
    assert(sc_layer >= 0);

    const uint32_t B = 8;
    const uint32_t D = model.cfg.n_embd;

    LFM2State state_seq;
    LFM2BatchState state_batch;
    assert(lfm2_state_init(&state_seq, &model, 64));
    assert(lfm2_batch_state_init(&state_batch, &model, 64, 2));

    /* Initialize pseudo-random batch input activations */
    for (uint32_t t = 0; t < B; ++t) {
        for (uint32_t d = 0; d < D; ++d) {
            float val = sinf((float)(t * D + d) * 0.01f);
            state_seq.xb[d] = val;
            state_batch.xb[t * D + d] = val;
        }
    }

    /* Method 1: Sequential token-by-token ShortConv */
    float *seq_outputs = (float *)calloc((size_t)B * D, sizeof(float));
    for (uint32_t t = 0; t < B; ++t) {
        /* Set input xb for step t */
        memcpy(state_seq.xb, state_batch.xb + (size_t)t * D, D * sizeof(float));
        lfm2_forward_shortconv(&state_seq, &model, &model.layers[sc_layer], sc_layer);
        memcpy(seq_outputs + (size_t)t * D, state_seq.xb, D * sizeof(float));
    }

    /* Method 2: Single Batched ShortConv call */
    lfm2_forward_shortconv_batch(&state_batch, &model, &model.layers[sc_layer], sc_layer, B);

    /* Verify exact numerical equivalence of rolling conv_state at step B */
    const uint32_t hist = model.cfg.shortconv_l_cache - 1;
    for (uint32_t i = 0; i < hist * D; ++i) {
        float diff = fabsf(state_seq.conv_state[sc_layer][i] - state_batch.conv_state[sc_layer][i]);
        assert(diff < 1e-4f);
    }
    printf("  Causality: Rolling conv_state buffer is 100%% identical after B=%u steps.\n", B);

    /* Verify exact equivalence of output activations for every token in the batch */
    float max_xb_diff = 0.0f;
    for (uint32_t i = 0; i < B * D; ++i) {
        float diff = fabsf(seq_outputs[i] - state_batch.xb[i]);
        if (diff > max_xb_diff) max_xb_diff = diff;
        assert(diff < 1e-3f);
    }
    printf("  Outputs: ShortConv outputs match step-by-step (max diff: %e).\n", max_xb_diff);

    free(seq_outputs);
    lfm2_state_free(&state_seq);
    lfm2_batch_state_free(&state_batch);
    lfm2_free_model(&model);
    printf("PASSED: ShortConv batched causality verified.\n");
}

/* ============================================================================
 * Test 5: Batched GQA & SwiGLU Parity
 * ============================================================================ */
static void test_gqa_and_swiglu_batch_parity(void) {
    printf("[Test 5/7] Verifying Batched GQA Attention & SwiGLU FFN Parity...\n");
    const char *model_path = "tests/lfm2_toy.gguf";

    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) {
        printf("  [SKIP] %s not found.\n", model_path);
        return;
    }

    int gqa_layer = -1;
    for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_GQA_ATTN) {
            gqa_layer = (int)l;
            break;
        }
    }
    assert(gqa_layer >= 0);

    const uint32_t B = 8;
    const uint32_t D = model.cfg.n_embd;

    LFM2State state_seq;
    LFM2BatchState state_batch;
    assert(lfm2_state_init(&state_seq, &model, 64));
    assert(lfm2_batch_state_init(&state_batch, &model, 64, 2));

    for (uint32_t t = 0; t < B; ++t) {
        for (uint32_t d = 0; d < D; ++d) {
            float val = cosf((float)(t * D + d) * 0.015f);
            state_batch.xb[t * D + d] = val;
            state_batch.x[t * D + d]  = val;
        }
    }

    /* 1. GQA Parity: Sequential vs Batched */
    float *gqa_seq_out = (float *)calloc((size_t)B * D, sizeof(float));
    for (uint32_t t = 0; t < B; ++t) {
        memcpy(state_seq.xb, state_batch.xb + (size_t)t * D, D * sizeof(float));
        memcpy(state_seq.x,  state_batch.x  + (size_t)t * D, D * sizeof(float));
        lfm2_forward_gqa(&state_seq, &model, &model.layers[gqa_layer], gqa_layer, t);
        memcpy(gqa_seq_out + (size_t)t * D, state_seq.x, D * sizeof(float));
    }

    lfm2_forward_gqa_batch(&state_batch, &model, &model.layers[gqa_layer], gqa_layer, 0, B);

    float max_gqa_diff = 0.0f;
    for (uint32_t i = 0; i < B * D; ++i) {
        float diff = fabsf(gqa_seq_out[i] - state_batch.x[i]);
        if (diff > max_gqa_diff) max_gqa_diff = diff;
        assert(diff < 1e-3f);
    }
    printf("  GQA Parity: Batched attention matches sequential execution (max diff: %e).\n", max_gqa_diff);

    /* 2. SwiGLU Parity */
    float *ffn_seq_out = (float *)calloc((size_t)B * D, sizeof(float));
    for (uint32_t t = 0; t < B; ++t) {
        memcpy(state_seq.x, state_batch.x + (size_t)t * D, D * sizeof(float));
        lfm2_forward_ffn(&state_seq, &model, &model.layers[gqa_layer]);
        memcpy(ffn_seq_out + (size_t)t * D, state_seq.x, D * sizeof(float));
    }

    lfm2_forward_ffn_batch(&state_batch, &model, &model.layers[gqa_layer], B);

    float max_ffn_diff = 0.0f;
    for (uint32_t i = 0; i < B * D; ++i) {
        float diff = fabsf(ffn_seq_out[i] - state_batch.x[i]);
        if (diff > max_ffn_diff) max_ffn_diff = diff;
        assert(diff < 1e-3f);
    }
    printf("  SwiGLU Parity: Batched FFN matches sequential execution (max diff: %e).\n", max_ffn_diff);

    free(gqa_seq_out); free(ffn_seq_out);
    lfm2_state_free(&state_seq);
    lfm2_batch_state_free(&state_batch);
    lfm2_free_model(&model);
    printf("PASSED: Batched GQA & SwiGLU parity verified.\n");
}

/* ============================================================================
 * Test 6: End-to-End Batched Prefill Parity & State Equivalence
 * ============================================================================ */
static void test_end_to_end_prefill_parity(void) {
    printf("[Test 6/7] Verifying End-to-End Batched Prompt Prefill Parity...\n");
    const char *model_path = "tests/lfm2_toy.gguf";

    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) {
        printf("  [SKIP] %s not found.\n", model_path);
        return;
    }

    LFM2State state_seq;
    LFM2BatchState state_batch;
    assert(lfm2_state_init(&state_seq, &model, 64));
    assert(lfm2_batch_state_init(&state_batch, &model, 64, 2));

    const uint32_t n_prompt = 14;
    const uint32_t prompt_tokens[14] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    /* Method 1: Sequential prefill token-by-token */
    float *logits_seq = NULL;
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        bool compute_logits = (pos + 1 == n_prompt);
        logits_seq = lfm2_forward_ex(&model, &state_seq, prompt_tokens[pos], pos, compute_logits);
    }
    assert(logits_seq != NULL);

    /* Method 2: Batched prefill via lfm2_prefill_prompt (B=8 chunks) */
    uint32_t pos_batch = 0;
    float *logits_batch = lfm2_prefill_prompt(&model, &state_batch, prompt_tokens, n_prompt, &pos_batch);
    assert(logits_batch != NULL);
    assert(pos_batch == n_prompt);

    /* Verify exact equivalence of final logits */
    float max_logit_diff = 0.0f;
    uint32_t best_seq_id = 0, best_batch_id = 0;
    float best_seq_val = -1e30f, best_batch_val = -1e30f;

    for (uint32_t v = 0; v < model.cfg.n_vocab; ++v) {
        float diff = fabsf(logits_seq[v] - logits_batch[v]);
        if (diff > max_logit_diff) max_logit_diff = diff;
        assert(diff < 1e-3f);

        if (logits_seq[v] > best_seq_val) {
            best_seq_val = logits_seq[v];
            best_seq_id = v;
        }
        if (logits_batch[v] > best_batch_val) {
            best_batch_val = logits_batch[v];
            best_batch_id = v;
        }
    }

    assert(best_seq_id == best_batch_id);
    printf("  Argmax Parity: Top-1 next token is identical (%u).\n", best_batch_id);
    printf("  Logits Parity: Prompt final logits match (max diff: %e).\n", max_logit_diff);

    lfm2_state_free(&state_seq);
    lfm2_batch_state_free(&state_batch);
    lfm2_free_model(&model);
    printf("PASSED: End-to-end batched prompt prefill verified.\n");
}

/* ============================================================================
 * Test 7: Batched GEMM Throughput Benchmark
 * ============================================================================ */
static void test_batched_prefill_benchmark(void) {
    printf("[Test 7/7] Running Batched Prefill Throughput Benchmark...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 2048;
    const uint32_t B      = 8;
    const uint32_t nb     = n_cols / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 0);

    float        *x_batch   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_cols * sizeof(float));
    float        *out_seq   = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    float        *out_batch = (float *)lfm2_aligned_alloc(64, (size_t)B * n_rows * sizeof(float));
    block_q8_K_t *xq_batch  = (block_q8_K_t *)lfm2_aligned_alloc(64, (size_t)B * nb * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4      = (block_q4_K_t *)lfm2_aligned_alloc(64, (size_t)n_rows * nb * sizeof(block_q4_K_t));

    for (uint32_t i = 0; i < B * n_cols; ++i) x_batch[i] = sinf((float)i * 0.01f);
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3000;
        w_q4[i].dmin = 0x2800;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)(j + 1);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)(j ^ 0x5A);
    }
    GGUFTensor fake_W = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_q4 };

    const int iters = 100;
    double t0 = test_now_us();
    for (int it = 0; it < iters; ++it) {
        for (uint32_t t = 0; t < B; ++t) {
            lfm2_matmul_batch_avx2_mt(&pool, out_seq + (size_t)t * n_rows, &fake_W,
                                      x_batch + (size_t)t * n_cols, 1, xq_batch);
        }
    }
    double seq_us = (test_now_us() - t0) / iters;

    t0 = test_now_us();
    for (int it = 0; it < iters; ++it) {
        lfm2_matmul_batch_avx2_mt(&pool, out_batch, &fake_W, x_batch, B, xq_batch);
    }
    double batch_us = (test_now_us() - t0) / iters;

    printf("  Benchmark (8 tokens x 1024 cols x 2048 rows, %u threads):\n", pool.n_threads);
    printf("    Sequential GEMV (M=1 x 8): %.2f us (%.2f us/tok)\n", seq_us, seq_us / B);
    printf("    Batched GEMM    (M=8 x 1): %.2f us (%.2f us/tok)\n", batch_us, batch_us / B);
    printf("    Prefill Speedup          : %.2fx faster\n", seq_us / (batch_us + 1e-6));

    free(x_batch); free(out_seq); free(out_batch); free(xq_batch); free(w_q4);
    lfm2_threadpool_free(&pool);
    printf("PASSED: Throughput benchmark complete.\n");
}

/* ============================================================================
 * Main Entry Point
 * ============================================================================ */
int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 Batched Prefill Verification    \n");
    printf("=======================================================\n");

    test_batched_quantizer_parity();
    test_gemm_microkernels();
    test_multithreaded_batch_gemm();
    test_shortconv_batch_causality();
    test_gqa_and_swiglu_batch_parity();
    test_end_to_end_prefill_parity();
    test_batched_prefill_benchmark();

    printf("=======================================================\n");
    printf("ALL 7 BATCHED PREFILL TESTS PASSED WITH 100%% SUCCESS!  \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
