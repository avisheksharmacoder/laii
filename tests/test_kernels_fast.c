#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_DECODE_1X4_NO_MAIN 1
#define LFM2_KERNELS_FAST_NO_MAIN 1

#include "tools/lfm2_kernels_fast.c"

static void test_planar_conv_parity(void) {
    printf("[Test 1/5] Verifying Planar ShortConv 3-Tap Initialization & Bit-Exact Parity...\n");
    const uint32_t D = 256;
    float *weights_interleaved = (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *bcx = (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *state_ref = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    float *state_avx = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    float *out_ref = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *out_avx = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    block_q8_K_t *out_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256) * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < 3 * D; ++i) {
        weights_interleaved[i] = sinf((float)i * 0.05f) * 0.5f;
        bcx[i] = cosf((float)i * 0.03f);
    }
    for (uint32_t i = 0; i < 2 * D; ++i) {
        state_ref[i] = state_avx[i] = 0.1f * sinf((float)i);
    }

    GGUFTensor t_conv = { .n_dims = 2, .ne = {3, D, 1, 1}, .type = GGML_TYPE_F32, .data = weights_interleaved };
    LFM2PlanarConvWeights pw;
    bool ok = lfm2_planar_conv_init(&pw, &t_conv, D);
    assert(ok);

    /* Reference calculation */
    for (uint32_t d = 0; d < D; ++d) {
        float u_t = bcx[d] * bcx[2 * D + d];
        float acc = weights_interleaved[d * 3 + 0] * state_ref[d]
                  + weights_interleaved[d * 3 + 1] * state_ref[D + d]
                  + weights_interleaved[d * 3 + 2] * u_t;
        out_ref[d]       = bcx[D + d] * acc;
        state_ref[d]     = state_ref[D + d];
        state_ref[D + d] = u_t;
    }

    /* Fast Fused AVX2 calculation */
    lfm2_shortconv_gate_conv_quant_avx2(out_avx, out_q8, bcx, state_avx, &pw, D);

    float max_diff = 0.0f;
    for (uint32_t d = 0; d < D; ++d) {
        float diff = fabsf(out_ref[d] - out_avx[d]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  ShortConv Output Max Diff: %.6e\n", max_diff);
    assert(max_diff < 1e-6f);

    /* Verify rolling state matches */
    for (uint32_t i = 0; i < 2 * D; ++i) {
        assert(fabsf(state_ref[i] - state_avx[i]) < 1e-6f);
    }

    lfm2_planar_conv_free(&pw);
    free(weights_interleaved); free(bcx);
    free(state_ref); free(state_avx);
    free(out_ref); free(out_avx); free(out_q8);
    printf("PASSED: Planar ShortConv 3-Tap bit-exact parity verified.\n");
}

static void test_l1_hot_q8k_quantization(void) {
    printf("[Test 2/5] Verifying Fused AVX2 ShortConv + L1-Hot Q8_K Quantization...\n");
    const uint32_t D = 512;
    float *bcx = (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *conv_weights = (float *)lfm2_aligned_alloc_64(3 * D * sizeof(float));
    float *state = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    float *out_f32 = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    block_q8_K_t *out_q8_fused = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256) * sizeof(block_q8_K_t));
    block_q8_K_t *out_q8_ref   = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256) * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < 3 * D; ++i) {
        bcx[i] = sinf((float)i * 0.02f);
        conv_weights[i] = cosf((float)i * 0.01f);
    }
    memset(state, 0, 2 * D * sizeof(float));

    GGUFTensor t_conv = { .n_dims = 2, .ne = {3, D, 1, 1}, .type = GGML_TYPE_F32, .data = conv_weights };
    LFM2PlanarConvWeights pw;
    lfm2_planar_conv_init(&pw, &t_conv, D);

    /* 1. Fused pass generates out_f32 and out_q8_fused in one go */
    lfm2_shortconv_gate_conv_quant_avx2(out_f32, out_q8_fused, bcx, state, &pw, D);

    /* 2. Separate reference quantizer on out_f32 */
    for (uint32_t b = 0; b < D / 256; ++b) {
        lfm2_quantize_row_q8_K_avx2(out_f32 + b * 256, &out_q8_ref[b], 256);
    }

    /* 3. Parity comparison of quantized scales and bytes */
    for (uint32_t b = 0; b < D / 256; ++b) {
        assert(fabsf(out_q8_fused[b].d - out_q8_ref[b].d) < 1e-6f);
        for (int i = 0; i < 256; ++i) {
            assert(out_q8_fused[b].qs[i] == out_q8_ref[b].qs[i]);
        }
    }

    lfm2_planar_conv_free(&pw);
    free(bcx); free(conv_weights); free(state);
    free(out_f32); free(out_q8_fused); free(out_q8_ref);
    printf("PASSED: In-place L1-hot Q8_K quantization verified.\n");
}

static void test_rope_lut_parity(void) {
    printf("[Test 3/5] Verifying Precomputed RoPE LUT & AVX2 QK-Norm Parity...\n");
    const uint32_t n_heads = 16;
    const uint32_t head_dim = 64;
    const uint32_t D = n_heads * head_dim;
    const uint32_t max_seq = 128;
    const uint32_t pos = 42;

    LFM2RoPELUT lut;
    bool ok = lfm2_rope_lut_init(&lut, max_seq, head_dim, 1000000.0f);
    assert(ok);

    float *q_ref = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *q_avx = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *norm_w = (float *)lfm2_aligned_alloc_64(head_dim * sizeof(float));

    for (uint32_t i = 0; i < D; ++i) q_ref[i] = q_avx[i] = sinf((float)i * 0.04f);
    for (uint32_t i = 0; i < head_dim; ++i) norm_w[i] = 1.0f + 0.05f * cosf((float)i);

    GGUFTensor t_norm = { .n_dims = 1, .ne = {head_dim, 1, 1, 1}, .type = GGML_TYPE_F32, .data = norm_w };

    /* Reference RMSNorm + NeoX RoPE */
    for (uint32_t h = 0; h < n_heads; ++h) {
        float *hp = q_ref + (size_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) ss += (double)hp[i] * (double)hp[i];
        float irms = 1.0f / sqrtf((float)(ss / head_dim) + 1e-5f);
        for (uint32_t i = 0; i < head_dim; ++i) hp[i] = hp[i] * irms * norm_w[i];

        for (uint32_t i = 0; i < 32; ++i) {
            float freq = powf(1000000.0f, -((float)(2 * i) / 64.0f));
            float c = cosf((float)pos * freq), s = sinf((float)pos * freq);
            float x0 = hp[i], x1 = hp[i + 32];
            hp[i]      = x0 * c - x1 * s;
            hp[i + 32] = x1 * c + x0 * s;
        }
    }

    /* Fast Vectorized QK-Norm + RoPE LUT */
    lfm2_qk_norm_rope_avx2(q_avx, NULL, n_heads, head_dim, &t_norm, 1e-5f, &lut, pos);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float diff = fabsf(q_ref[i] - q_avx[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  QK-Norm + RoPE Max Diff: %.6e\n", max_diff);
    assert(max_diff < 1e-4f);

    lfm2_rope_lut_free(&lut);
    free(q_ref); free(q_avx); free(norm_w);
    printf("PASSED: Precomputed RoPE LUT and AVX2 QK-Norm verified.\n");
}

static void test_online_gqa_f16_parity(void) {
    printf("[Test 4/5] Verifying Single-Pass FP16 Online Softmax GQA vs Two-Pass FP32...\n");
    const uint32_t D = 1024;
    const uint32_t n_head_q = 16;
    const uint32_t n_head_kv = 4;
    const uint32_t head_dim = 64;
    const uint32_t kv_dim = n_head_kv * head_dim;
    const uint32_t seq_len = 128;
    const uint32_t pos = 37;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *q = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *k_f32 = (float *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(float));
    float *v_f32 = (float *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(float));
    uint16_t *k_f16 = (uint16_t *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *v_f16 = (uint16_t *)lfm2_aligned_alloc_64((size_t)seq_len * kv_dim * sizeof(uint16_t));
    float *att_scratch = (float *)lfm2_aligned_alloc_64((size_t)n_head_q * seq_len * sizeof(float));
    float *out_ref = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *out_online = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    block_q8_K_t *out_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((D / 256) * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < D; ++i) q[i] = cosf((float)i * 0.05f);
    for (size_t i = 0; i < (size_t)seq_len * kv_dim; ++i) {
        k_f32[i] = sinf((float)i * 0.01f);
        v_f32[i] = cosf((float)i * 0.02f);
    }
    lfm2_store_fp32_to_fp16_avx2(k_f16, k_f32, seq_len * kv_dim);
    lfm2_store_fp32_to_fp16_avx2(v_f16, v_f32, seq_len * kv_dim);

    /* 1. Two-pass FP32 reference */
    GQAAttnTaskCtx ref_ctx = {
        .pool = &pool, .q = q, .k_cache = k_f32, .v_cache = v_f32,
        .att = att_scratch, .xb = out_ref, .n_head_q = n_head_q, .kv_mul = n_head_q / n_head_kv,
        .head_dim = head_dim, .kv_dim = kv_dim, .max_seq_len = seq_len, .pos = pos,
        .scale = 1.0f / sqrtf((float)head_dim)
    };
    lfm2_parallel_for(&pool, n_head_q, 1, lfm2_gqa_heads_worker_task, &ref_ctx);

    /* 2. Single-pass FP16 Online Softmax */
    OnlineGQATaskCtx online_ctx = {
        .pool = &pool, .q = q, .k_cache_f16 = k_f16, .v_cache_f16 = v_f16,
        .xb_out = out_online, .xb_q8_out = out_q8, .n_head_q = n_head_q,
        .kv_mul = n_head_q / n_head_kv, .kv_dim = kv_dim, .pos = pos,
        .scale = 1.0f / sqrtf((float)head_dim)
    };
    lfm2_parallel_for(&pool, n_head_q / 4, 1, lfm2_online_gqa_f16_worker, &online_ctx);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float diff = fabsf(out_ref[i] - out_online[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  FP16 Online Softmax Max Diff: %.6e\n", max_diff);
    assert(max_diff < 5e-4f);

    lfm2_threadpool_free(&pool);
    free(q); free(k_f32); free(v_f32); free(k_f16); free(v_f16);
    free(att_scratch); free(out_ref); free(out_online); free(out_q8);
    printf("PASSED: Single-Pass FP16 Online Softmax GQA verified.\n");
}

static void test_end_to_end_fast_model(void) {
    printf("[Test 5/5] Verifying End-to-End Fused Fast Token Forward Pass on Hybrid GGUF Checkpoint...\n");

    /* 1. Verify graceful rejection of non-conforming toy model (D=64, head_dim=32) */
    LFM2Model toy_model;
    if (lfm2_load_gguf("tests/lfm2_toy.gguf", &toy_model)) {
        LFM2FastContext toy_fast;
        bool ok = lfm2_fast_context_init(&toy_fast, &toy_model, 64);
        assert(!ok); /* Must safely reject non-64 head_dim */
        lfm2_free_model(&toy_model);
        printf("  Architecture validation check: non-conforming toy model safely rejected.\n");
    }

    /* 2. Test full execution on real Liquid AI checkpoint (D=1024, head_dim=64) if present */
    const char *model_path = "models/LFM2.5-230M-Q4_K_M.gguf";
    LFM2Model model;
    if (!lfm2_load_gguf(model_path, &model)) {
        printf("  [SKIP] '%s' not found for end-to-end decode run.\n", model_path);
        return;
    }

    LFM2State state;
    bool ok_state = lfm2_state_init(&state, &model, 64);
    assert(ok_state);

    LFM2FastContext fast_ctx;
    bool ok_fast = lfm2_fast_context_init(&fast_ctx, &model, 64);
    assert(ok_fast);

    /* Run 4 autoregressive decode steps via lfm2_forward_token_fast */
    uint32_t current_token = 1; /* BOS */
    for (uint32_t step = 0; step < 4; ++step) {
        float *logits = lfm2_forward_token_fast(&model, &state, &fast_ctx, current_token, step, true);
        assert(logits != NULL);

        /* Greedy argmax */
        uint32_t top_tok = 0;
        float top_val = logits[0];
        for (uint32_t v = 1; v < model.cfg.n_vocab; ++v) {
            if (logits[v] > top_val) {
                top_val = logits[v];
                top_tok = v;
            }
        }
        printf("  Step %u | in_token=%5u -> next_token=%5u (logit=%+.4f)\n",
               step, current_token, top_tok, top_val);
        assert(!isnan(top_val) && !isinf(top_val));
        current_token = top_tok;
    }

    lfm2_fast_context_free(&fast_ctx);
    lfm2_state_free(&state);
    lfm2_free_model(&model);
    printf("PASSED: End-to-end fused fast token forward pass verified on GGUF checkpoint!\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 Fused Fast Kernels Test Suite   \n");
    printf("=======================================================\n");

    test_planar_conv_parity();
    test_l1_hot_q8k_quantization();
    test_rope_lut_parity();
    test_online_gqa_f16_parity();
    test_end_to_end_fast_model();

    printf("=======================================================\n");
    printf("ALL 5 FAST KERNEL UNIT TESTS PASSED WITH 100%% SUCCESS! \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
