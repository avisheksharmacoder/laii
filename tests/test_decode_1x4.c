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

#include "tools/lfm2_decode_1x4.c"

static void test_hsum4_epi32_avx2(void) {
    printf("[Test 1/7] Verifying 4-Row SIMD Horizontal Reduction (hsum4_epi32_avx2)...\n");
    int32_t a0[8] = { 1,  2,  3,  4,  5,  6,  7,  8};
    int32_t a1[8] = {-1, -2, -3, -4, -5, -6, -7, -8};
    int32_t a2[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int32_t a3[8] = { 0, 15, -5, 25, -35, 45, -55, 65};

    int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    for (int i = 0; i < 8; ++i) {
        sum0 += a0[i];
        sum1 += a1[i];
        sum2 += a2[i];
        sum3 += a3[i];
    }

    __m256i r0 = _mm256_loadu_si256((const __m256i *)a0);
    __m256i r1 = _mm256_loadu_si256((const __m256i *)a1);
    __m256i r2 = _mm256_loadu_si256((const __m256i *)a2);
    __m256i r3 = _mm256_loadu_si256((const __m256i *)a3);

    __m128i vsum = hsum4_epi32_avx2(r0, r1, r2, r3);

    int32_t res[4];
    _mm_storeu_si128((__m128i *)res, vsum);

    assert(res[0] == sum0);
    assert(res[1] == sum1);
    assert(res[2] == sum2);
    assert(res[3] == sum3);

    /* Also verify with arbitrary values */
    for (int trial = 0; trial < 100; ++trial) {
        int32_t s[4] = {0};
        for (int i = 0; i < 8; ++i) {
            a0[i] = (rand() % 1000) - 500; s[0] += a0[i];
            a1[i] = (rand() % 1000) - 500; s[1] += a1[i];
            a2[i] = (rand() % 1000) - 500; s[2] += a2[i];
            a3[i] = (rand() % 1000) - 500; s[3] += a3[i];
        }
        r0 = _mm256_loadu_si256((const __m256i *)a0);
        r1 = _mm256_loadu_si256((const __m256i *)a1);
        r2 = _mm256_loadu_si256((const __m256i *)a2);
        r3 = _mm256_loadu_si256((const __m256i *)a3);
        _mm_storeu_si128((__m128i *)res, hsum4_epi32_avx2(r0, r1, r2, r3));
        assert(res[0] == s[0] && res[1] == s[1] && res[2] == s[2] && res[3] == s[3]);
    }

    printf("  Parity: Bit-exact 4-way horizontal reduction across all lanes.\n");
    printf("PASSED: 4-Way SIMD Horizontal Reduction verified.\n");
}

static void test_fused_rmsnorm_quantize(void) {
    printf("[Test 2/7] Verifying Fused RMSNorm + Q8_K Quantizer...\n");
    const uint32_t dim = 1024;
    const uint32_t nb  = dim / 256;

    float *x      = (float *)lfm2_aligned_alloc_64(dim * sizeof(float));
    float *norm_w = (float *)lfm2_aligned_alloc_64(dim * sizeof(float));
    float *xb_ref = (float *)lfm2_aligned_alloc_64(dim * sizeof(float));
    float *xb_fus = (float *)lfm2_aligned_alloc_64(dim * sizeof(float));
    block_q8_K_t *xq_ref = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));
    block_q8_K_t *xq_fus = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < dim; ++i) {
        x[i]      = sinf((float)i * 0.023f);
        norm_w[i] = 0.9f + 0.1f * cosf((float)i * 0.05f);
    }
    GGUFTensor w_tensor = { .n_dims = 1, .ne = {dim, 1, 1, 1}, .type = GGML_TYPE_F32, .data = norm_w };

    /* 1. Unfused Reference: separate rmsnorm + separate quantize */
    double ss = 0.0;
    for (uint32_t i = 0; i < dim; ++i) ss += (double)x[i] * (double)x[i];
    float inv_rms = 1.0f / sqrtf((float)(ss / dim) + 1e-5f);
    for (uint32_t i = 0; i < dim; ++i) xb_ref[i] = x[i] * inv_rms * norm_w[i];
    lfm2_quantize_row_q8_K_avx2(xb_ref, xq_ref, dim);

    /* 2. Fused Pass */
    lfm2_rmsnorm_quantize_q8_K_avx2(xb_fus, xq_fus, x, &w_tensor, dim, 1e-5f);

    float max_diff_xb = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = fabsf(xb_ref[i] - xb_fus[i]);
        if (d > max_diff_xb) max_diff_xb = d;
    }
    assert(max_diff_xb < 1e-5f);

    float max_diff_d = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        float d = fabsf(xq_ref[b].d - xq_fus[b].d);
        if (d > max_diff_d) max_diff_d = d;
        for (int j = 0; j < 16; ++j) {
            assert(abs(xq_ref[b].bsums[j] - xq_fus[b].bsums[j]) <= 1);
        }
    }
    assert(max_diff_d < 1e-5f);

    printf("  Normalized Floats Max Diff: %.6e\n", max_diff_xb);
    printf("  Quantized Scale Max Diff  : %.6e\n", max_diff_d);
    printf("PASSED: Fused RMSNorm + Q8_K Quantizer verified.\n");

    free(x); free(norm_w); free(xb_ref); free(xb_fus); free(xq_ref); free(xq_fus);
}

static void test_1x4_microkernels(void) {
    printf("[Test 3/7] Verifying 1x4 Row-Unrolled Q4_K & Q6_K Micro-Kernels...\n");
    const uint32_t n_cols = 1024;
    const uint32_t nb = n_cols / 256;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *xq = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));
    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.017f);
    lfm2_quantize_row_q8_K_avx2(x, xq, n_cols);

    /* Test Q4_K 1x4 */
    block_q4_K_t *w_q4 = (block_q4_K_t *)lfm2_aligned_alloc_64(4 * nb * sizeof(block_q4_K_t));
    for (size_t i = 0; i < 4 * nb; ++i) {
        w_q4[i].d    = 0x2A00;
        w_q4[i].dmin = 0x2400;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i * 7 + j * 3) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 11 + j * 17) & 0xFF);
    }

    float out_q4_ref[4] = {0};
    for (uint32_t r = 0; r < 4; ++r) {
        out_q4_ref[r] = lfm2_vec_dot_q4_K_q8_K_avx2(w_q4 + (size_t)r * nb, xq, n_cols);
    }
    float out_q4_1x4[4] = {0};
    lfm2_vec_dot_q4_K_q8_K_1x4_avx2(w_q4, nb, xq, n_cols, out_q4_1x4);

    float max_diff_q4 = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float diff = fabsf(out_q4_ref[k] - out_q4_1x4[k]);
        if (diff > max_diff_q4) max_diff_q4 = diff;
    }
    printf("  Q4_K 1x4 Max Diff: %.6e\n", max_diff_q4);
    assert(max_diff_q4 < 1e-4f);

    /* Test Q6_K 1x4 */
    block_q6_K_t *w_q6 = (block_q6_K_t *)lfm2_aligned_alloc_64(4 * nb * sizeof(block_q6_K_t));
    for (size_t i = 0; i < 4 * nb; ++i) {
        w_q6[i].d = 0x2B00;
        for (int j = 0; j < 16; ++j)  w_q6[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_q6[i].ql[j]     = (uint8_t)((i * 19 + j * 11) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_q6[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    float out_q6_ref[4] = {0};
    for (uint32_t r = 0; r < 4; ++r) {
        out_q6_ref[r] = lfm2_vec_dot_q6_K_q8_K_avx2(w_q6 + (size_t)r * nb, xq, n_cols);
    }
    float out_q6_1x4[4] = {0};
    lfm2_vec_dot_q6_K_q8_K_1x4_avx2(w_q6, nb, xq, n_cols, out_q6_1x4);

    float max_diff_q6 = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float diff = fabsf(out_q6_ref[k] - out_q6_1x4[k]);
        if (diff > max_diff_q6) max_diff_q6 = diff;
    }
    printf("  Q6_K 1x4 Max Diff: %.6e\n", max_diff_q6);
    assert(max_diff_q6 < 1e-4f);

    printf("PASSED: 1x4 Row-Unrolled Q4_K & Q6_K Micro-Kernels verified.\n");
    free(x); free(xq); free(w_q4); free(w_q6);
}

static void test_matmul_1x4_prequant(void) {
    printf("[Test 4/7] Verifying Pre-Quantized 1x4 Multi-Threaded MatMul...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 128;
    const uint32_t nb     = n_cols / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *x   = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    float *out_ref = (float *)lfm2_aligned_alloc_64(n_rows * sizeof(float));
    float *out_1x4 = (float *)lfm2_aligned_alloc_64(n_rows * sizeof(float));
    block_q8_K_t *xq = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = cosf((float)i * 0.031f);
    lfm2_quantize_row_q8_K_avx2(x, xq, n_cols);

    block_q4_K_t *w_q4 = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb * sizeof(block_q4_K_t));
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x2C00;
        w_q4[i].dmin = 0x2200;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i + j * 2) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 3 + j * 5) & 0xFF);
    }
    GGUFTensor t_W = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_q4 };

    lfm2_matmul_vec_avx2_mt(&pool, out_ref, w_q4, GGML_TYPE_Q4_K, n_cols, n_rows, x, xq);
    lfm2_matmul_1x4_prequant_mt(&pool, out_1x4, &t_W, x, xq);

    float max_diff = 0.0f;
    for (uint32_t r = 0; r < n_rows; ++r) {
        float diff = fabsf(out_ref[r] - out_1x4[r]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  MatMul 1x4 Pre-Quant Max Diff: %.6e\n", max_diff);
    assert(max_diff < 1e-4f);

    lfm2_threadpool_free(&pool);
    free(x); free(out_ref); free(out_1x4); free(xq); free(w_q4);
    printf("PASSED: Pre-Quantized 1x4 Multi-Threaded MatMul verified.\n");
}

static void test_fused_qkv_1x4(void) {
    printf("[Test 5/7] Verifying Fused Single-Barrier QKV Projection (lfm2_gqa_qkv_fused_1x4_mt)...\n");
    const uint32_t D      = 1024;
    const uint32_t kv_dim = 256;
    const uint32_t nb     = D / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *x   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *q_ref = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *k_ref = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    float *v_ref = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));

    float *q_fus = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *k_fus = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    float *v_fus = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));

    block_q8_K_t *xq = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));
    for (uint32_t i = 0; i < D; ++i) x[i] = sinf((float)i * 0.013f);
    lfm2_quantize_row_q8_K_avx2(x, xq, D);

    block_q4_K_t *wq = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)D * nb * sizeof(block_q4_K_t));
    block_q4_K_t *wk = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)kv_dim * nb * sizeof(block_q4_K_t));
    block_q4_K_t *wv = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)kv_dim * nb * sizeof(block_q4_K_t));

    for (size_t i = 0; i < (size_t)D * nb; ++i) {
        wq[i].d = 0x2A00; wq[i].dmin = 0x2200;
        for (int j = 0; j < 12; ++j) wq[i].scales[j] = (uint8_t)((i + j * 3) & 0x3F);
        for (int j = 0; j < 128; ++j) wq[i].qs[j] = (uint8_t)((i * 5 + j * 7) & 0xFF);
    }
    for (size_t i = 0; i < (size_t)kv_dim * nb; ++i) {
        wk[i].d = 0x2B00; wk[i].dmin = 0x2300;
        for (int j = 0; j < 12; ++j) wk[i].scales[j] = (uint8_t)((i * 2 + j * 5) & 0x3F);
        for (int j = 0; j < 128; ++j) wk[i].qs[j] = (uint8_t)((i * 7 + j * 11) & 0xFF);

        wv[i].d = 0x2C00; wv[i].dmin = 0x2400;
        for (int j = 0; j < 12; ++j) wv[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
        for (int j = 0; j < 128; ++j) wv[i].qs[j] = (uint8_t)((i * 11 + j * 13) & 0xFF);
    }

    GGUFTensor t_wq = { .n_dims = 2, .ne = {D, D, 1, 1},      .type = GGML_TYPE_Q4_K, .data = wq };
    GGUFTensor t_wk = { .n_dims = 2, .ne = {D, kv_dim, 1, 1}, .type = GGML_TYPE_Q4_K, .data = wk };
    GGUFTensor t_wv = { .n_dims = 2, .ne = {D, kv_dim, 1, 1}, .type = GGML_TYPE_Q4_K, .data = wv };

    /* Unfused 1x1 separate dispatches */
    lfm2_matmul_vec_avx2_mt(&pool, q_ref, wq, GGML_TYPE_Q4_K, D, D, x, xq);
    lfm2_matmul_vec_avx2_mt(&pool, k_ref, wk, GGML_TYPE_Q4_K, D, kv_dim, x, xq);
    lfm2_matmul_vec_avx2_mt(&pool, v_ref, wv, GGML_TYPE_Q4_K, D, kv_dim, x, xq);

    /* Fused single-barrier dispatch */
    lfm2_gqa_qkv_fused_1x4_mt(&pool, q_fus, k_fus, v_fus, &t_wq, &t_wk, &t_wv, x, xq);

    float max_q = 0.0f, max_k = 0.0f, max_v = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float d = fabsf(q_ref[i] - q_fus[i]);
        if (d > max_q) max_q = d;
    }
    for (uint32_t i = 0; i < kv_dim; ++i) {
        float d = fabsf(k_ref[i] - k_fus[i]);
        if (d > max_k) max_k = d;
        d = fabsf(v_ref[i] - v_fus[i]);
        if (d > max_v) max_v = d;
    }
    printf("  Fused Q Diff: %.6e | K Diff: %.6e | V Diff: %.6e\n", max_q, max_k, max_v);
    assert(max_q < 1e-4f && max_k < 1e-4f && max_v < 1e-4f);

    lfm2_threadpool_free(&pool);
    free(x); free(q_ref); free(k_ref); free(v_ref);
    free(q_fus); free(k_fus); free(v_fus);
    free(xq); free(wq); free(wk); free(wv);
    printf("PASSED: Fused Single-Barrier QKV Projection verified.\n");
}

static void test_fused_swiglu_1x4(void) {
    printf("[Test 6/7] Verifying Fused SwiGLU + In-Place Q8_K Quantization (lfm2_ffn_swiglu_fused_1x4_mt)...\n");
    const uint32_t D    = 1024;
    const uint32_t n_ff = 2048;
    const uint32_t nb_d = D / 256;
    const uint32_t nb_ff = n_ff / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *x   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *gate_ref = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    float *up_ref   = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    float *hb_ref   = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    float *hb_fus   = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));

    block_q8_K_t *xq = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_d * sizeof(block_q8_K_t));
    block_q8_K_t *hb_q8_ref = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_ff * sizeof(block_q8_K_t));
    block_q8_K_t *hb_q8_fus = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_ff * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < D; ++i) x[i] = sinf((float)i * 0.021f);
    lfm2_quantize_row_q8_K_avx2(x, xq, D);

    block_q4_K_t *w_gate = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_up   = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));

    for (size_t i = 0; i < (size_t)n_ff * nb_d; ++i) {
        w_gate[i].d = 0x2A00; w_gate[i].dmin = 0x2200;
        w_up[i].d   = 0x2B00; w_up[i].dmin   = 0x2300;
        for (int j = 0; j < 12; ++j) {
            w_gate[i].scales[j] = (uint8_t)((i + j * 3) & 0x3F);
            w_up[i].scales[j]   = (uint8_t)((i * 2 + j * 5) & 0x3F);
        }
        for (int j = 0; j < 128; ++j) {
            w_gate[i].qs[j] = (uint8_t)((i * 5 + j * 7) & 0xFF);
            w_up[i].qs[j]   = (uint8_t)((i * 7 + j * 11) & 0xFF);
        }
    }

    GGUFTensor t_gate = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_gate };
    GGUFTensor t_up   = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_up   };

    /* Unfused reference */
    lfm2_matmul_vec_avx2_mt(&pool, gate_ref, w_gate, GGML_TYPE_Q4_K, D, n_ff, x, xq);
    lfm2_matmul_vec_avx2_mt(&pool, up_ref,   w_up,   GGML_TYPE_Q4_K, D, n_ff, x, xq);
    for (uint32_t i = 0; i < n_ff; ++i) {
        const float g = gate_ref[i];
        const float silu = g / (1.0f + expf(-g));
        hb_ref[i] = silu * up_ref[i];
    }
    lfm2_quantize_row_q8_K_avx2(hb_ref, hb_q8_ref, n_ff);

    /* Fused pass: gate + up + swiglu + in-place Q8_K quantization */
    lfm2_ffn_swiglu_fused_1x4_mt(&pool, hb_fus, hb_q8_fus, &t_gate, &t_up, x, xq);

    float max_diff_hb = 0.0f;
    for (uint32_t i = 0; i < n_ff; ++i) {
        float d = fabsf(hb_ref[i] - hb_fus[i]);
        if (d > max_diff_hb) max_diff_hb = d;
    }
    float max_diff_q8_d = 0.0f;
    for (uint32_t b = 0; b < nb_ff; ++b) {
        float d = fabsf(hb_q8_ref[b].d - hb_q8_fus[b].d);
        if (d > max_diff_q8_d) max_diff_q8_d = d;
    }

    printf("  SwiGLU Output Diff: %.6e | In-Place Q8_K Scale Diff: %.6e\n", max_diff_hb, max_diff_q8_d);
    assert(max_diff_hb < 1e-4f);
    assert(max_diff_q8_d < 1e-4f);

    lfm2_threadpool_free(&pool);
    free(x); free(gate_ref); free(up_ref); free(hb_ref); free(hb_fus);
    free(xq); free(hb_q8_ref); free(hb_q8_fus); free(w_gate); free(w_up);
    printf("PASSED: Fused SwiGLU + In-Place Q8_K Quantization verified.\n");
}

static void test_full_ffn_layer_and_benchmark(void) {
    printf("[Test 7/7] Verifying Full FFN Layer Fused 1x4 Parity & Throughput Benchmark...\n");
    const uint32_t D     = 2048;
    const uint32_t n_ff  = 4096;
    const uint32_t nb_d  = D / 256;
    const uint32_t nb_ff = n_ff / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *x_ref   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *x_fus   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *xb      = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *norm_w  = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *hb      = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    float *hb2     = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    block_q8_K_t *xq_buf = (block_q8_K_t *)lfm2_aligned_alloc_64((nb_d + nb_ff + 16) * sizeof(block_q8_K_t));

    block_q4_K_t *w_gate = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q4_K_t *w_up   = (block_q4_K_t *)lfm2_aligned_alloc_64((size_t)n_ff * nb_d * sizeof(block_q4_K_t));
    block_q6_K_t *w_down = (block_q6_K_t *)lfm2_aligned_alloc_64((size_t)D * nb_ff * sizeof(block_q6_K_t));

    for (uint32_t i = 0; i < D; ++i) {
        x_ref[i] = x_fus[i] = cosf((float)i * 0.019f);
        norm_w[i] = 1.0f + 0.05f * sinf((float)i * 0.02f);
    }
    for (size_t i = 0; i < (size_t)n_ff * nb_d; ++i) {
        w_gate[i].d = 0x2A00; w_gate[i].dmin = 0x2200;
        w_up[i].d   = 0x2B00; w_up[i].dmin   = 0x2300;
        for (int j = 0; j < 12; ++j) {
            w_gate[i].scales[j] = (uint8_t)((i + j * 3) & 0x3F);
            w_up[i].scales[j]   = (uint8_t)((i * 2 + j * 5) & 0x3F);
        }
        for (int j = 0; j < 128; ++j) {
            w_gate[i].qs[j] = (uint8_t)((i * 5 + j * 7) & 0xFF);
            w_up[i].qs[j]   = (uint8_t)((i * 7 + j * 11) & 0xFF);
        }
    }
    for (size_t i = 0; i < (size_t)D * nb_ff; ++i) {
        w_down[i].d = 0x2C00;
        for (int j = 0; j < 16; ++j)  w_down[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_down[i].ql[j]     = (uint8_t)((i * 13 + j * 7) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down[i].qh[j]     = (uint8_t)((i * 17 + j * 19) & 0xFF);
    }

    GGUFTensor t_norm = { .n_dims = 1, .ne = {D, 1, 1, 1},    .type = GGML_TYPE_F32,  .data = norm_w };
    GGUFTensor t_gate = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_gate };
    GGUFTensor t_up   = { .n_dims = 2, .ne = {D, n_ff, 1, 1}, .type = GGML_TYPE_Q4_K, .data = w_up   };
    GGUFTensor t_down = { .n_dims = 2, .ne = {n_ff, D, 1, 1}, .type = GGML_TYPE_Q6_K, .data = w_down };

    LFM2Layer fake_layer = {
        .ffn_norm = &t_norm, .ffn_gate = &t_gate, .ffn_up = &t_up, .ffn_down = &t_down
    };

    /* Reference Unfused 1x1 */
    {
        double ss = 0.0;
        for (uint32_t i = 0; i < D; ++i) ss += (double)x_ref[i] * (double)x_ref[i];
        float inv_rms = 1.0f / sqrtf((float)(ss / D) + 1e-5f);
        for (uint32_t i = 0; i < D; ++i) xb[i] = x_ref[i] * inv_rms * norm_w[i];

        lfm2_matmul_vec_avx2_mt(&pool, hb,  w_gate, GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        lfm2_matmul_vec_avx2_mt(&pool, hb2, w_up,   GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        for (uint32_t i = 0; i < n_ff; ++i) {
            hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
        }
        lfm2_matmul_vec_avx2_mt(&pool, xb, w_down, GGML_TYPE_Q6_K, n_ff, D, hb, xq_buf);
        for (uint32_t i = 0; i < D; ++i) x_ref[i] += xb[i];
    }

    /* Fused 1x4 */
    float *hb_fused_check = (float *)lfm2_aligned_alloc_64(n_ff * sizeof(float));
    lfm2_forward_ffn_fused_1x4(&pool, x_fus, xb, hb_fused_check, xq_buf, &fake_layer, D, 1e-5f);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float diff = fabsf(x_ref[i] - x_fus[i]);
        if (diff > max_diff) max_diff = diff;
    }
    float max_hb_diff = 0.0f;
    for (uint32_t i = 0; i < n_ff; ++i) {
        float diff = fabsf(hb[i] - hb_fused_check[i]);
        if (diff > max_hb_diff) max_hb_diff = diff;
    }
    printf("  Intermediate hb Diff: %.6e\n", max_hb_diff);
    printf("  x_ref[0]=%.6f, x_fus[0]=%.6f\n", x_ref[0], x_fus[0]);
    printf("  Full FFN Layer Output Max Diff: %.6e\n", max_diff);
    free(hb_fused_check);
    assert(max_diff < 1e-4f);

    /* Throughput Benchmark */
    const int iters = 40;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        double ss = 0.0;
        for (uint32_t i = 0; i < D; ++i) ss += (double)x_ref[i] * (double)x_ref[i];
        float inv_rms = 1.0f / sqrtf((float)(ss / D) + 1e-5f);
        for (uint32_t i = 0; i < D; ++i) xb[i] = x_ref[i] * inv_rms * norm_w[i];
        lfm2_matmul_vec_avx2_mt(&pool, hb,  w_gate, GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        lfm2_matmul_vec_avx2_mt(&pool, hb2, w_up,   GGML_TYPE_Q4_K, D, n_ff, xb, xq_buf);
        for (uint32_t i = 0; i < n_ff; ++i) hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
        lfm2_matmul_vec_avx2_mt(&pool, xb, w_down, GGML_TYPE_Q6_K, n_ff, D, hb, xq_buf);
        for (uint32_t i = 0; i < D; ++i) x_ref[i] += xb[i];
    }
    double unfused_ms = (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_forward_ffn_fused_1x4(&pool, x_fus, xb, hb, xq_buf, &fake_layer, D, 1e-5f);
    }
    double fused_ms = (now_ms() - t0) / iters;

    printf("  Benchmark (D=%u, n_ff=%u, %u threads):\n", D, n_ff, pool.n_threads);
    printf("    Unfused 1x1 FFN: %.3f ms\n", unfused_ms);
    printf("    Fused   1x4 FFN: %.3f ms (Speedup: %.2fx)\n", fused_ms, unfused_ms / fused_ms);

    lfm2_threadpool_free(&pool);
    free(x_ref); free(x_fus); free(xb); free(norm_w);
    free(hb); free(hb2); free(xq_buf);
    free(w_gate); free(w_up); free(w_down);
    printf("PASSED: Full FFN Layer Fused 1x4 verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 1x4 AVX2 & Fused Layer Tests    \n");
    printf("=======================================================\n");
    test_hsum4_epi32_avx2();
    printf("-------------------------------------------------------\n");
    test_fused_rmsnorm_quantize();
    printf("-------------------------------------------------------\n");
    test_1x4_microkernels();
    printf("-------------------------------------------------------\n");
    test_matmul_1x4_prequant();
    printf("-------------------------------------------------------\n");
    test_fused_qkv_1x4();
    printf("-------------------------------------------------------\n");
    test_fused_swiglu_1x4();
    printf("-------------------------------------------------------\n");
    test_full_ffn_layer_and_benchmark();
    printf("=======================================================\n");
    printf("ALL 7 1X4 & FUSED KERNEL TESTS PASSED WITH 100%% SUCCESS!\n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
