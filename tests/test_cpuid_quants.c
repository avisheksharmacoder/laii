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
#define LFM2_CPUID_QUANTS_NO_MAIN 1

#include "tools/lfm2_decode_1x4.c"

/* ============================================================================
 * Test 1: CPUID Hardware Detection & Kernel Function Pointer Table
 * ============================================================================ */
static void test_cpuid_detection_and_table(void) {
    printf("[Test 1/7] Verifying CPUID Detection & Dynamic Function-Pointer Table...\n");

    LFM2KernelTable kt;
    lfm2_init_cpu_kernels(&kt);

    printf("  CPU Brand String: %s\n", kt.cpu_brand);
    printf("  Features Detected: SSE4.1=%d | AVX=%d | F16C=%d | FMA=%d | AVX2=%d | AVX-VNNI=%d\n",
           kt.has_sse41, kt.has_avx, kt.has_f16c, kt.has_fma, kt.has_avx2, kt.has_avx_vnni);
    printf("  Selected ISA Engine: %s\n", kt.isa_name);

    assert(kt.cpu_brand[0] != '\0');
    assert(kt.quantize_q8_K != NULL);
    assert(kt.vec_dot_1x1[GGML_TYPE_Q4_0] != NULL);
    assert(kt.vec_dot_1x4[GGML_TYPE_Q4_0] != NULL);
    assert(kt.vec_dot_1x1[GGML_TYPE_Q5_K] != NULL);
    assert(kt.vec_dot_1x4[GGML_TYPE_Q5_K] != NULL);
    assert(kt.vec_dot_1x1[GGML_TYPE_IQ4_XS] != NULL);
    assert(kt.vec_dot_1x4[GGML_TYPE_IQ4_XS] != NULL);

    if (kt.has_avx2 && kt.has_fma && kt.has_f16c) {
        assert(strcmp(kt.isa_name, "AVX2 + FMA + F16C") == 0);
        assert(kt.quantize_q8_K == lfm2_quantize_row_q8_K_avx2_cpuid);
        assert(kt.vec_dot_1x1[GGML_TYPE_Q4_0] == lfm2_vec_dot_q4_0_q8_K_avx2);
        assert(kt.vec_dot_1x4[GGML_TYPE_Q4_0] == lfm2_vec_dot_q4_0_q8_K_1x4_avx2);
        assert(kt.vec_dot_1x1[GGML_TYPE_Q5_K] == lfm2_vec_dot_q5_K_q8_K_avx2);
        assert(kt.vec_dot_1x4[GGML_TYPE_Q5_K] == lfm2_vec_dot_q5_K_q8_K_1x4_avx2);
        assert(kt.vec_dot_1x1[GGML_TYPE_IQ4_XS] == lfm2_vec_dot_iq4_xs_q8_K_avx2);
        assert(kt.vec_dot_1x4[GGML_TYPE_IQ4_XS] == lfm2_vec_dot_iq4_xs_q8_K_1x4_avx2);
    }

    printf("PASSED: CPUID features and function-pointer table verified.\n");
}

/* ============================================================================
 * Test 2: Q8_K Activation Quantization Parity (AVX2 vs Portable Scalar)
 * ============================================================================ */
static void test_q8_K_quantization_parity(void) {
    printf("[Test 2/7] Verifying Q8_K Activation Quantization Parity (AVX2 vs Scalar)...\n");

    const uint32_t n_cols = 2048;
    const uint32_t nb = n_cols / 256;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *y_scalar = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));
    block_q8_K_t *y_avx2   = (block_q8_K_t *)lfm2_aligned_alloc_64(nb * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < n_cols; ++i) {
        x[i] = sinf((float)i * 0.031f) * ((i % 2 == 0) ? 2.5f : -1.8f);
    }

    lfm2_quantize_row_q8_K_scalar(x, y_scalar, n_cols);
    lfm2_quantize_row_q8_K_avx2_cpuid(x, y_avx2, n_cols);

    for (uint32_t b = 0; b < nb; ++b) {
        float diff_d = fabsf(y_scalar[b].d - y_avx2[b].d);
        assert(diff_d < 1e-6f);

        for (int i = 0; i < 256; ++i) {
            assert(y_scalar[b].qs[i] == y_avx2[b].qs[i]);
        }
        for (int j = 0; j < 16; ++j) {
            assert(y_scalar[b].bsums[j] == y_avx2[b].bsums[j]);
        }
    }

    /* Edge case: all zeroes */
    memset(x, 0, n_cols * sizeof(float));
    lfm2_quantize_row_q8_K_scalar(x, y_scalar, n_cols);
    lfm2_quantize_row_q8_K_avx2_cpuid(x, y_avx2, n_cols);
    for (uint32_t b = 0; b < nb; ++b) {
        assert(y_scalar[b].d == 0.0f);
        assert(y_avx2[b].d == 0.0f);
    }

    free(x);
    free(y_scalar);
    free(y_avx2);
    printf("PASSED: Q8_K activation quantization verified bit-exact against scalar ref.\n");
}

/* ============================================================================
 * Test 3: Q4_0 Quantization Parity (Scalar vs AVX2 1x1 vs AVX2 1x4)
 * ============================================================================ */
static void test_q4_0_parity(void) {
    printf("[Test 3/7] Verifying Q4_0 AVX2 1x1 & 1x4 Parity vs Scalar...\n");

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 16;
    const uint32_t nb_k   = n_cols / 256;
    const uint32_t nb_32  = n_cols / 32;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q4_0_t *w = (block_q4_0_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_32 * sizeof(block_q4_0_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = cosf((float)i * 0.019f) * 1.5f;
    lfm2_quantize_row_q8_K_scalar(x, x_q8, n_cols);

    for (size_t i = 0; i < (size_t)n_rows * nb_32; ++i) {
        w[i].d = 0x2E00; /* FP16 delta scale ~ 0.0156 */
        for (int j = 0; j < 16; ++j) {
            w[i].qs[j] = (uint8_t)((i * 17 + j * 31) & 0xFF);
        }
    }

    float max_diff_1x1 = 0.0f;
    float max_diff_1x4 = 0.0f;

    for (uint32_t r = 0; r < n_rows; ++r) {
        float ref = lfm2_vec_dot_q4_0_q8_K_scalar(w + (size_t)r * nb_32, x_q8, n_cols);
        float avx = lfm2_vec_dot_q4_0_q8_K_avx2(w + (size_t)r * nb_32, x_q8, n_cols);
        float diff = fabsf(ref - avx);
        if (diff > max_diff_1x1) max_diff_1x1 = diff;
    }

    for (uint32_t r = 0; r < n_rows; r += 4) {
        float out4[4];
        lfm2_vec_dot_q4_0_q8_K_1x4_avx2(w + (size_t)r * nb_32, nb_32, x_q8, n_cols, out4);
        for (int k = 0; k < 4; ++k) {
            float ref = lfm2_vec_dot_q4_0_q8_K_scalar(w + (size_t)(r + k) * nb_32, x_q8, n_cols);
            float diff = fabsf(ref - out4[k]);
            if (diff > max_diff_1x4) max_diff_1x4 = diff;
        }
    }

    printf("  Q4_0 Max Diff (1x1 vs Scalar): %.8f\n", max_diff_1x1);
    printf("  Q4_0 Max Diff (1x4 vs Scalar): %.8f\n", max_diff_1x4);
    assert(max_diff_1x1 == 0.0f);
    assert(max_diff_1x4 == 0.0f);

    free(x);
    free(x_q8);
    free(w);
    printf("PASSED: Q4_0 AVX2 1x1 & 1x4 kernels verified bit-exact against scalar ref.\n");
}

/* ============================================================================
 * Test 4: Q5_K Quantization Parity (Scalar vs AVX2 1x1 vs AVX2 1x4)
 * ============================================================================ */
static void test_q5_K_parity(void) {
    printf("[Test 4/7] Verifying Q5_K AVX2 1x1 & 1x4 Parity vs Scalar...\n");

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 16;
    const uint32_t nb_k   = n_cols / 256;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q5_K_t *w = (block_q5_K_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_k * sizeof(block_q5_K_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.043f) * 2.0f;
    lfm2_quantize_row_q8_K_scalar(x, x_q8, n_cols);

    for (size_t i = 0; i < (size_t)n_rows * nb_k; ++i) {
        w[i].d    = 0x2E00;
        w[i].dmin = 0x2600;
        for (int j = 0; j < 12; ++j)  w[i].scales[j] = (uint8_t)((i * 5 + j * 13) & 0x3F);
        for (int j = 0; j < 32; ++j)  w[i].qh[j]     = (uint8_t)((i * 11 + j * 23) & 0xFF);
        for (int j = 0; j < 128; ++j) w[i].qs[j]     = (uint8_t)((i * 17 + j * 37) & 0xFF);
    }

    float max_diff_1x1 = 0.0f;
    float max_diff_1x4 = 0.0f;

    for (uint32_t r = 0; r < n_rows; ++r) {
        float ref = lfm2_vec_dot_q5_K_q8_K_scalar(w + (size_t)r * nb_k, x_q8, n_cols);
        float avx = lfm2_vec_dot_q5_K_q8_K_avx2(w + (size_t)r * nb_k, x_q8, n_cols);
        float diff = fabsf(ref - avx);
        if (diff > max_diff_1x1) max_diff_1x1 = diff;
    }

    for (uint32_t r = 0; r < n_rows; r += 4) {
        float out4[4];
        lfm2_vec_dot_q5_K_q8_K_1x4_avx2(w + (size_t)r * nb_k, nb_k, x_q8, n_cols, out4);
        for (int k = 0; k < 4; ++k) {
            float ref = lfm2_vec_dot_q5_K_q8_K_scalar(w + (size_t)(r + k) * nb_k, x_q8, n_cols);
            float diff = fabsf(ref - out4[k]);
            if (diff > max_diff_1x4) max_diff_1x4 = diff;
        }
    }

    printf("  Q5_K Max Diff (1x1 vs Scalar): %.8f\n", max_diff_1x1);
    printf("  Q5_K Max Diff (1x4 vs Scalar): %.8f\n", max_diff_1x4);
    assert(max_diff_1x1 < 5e-3f);
    assert(max_diff_1x4 < 5e-3f);

    free(x);
    free(x_q8);
    free(w);
    printf("PASSED: Q5_K AVX2 1x1 & 1x4 kernels verified within FP16 precision.\n");
}

/* ============================================================================
 * Test 5: IQ4_XS Quantization Parity (Scalar vs AVX2 vpshufb 1x1 vs 1x4)
 * ============================================================================ */
static void test_iq4_xs_parity(void) {
    printf("[Test 5/7] Verifying IQ4_XS AVX2 vpshufb 1x1 & 1x4 Parity vs Scalar...\n");

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 16;
    const uint32_t nb_k   = n_cols / 256;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_iq4_xs_t *w = (block_iq4_xs_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_k * sizeof(block_iq4_xs_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.057f) * 1.8f;
    lfm2_quantize_row_q8_K_scalar(x, x_q8, n_cols);

    for (size_t i = 0; i < (size_t)n_rows * nb_k; ++i) {
        w[i].d        = 0x2800; /* FP16 delta scale ~ 0.0078 */
        w[i].scales_h = (uint16_t)((i * 97) & 0xFFFF);
        for (int j = 0; j < 4; ++j)   w[i].scales_l[j] = (uint8_t)((i * 19 + j * 47) & 0xFF);
        for (int j = 0; j < 128; ++j) w[i].qs[j]       = (uint8_t)((i * 29 + j * 41) & 0xFF);
    }

    float max_diff_1x1 = 0.0f;
    float max_diff_1x4 = 0.0f;

    for (uint32_t r = 0; r < n_rows; ++r) {
        float ref = lfm2_vec_dot_iq4_xs_q8_K_scalar(w + (size_t)r * nb_k, x_q8, n_cols);
        float avx = lfm2_vec_dot_iq4_xs_q8_K_avx2(w + (size_t)r * nb_k, x_q8, n_cols);
        float diff = fabsf(ref - avx);
        if (diff > max_diff_1x1) max_diff_1x1 = diff;
    }

    for (uint32_t r = 0; r < n_rows; r += 4) {
        float out4[4];
        lfm2_vec_dot_iq4_xs_q8_K_1x4_avx2(w + (size_t)r * nb_k, nb_k, x_q8, n_cols, out4);
        for (int k = 0; k < 4; ++k) {
            float ref = lfm2_vec_dot_iq4_xs_q8_K_scalar(w + (size_t)(r + k) * nb_k, x_q8, n_cols);
            float diff = fabsf(ref - out4[k]);
            if (diff > max_diff_1x4) max_diff_1x4 = diff;
        }
    }

    printf("  IQ4_XS Max Diff (1x1 vs Scalar): %.8f\n", max_diff_1x1);
    printf("  IQ4_XS Max Diff (1x4 vs Scalar): %.8f\n", max_diff_1x4);
    assert(max_diff_1x1 < 1e-3f);
    assert(max_diff_1x4 < 1e-3f);

    free(x);
    free(x_q8);
    free(w);
    printf("PASSED: IQ4_XS AVX2 vpshufb 1x1 & 1x4 kernels verified within FP16 precision.\n");
}

/* ============================================================================
 * Test 6: Forced Scalar CPUID Fallback (LFM2_FORCE_ISA=scalar)
 * ============================================================================ */
static void test_forced_scalar_dispatch(void) {
    printf("[Test 6/7] Verifying Forced Scalar Fallback (LFM2_FORCE_ISA=scalar)...\n");

    setenv("LFM2_FORCE_ISA", "scalar", 1);
    LFM2KernelTable kt_scalar;
    lfm2_init_cpu_kernels(&kt_scalar);

    printf("  Fallback ISA Engine Name: %s\n", kt_scalar.isa_name);
    assert(strcmp(kt_scalar.isa_name, "GENERIC_SCALAR_X86_64") == 0);
    assert(kt_scalar.quantize_q8_K == lfm2_quantize_row_q8_K_scalar);
    assert(kt_scalar.vec_dot_1x1[GGML_TYPE_Q4_0] == lfm2_vec_dot_q4_0_q8_K_scalar);
    assert(kt_scalar.vec_dot_1x4[GGML_TYPE_Q4_0] == lfm2_vec_dot_q4_0_q8_K_1x4_scalar);
    assert(kt_scalar.vec_dot_1x1[GGML_TYPE_Q5_K] == lfm2_vec_dot_q5_K_q8_K_scalar);
    assert(kt_scalar.vec_dot_1x4[GGML_TYPE_Q5_K] == lfm2_vec_dot_q5_K_q8_K_1x4_scalar);
    assert(kt_scalar.vec_dot_1x1[GGML_TYPE_IQ4_XS] == lfm2_vec_dot_iq4_xs_q8_K_scalar);
    assert(kt_scalar.vec_dot_1x4[GGML_TYPE_IQ4_XS] == lfm2_vec_dot_iq4_xs_q8_K_1x4_scalar);

    /* Verify fallback table execution produces identical scalar output */
    const uint32_t n_cols = 512;
    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64((n_cols / 256) * sizeof(block_q8_K_t));
    block_q4_0_t *w_q4 = (block_q4_0_t *)lfm2_aligned_alloc_64(4 * (n_cols / 32) * sizeof(block_q4_0_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = 1.0f;
    kt_scalar.quantize_q8_K(x, x_q8, n_cols);

    for (size_t i = 0; i < 4 * (n_cols / 32); ++i) {
        w_q4[i].d = 0x2E00;
        for (int j = 0; j < 16; ++j) w_q4[i].qs[j] = 0x88; /* (8-8) = 0 */
    }

    float out_fallback = kt_scalar.vec_dot_1x1[GGML_TYPE_Q4_0](w_q4, x_q8, n_cols);
    assert(fabsf(out_fallback) < 1e-6f);

    unsetenv("LFM2_FORCE_ISA");
    free(x);
    free(x_q8);
    free(w_q4);
    printf("PASSED: Forced scalar fallback verified with zero SIGILL risk on legacy CPUs.\n");
}

/* ============================================================================
 * Test 7: Multi-Threaded 1x4 MatMul with Q4_0, Q5_K & IQ4_XS via Threadpool
 * ============================================================================ */
static void test_multithreaded_matmul_integration(void) {
    printf("[Test 7/7] Verifying Multi-Threaded 1x4 MatMul with ThreadPool...\n");

    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 32;
    const uint32_t nb_k   = n_cols / 256;
    const uint32_t nb_32  = n_cols / 32;

    float *x = (float *)lfm2_aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_k * sizeof(block_q8_K_t));

    block_q4_0_t   *w_q4_0 = (block_q4_0_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_32 * sizeof(block_q4_0_t));
    block_q5_K_t   *w_q5_k = (block_q5_K_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_k  * sizeof(block_q5_K_t));
    block_iq4_xs_t *w_iq4  = (block_iq4_xs_t *)lfm2_aligned_alloc_64((size_t)n_rows * nb_k * sizeof(block_iq4_xs_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.02f);
    lfm2_quantize_row_q8_K_scalar(x, x_q8, n_cols);

    for (size_t i = 0; i < (size_t)n_rows * nb_32; ++i) {
        w_q4_0[i].d = 0x2E00;
        for (int j = 0; j < 16; ++j) w_q4_0[i].qs[j] = (uint8_t)((i * 7 + j * 13) & 0xFF);
    }
    for (size_t i = 0; i < (size_t)n_rows * nb_k; ++i) {
        w_q5_k[i].d    = 0x2E00;
        w_q5_k[i].dmin = 0x2600;
        for (int j = 0; j < 12; ++j)  w_q5_k[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
        for (int j = 0; j < 32; ++j)  w_q5_k[i].qh[j]     = (uint8_t)((i * 11 + j * 19) & 0xFF);
        for (int j = 0; j < 128; ++j) w_q5_k[i].qs[j]     = (uint8_t)((i * 17 + j * 31) & 0xFF);

        w_iq4[i].d        = 0x2800;
        w_iq4[i].scales_h = (uint16_t)((i * 73) & 0xFFFF);
        for (int j = 0; j < 4; ++j)   w_iq4[i].scales_l[j] = (uint8_t)((i * 19 + j * 41) & 0xFF);
        for (int j = 0; j < 128; ++j) w_iq4[i].qs[j]       = (uint8_t)((i * 23 + j * 37) & 0xFF);
    }

    GGUFTensor t_q4_0 = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q4_0,   .data = w_q4_0 };
    GGUFTensor t_q5_k = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_Q5_K,   .data = w_q5_k };
    GGUFTensor t_iq4  = { .n_dims = 2, .ne = {n_cols, n_rows, 1, 1}, .type = GGML_TYPE_IQ4_XS, .data = w_iq4 };

    float *out_q4_0 = (float *)lfm2_aligned_alloc_64(n_rows * sizeof(float));
    float *out_q5_k = (float *)lfm2_aligned_alloc_64(n_rows * sizeof(float));
    float *out_iq4  = (float *)lfm2_aligned_alloc_64(n_rows * sizeof(float));

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 3);

    lfm2_matmul_1x4_prequant_mt(&pool, out_q4_0, &t_q4_0, x, x_q8);
    lfm2_matmul_1x4_prequant_mt(&pool, out_q5_k, &t_q5_k, x, x_q8);
    lfm2_matmul_1x4_prequant_mt(&pool, out_iq4,  &t_iq4,  x, x_q8);

    /* Verify each multi-threaded row against scalar reference */
    for (uint32_t r = 0; r < n_rows; ++r) {
        float ref_q4 = lfm2_vec_dot_q4_0_q8_K_scalar(w_q4_0 + (size_t)r * nb_32, x_q8, n_cols);
        float ref_q5 = lfm2_vec_dot_q5_K_q8_K_scalar(w_q5_k + (size_t)r * nb_k,  x_q8, n_cols);
        float ref_iq = lfm2_vec_dot_iq4_xs_q8_K_scalar(w_iq4 + (size_t)r * nb_k, x_q8, n_cols);

        assert(fabsf(out_q4_0[r] - ref_q4) == 0.0f);
        assert(fabsf(out_q5_k[r] - ref_q5) < 5e-3f);
        assert(fabsf(out_iq4[r]  - ref_iq) < 5e-3f);
    }

    lfm2_threadpool_free(&pool);
    free(x); free(x_q8); free(w_q4_0); free(w_q5_k); free(w_iq4);
    free(out_q4_0); free(out_q5_k); free(out_iq4);

    printf("PASSED: Multi-threaded 1x4 MatMul with Q4_0, Q5_K & IQ4_XS verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI CPUID Multi-ISA & Quantized Test Suite\n");
    printf("=======================================================\n");

    test_cpuid_detection_and_table();
    test_q8_K_quantization_parity();
    test_q4_0_parity();
    test_q5_K_parity();
    test_iq4_xs_parity();
    test_forced_scalar_dispatch();
    test_multithreaded_matmul_integration();

    printf("=======================================================\n");
    printf("ALL 7 CPUID & QUANTIZATION TESTS PASSED WITH 100%% SUCCESS!\n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
