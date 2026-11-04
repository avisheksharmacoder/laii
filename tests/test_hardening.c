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
#include <stdatomic.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "lfm2_simd_avx2.h"
#include "lfm2.h"

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1

#include "tools/lfm2_threadpool.c"

/* ============================================================================
 * Test 1: Flush-to-Zero (FTZ) & Denormals-are-Zero (DAZ) Status & Microcode Assist Immunity
 * ============================================================================ */
static void test_ftz_daz_basic(void) {
    printf("[Test 1/5] Verifying FTZ and DAZ modes & subnormal float flushing...\n");

    lfm2_set_denormals_zero_mode();
    bool active = lfm2_get_denormals_zero_mode();
    assert(active);
    printf("  FTZ and DAZ bits verified active in MXCSR register.\n");

    /* Subnormal multiplication flush test */
    volatile float subnormal = 1e-40f;
    volatile float half = 0.5f;
    volatile float prod = subnormal * half;
    assert(prod == 0.0f);

    /* Subnormal input flush test (Denormals Are Zero) */
    volatile float one = 1.0f;
    volatile float sum = subnormal + one;
    assert(sum == 1.0f);

    printf("  Subnormal arithmetic (%e * %e) cleanly flushed to 0.0f without microcode traps.\n",
           subnormal, half);
}

/* ============================================================================
 * Test 2: Worker Threads Inherit FTZ/DAZ Across All Physical Cores
 * ============================================================================ */
typedef struct {
    _Atomic int ok_count;
    _Atomic int fail_count;
} ThreadTestArg;

static void worker_check_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid;
    (void)n_threads;
    ThreadTestArg *targ = (ThreadTestArg *)arg;

    bool ftz_daz = lfm2_get_denormals_zero_mode();
    volatile float sub = 1e-40f;
    volatile float mul = sub * 0.5f;

    if (ftz_daz && mul == 0.0f) {
        atomic_fetch_add(&targ->ok_count, 1);
    } else {
        atomic_fetch_add(&targ->fail_count, 1);
    }
}

static void test_threadpool_workers_ftz_daz(void) {
    printf("[Test 2/5] Verifying FTZ and DAZ active across all threadpool background workers...\n");

    LFM2ThreadPool pool;
    bool ok = lfm2_threadpool_init(&pool, 0);
    assert(ok);
    assert(pool.n_threads >= 1);

    ThreadTestArg targ = { .ok_count = 0, .fail_count = 0 };
    lfm2_parallel_for(&pool, pool.n_threads, 1, worker_check_task, &targ);

    printf("  Dispatched to %u workers: %d verified OK, %d failed.\n",
           pool.n_threads, atomic_load(&targ.ok_count), atomic_load(&targ.fail_count));

    assert(atomic_load(&targ.fail_count) == 0);
    assert(atomic_load(&targ.ok_count) == (int)pool.n_threads);

    lfm2_threadpool_free(&pool);
    printf("  All worker threads correctly inherit FTZ/DAZ mode without microcode penalties.\n");
}

/* ============================================================================
 * Test 3: Quantization of Non-256 Divisible Column Dimensions (n_cols % 256 != 0)
 * ============================================================================ */
static void test_quantize_non_256_multiple(void) {
    printf("[Test 3/5] Verifying activation quantizer on non-256 multiple dimensions...\n");

    const uint32_t test_dims[] = { 100, 128, 300, 2048, 2688, 5632 };
    const size_t n_test_dims = sizeof(test_dims) / sizeof(test_dims[0]);

    for (size_t d = 0; d < n_test_dims; ++d) {
        uint32_t n_cols = test_dims[d];
        uint32_t nb = (n_cols + 255) / 256;
        uint32_t rem = n_cols % 256;

        float *x = (float *)malloc(n_cols * sizeof(float));
        assert(x != NULL);
        for (uint32_t i = 0; i < n_cols; ++i) {
            x[i] = sinf((float)i * 0.05f) * 2.5f;
        }

        block_q8_K_t *y_avx2 = (block_q8_K_t *)calloc(nb, sizeof(block_q8_K_t));
        block_q8_K_t *y_fb   = (block_q8_K_t *)calloc(nb, sizeof(block_q8_K_t));
        assert(y_avx2 && y_fb);

        lfm2_quantize_row_q8_K_avx2(x, y_avx2, n_cols);
        lfm2_quantize_row_q8_K_fallback(x, y_fb, n_cols);

        /* Verify all blocks have valid scales and bsums */
        for (uint32_t b = 0; b < nb; ++b) {
            assert(y_avx2[b].d > 0.0f);
            assert(!isnan(y_avx2[b].d) && !isinf(y_avx2[b].d));
            assert(fabsf(y_avx2[b].d - y_fb[b].d) < 1e-5f);

            /* Check that bsums[j] matches the sum of elements in sub-block j */
            for (int j = 0; j < 16; ++j) {
                int32_t sum_qs = 0;
                for (int l = 0; l < 16; ++l) {
                    sum_qs += y_avx2[b].qs[j * 16 + l];
                }
                assert(y_avx2[b].bsums[j] == (int16_t)sum_qs);
            }
        }

        /* If there is a remainder, verify zero-padding in tail */
        if (rem > 0) {
            uint32_t last_block = nb - 1;
            for (uint32_t i = rem; i < 256; ++i) {
                assert(y_avx2[last_block].qs[i] == 0);
                assert(y_fb[last_block].qs[i] == 0);
            }
        }

        free(y_avx2);
        free(y_fb);
        free(x);
    }

    printf("  Checked column sizes [100, 128, 300, 2048, 2688, 5632]: zero-padded remainder bits exact!\n");
}

/* ============================================================================
 * Test 4: Quantized Vector Dot Products & Matmul on Non-256 Divisible Column Dimensions
 * ============================================================================ */
static void test_dot_products_non_256_multiple(void) {
    printf("[Test 4/5] Verifying Q4_K, Q6_K, Q8_0 dot products & matmul on non-256 multiple dimensions...\n");

    const uint32_t test_dims[] = { 100, 300, 2688 };
    const size_t n_test_dims = sizeof(test_dims) / sizeof(test_dims[0]);

    for (size_t d = 0; d < n_test_dims; ++d) {
        uint32_t n_cols = test_dims[d];
        uint32_t nb_k   = (n_cols + 255) / 256;
        uint32_t nb_q80 = (n_cols + 31) / 32;

        float *x = (float *)malloc(n_cols * sizeof(float));
        assert(x);
        for (uint32_t i = 0; i < n_cols; ++i) {
            x[i] = ((float)(i % 17) - 8.0f) * 0.1f;
        }

        block_q8_K_t *xq = (block_q8_K_t *)calloc(nb_k, sizeof(block_q8_K_t));
        assert(xq);
        lfm2_quantize_row_q8_K_avx2(x, xq, n_cols);

        /* 1. Test Q4_K dot product */
        block_q4_K_t *w_q4 = (block_q4_K_t *)calloc(nb_k, sizeof(block_q4_K_t));
        assert(w_q4);
        for (uint32_t b = 0; b < nb_k; ++b) {
            w_q4[b].d    = 0x3C00; /* FP16 1.0f */
            w_q4[b].dmin = 0x0000; /* FP16 0.0f */
            memset(w_q4[b].scales, 1, sizeof(w_q4[b].scales));
            memset(w_q4[b].qs, 0x22, sizeof(w_q4[b].qs));
        }

        float dot_q4_avx = lfm2_vec_dot_q4_K_q8_K_avx2(w_q4, xq, n_cols);
        assert(!isnan(dot_q4_avx) && !isinf(dot_q4_avx));

        /* Test 4-row unrolled Q4_K dot product */
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        lfm2_vec_dot_q4_K_q8_K_4rows_avx2(w_q4, w_q4, w_q4, w_q4, xq, n_cols, &out0, &out1, &out2, &out3);
        assert(fabsf(out0 - dot_q4_avx) < 1e-4f);
        assert(fabsf(out1 - dot_q4_avx) < 1e-4f);
        assert(fabsf(out2 - dot_q4_avx) < 1e-4f);
        assert(fabsf(out3 - dot_q4_avx) < 1e-4f);

        /* 2. Test Q6_K dot product */
        block_q6_K_t *w_q6 = (block_q6_K_t *)calloc(nb_k, sizeof(block_q6_K_t));
        assert(w_q6);
        for (uint32_t b = 0; b < nb_k; ++b) {
            w_q6[b].d = 0x3C00; /* FP16 1.0f */
            memset(w_q6[b].scales, 2, sizeof(w_q6[b].scales));
            memset(w_q6[b].ql, 0x11, sizeof(w_q6[b].ql));
            memset(w_q6[b].qh, 0x00, sizeof(w_q6[b].qh));
        }
        float dot_q6_avx = lfm2_vec_dot_q6_K_q8_K_avx2(w_q6, xq, n_cols);
        assert(!isnan(dot_q6_avx) && !isinf(dot_q6_avx));

        /* 3. Test Q8_0 dot product */
        block_q8_0_t *w_q80 = (block_q8_0_t *)calloc(nb_q80, sizeof(block_q8_0_t));
        assert(w_q80);
        for (uint32_t b = 0; b < nb_q80; ++b) {
            w_q80[b].d = 0x3C00; /* FP16 1.0f */
            memset(w_q80[b].qs, 1, sizeof(w_q80[b].qs));
        }
        float dot_q80_avx = lfm2_vec_dot_q8_0_q8_K_avx2(w_q80, xq, n_cols);
        assert(!isnan(dot_q80_avx) && !isinf(dot_q80_avx));

        /* 4. Test Unified Matmul */
        const uint32_t n_rows = 4;
        block_q4_K_t *w_mat_q4 = (block_q4_K_t *)calloc((size_t)n_rows * nb_k, sizeof(block_q4_K_t));
        assert(w_mat_q4);
        for (uint32_t r = 0; r < n_rows; ++r) {
            memcpy(w_mat_q4 + (size_t)r * nb_k, w_q4, nb_k * sizeof(block_q4_K_t));
        }
        float out_mat[4] = {0};
        lfm2_matmul_vec_avx2(out_mat, w_mat_q4, 12 /* GGML_TYPE_Q4_K */, n_cols, n_rows, x, xq);
        for (uint32_t r = 0; r < n_rows; ++r) {
            assert(!isnan(out_mat[r]) && !isinf(out_mat[r]));
            assert(fabsf(out_mat[r] - dot_q4_avx) < 1e-4f);
        }

        free(w_mat_q4);
        free(w_q80);
        free(w_q6);
        free(w_q4);
        free(xq);
        free(x);
    }

    printf("  Q4_K, 1x4 microkernel, Q6_K, Q8_0, and unified matmul validated for non-256 dimensions.\n");
}

/* ============================================================================
 * Test 5: End-to-End C-API FTZ & Zero-Copy Logit Invariants
 * ============================================================================ */
static void test_api_ftz_and_logits_safety(void) {
    printf("[Test 5/5] Verifying C-API context creation inherits FTZ/DAZ and exposes valid logits...\n");

    const char *toy_path = "tests/lfm2_toy.gguf";
    lfm2_model_t *model = lfm2_model_load(toy_path, 0);
    assert(model != NULL);

    lfm2_context_params_t params = lfm2_context_default_params();
    params.max_seq_len = 128;
    params.n_threads = 2;

    lfm2_context_t *ctx = lfm2_context_create(model, &params);
    assert(ctx != NULL);

    /* Verify calling thread has FTZ/DAZ active */
    assert(lfm2_get_denormals_zero_mode());

    /* Test zero-copy logit accessor */
    const float *logits = lfm2_context_get_logits_ptr(ctx);
    assert(logits != NULL);

    /* Test NULL-safety */
    assert(lfm2_context_get_logits_ptr(NULL) == NULL);

    lfm2_model_info_t info;
    assert(lfm2_model_get_info(model, &info) == LFM2_OK);
    assert(info.n_vocab > 0);

    /* Check finite values across vocabulary */
    for (uint32_t v = 0; v < info.n_vocab; ++v) {
        assert(!isnan(logits[v]) && !isinf(logits[v]));
    }

    lfm2_context_free(ctx);
    lfm2_model_free(model);
    printf("  C-API load, context create, logits inspection, and safe teardown verified successfully.\n");
}

int main(void) {
    printf("=================================================================\n");
    printf(" Liquid Foundation Models 2.0: System Hardening & Invariant Test Suite\n");
    printf("=================================================================\n");

    test_ftz_daz_basic();
    test_threadpool_workers_ftz_daz();
    test_quantize_non_256_multiple();
    test_dot_products_non_256_multiple();
    test_api_ftz_and_logits_safety();

    printf("=================================================================\n");
    printf(" All System Hardening & Boundary Invariant Tests Passed (100%%)!\n");
    printf("=================================================================\n");
    return 0;
}
