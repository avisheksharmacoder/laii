#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LFM2_SIMD_ARM64_NO_MAIN 1
#include "tools/lfm2_simd_arm64.c"

static void *aligned_alloc_64(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (bytes + 63) & ~63ULL) != 0) return NULL;
    return p;
}

static void test_kernel_table_init(void) {
    printf("[Test 1/5] Verifying Mobile Kernel Table Initialization & Capability Detection...\n");
    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);

    assert(kt.isa_name != NULL);
    printf("  Discovered ISA: %s (NEON=%d, FP16=%d, DotProd=%d, I8MM=%d)\n",
           kt.isa_name, kt.has_neon, kt.has_fp16, kt.has_dotprod, kt.has_i8mm);

    assert(kt.quantize_q8_K != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_K] != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_Q6_K] != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_Q5_K] != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_IQ4_XS] != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_0] != NULL);
    assert(kt.vec_dot_1x1[LFM2_GGML_TYPE_Q8_0] != NULL);

    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_K] != NULL);
    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_Q6_K] != NULL);
    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_Q5_K] != NULL);
    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_IQ4_XS] != NULL);
    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_0] != NULL);
    assert(kt.vec_dot_1x4[LFM2_GGML_TYPE_Q8_0] != NULL);

    printf("PASSED: Kernel table initialization verified.\n");
}

static void test_q8_k_quantization(void) {
    printf("[Test 2/5] Verifying Q8_K Activation Quantization & Sub-block bsums...\n");
    const uint32_t n_cols = 512;
    const uint32_t nb = n_cols / 256;
    float *x = (float *)aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *y = (block_q8_K_t *)aligned_alloc_64(nb * sizeof(block_q8_K_t));

    for (uint32_t i = 0; i < n_cols; ++i) {
        x[i] = ((float)(i % 100) - 50.0f) * 0.15f;
    }

    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);
    kt.quantize_q8_K(x, y, n_cols);

    for (uint32_t b = 0; b < nb; ++b) {
        assert(y[b].d > 0.0f);
        for (int j = 0; j < 16; ++j) {
            int32_t sum = 0;
            for (int l = 0; l < 16; ++l) {
                sum += y[b].qs[j * 16 + l];
            }
            assert((int16_t)sum == y[b].bsums[j]);
        }
    }

    /* All zeros case */
    memset(x, 0, n_cols * sizeof(float));
    kt.quantize_q8_K(x, y, n_cols);
    assert(y[0].d == 0.0f);
    assert(y[0].bsums[0] == 0);

    free(x);
    free(y);
    printf("PASSED: Q8_K quantization and exact bsums validated.\n");
}

static void test_dot_products_1x1(void) {
    printf("[Test 3/5] Verifying 1x1 Quantized Vector Dot Products (All Formats)...\n");
    const uint32_t n_cols = 2048;
    const uint32_t nb_k = n_cols / 256;
    const uint32_t nb_32 = n_cols / 32;

    float *x = (float *)aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4k = (block_q4_K_t *)aligned_alloc_64(nb_k * sizeof(block_q4_K_t));
    block_q6_K_t *w_q6k = (block_q6_K_t *)aligned_alloc_64(nb_k * sizeof(block_q6_K_t));
    block_q5_K_t *w_q5k = (block_q5_K_t *)aligned_alloc_64(nb_k * sizeof(block_q5_K_t));
    block_iq4_xs_t *w_iq4 = (block_iq4_xs_t *)aligned_alloc_64(nb_k * sizeof(block_iq4_xs_t));
    block_q4_0_t *w_q4_0 = (block_q4_0_t *)aligned_alloc_64(nb_32 * sizeof(block_q4_0_t));
    block_q8_0_t *w_q8_0 = (block_q8_0_t *)aligned_alloc_64(nb_32 * sizeof(block_q8_0_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = cosf((float)i * 0.031f);

    for (size_t i = 0; i < nb_k; ++i) {
        w_q4k[i].d = 0x2E00; w_q4k[i].dmin = 0x2600;
        w_q5k[i].d = 0x2E00; w_q5k[i].dmin = 0x2600;
        w_q6k[i].d = 0x2C00;
        w_iq4[i].d = 0x2800; w_iq4[i].scales_h = (uint16_t)(i * 47);
        for (int j = 0; j < 12; ++j) {
            w_q4k[i].scales[j] = (uint8_t)((i * 5 + j * 3) & 0x3F);
            w_q5k[i].scales[j] = (uint8_t)((i * 7 + j * 2) & 0x3F);
        }
        for (int j = 0; j < 16; ++j) w_q6k[i].scales[j] = (int8_t)(((int)(i * 3 + j * 7) % 63) - 31);
        for (int j = 0; j < 4; ++j)  w_iq4[i].scales_l[j] = (uint8_t)((i * 11 + j * 23) & 0xFF);
        for (int j = 0; j < 32; ++j) w_q5k[i].qh[j] = (uint8_t)((i * 13 + j * 17) & 0xFF);
        for (int j = 0; j < 64; ++j) w_q6k[i].qh[j] = (uint8_t)((i * 19 + j * 29) & 0xFF);
        for (int j = 0; j < 128; ++j) {
            uint8_t v = (uint8_t)((i * 23 + j * 19) & 0xFF);
            w_q4k[i].qs[j] = v;
            w_q5k[i].qs[j] = v;
            w_q6k[i].ql[j] = v;
            w_iq4[i].qs[j] = v;
        }
    }
    for (size_t i = 0; i < nb_32; ++i) {
        w_q4_0[i].d = 0x2E00;
        w_q8_0[i].d = 0x2E00;
        for (int j = 0; j < 16; ++j) w_q4_0[i].qs[j] = (uint8_t)((i * 17 + j * 31) & 0xFF);
        for (int j = 0; j < 32; ++j) w_q8_0[i].qs[j] = (int8_t)(((int)(i * 13 + j * 7) % 251) - 125);
    }

    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);
    kt.quantize_q8_K(x, x_q8, n_cols);

    float dot_q4k = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_K](w_q4k, x_q8, n_cols);
    float ref_q4k = lfm2_vec_dot_q4_K_q8_K_scalar(w_q4k, x_q8, n_cols);
    assert(fabsf(dot_q4k - ref_q4k) < 1e-4f);

    float dot_q6k = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q6_K](w_q6k, x_q8, n_cols);
    float ref_q6k = lfm2_vec_dot_q6_K_q8_K_scalar(w_q6k, x_q8, n_cols);
    assert(fabsf(dot_q6k - ref_q6k) < 1e-4f);

    float dot_q5k = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q5_K](w_q5k, x_q8, n_cols);
    float ref_q5k = lfm2_vec_dot_q5_K_q8_K_scalar(w_q5k, x_q8, n_cols);
    assert(fabsf(dot_q5k - ref_q5k) < 1e-4f);

    float dot_iq4 = kt.vec_dot_1x1[LFM2_GGML_TYPE_IQ4_XS](w_iq4, x_q8, n_cols);
    float ref_iq4 = lfm2_vec_dot_iq4_xs_q8_K_scalar(w_iq4, x_q8, n_cols);
    assert(fabsf(dot_iq4 - ref_iq4) < 1e-4f);

    float dot_q40 = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_0](w_q4_0, x_q8, n_cols);
    float ref_q40 = lfm2_vec_dot_q4_0_q8_K_scalar(w_q4_0, x_q8, n_cols);
    assert(fabsf(dot_q40 - ref_q40) < 1e-4f);

    float dot_q80 = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q8_0](w_q8_0, x_q8, n_cols);
    float ref_q80 = lfm2_vec_dot_q8_0_q8_K_scalar(w_q8_0, x_q8, n_cols);
    assert(fabsf(dot_q80 - ref_q80) < 1e-4f);

    printf("  1x1 parity vs reference: Q4_K=%.1e, Q6_K=%.1e, Q5_K=%.1e, IQ4_XS=%.1e, Q4_0=%.1e, Q8_0=%.1e\n",
           fabsf(dot_q4k - ref_q4k), fabsf(dot_q6k - ref_q6k), fabsf(dot_q5k - ref_q5k),
           fabsf(dot_iq4 - ref_iq4), fabsf(dot_q40 - ref_q40), fabsf(dot_q80 - ref_q80));

    free(x); free(x_q8); free(w_q4k); free(w_q6k);
    free(w_q5k); free(w_iq4); free(w_q4_0); free(w_q8_0);
    printf("PASSED: 1x1 quantized dot products verified.\n");
}

static void test_dot_products_1x4_unrolled(void) {
    printf("[Test 4/5] Verifying 1x4 Row-Unrolled Equivalence vs 1x1 Across 4 Rows...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 4;
    const uint32_t nb_k = n_cols / 256;
    const uint32_t nb_32 = n_cols / 32;

    float *x = (float *)aligned_alloc_64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)aligned_alloc_64(nb_k * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4k = (block_q4_K_t *)aligned_alloc_64(n_rows * nb_k * sizeof(block_q4_K_t));
    block_q6_K_t *w_q6k = (block_q6_K_t *)aligned_alloc_64(n_rows * nb_k * sizeof(block_q6_K_t));
    block_q4_0_t *w_q4_0 = (block_q4_0_t *)aligned_alloc_64(n_rows * nb_32 * sizeof(block_q4_0_t));

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.05f);

    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t b = 0; b < nb_k; ++b) {
            size_t idx = r * nb_k + b;
            w_q4k[idx].d = 0x2E00; w_q4k[idx].dmin = 0x2600;
            w_q6k[idx].d = 0x2C00;
            for (int j = 0; j < 12; ++j) w_q4k[idx].scales[j] = (uint8_t)((r * 7 + b * 3 + j) & 0x3F);
            for (int j = 0; j < 16; ++j) w_q6k[idx].scales[j] = (int8_t)(((int)(r * 5 + b * 2 + j) % 63) - 31);
            for (int j = 0; j < 64; ++j) w_q6k[idx].qh[j] = (uint8_t)((r * 11 + b * 13 + j) & 0xFF);
            for (int j = 0; j < 128; ++j) {
                uint8_t v = (uint8_t)((r * 17 + b * 19 + j) & 0xFF);
                w_q4k[idx].qs[j] = v;
                w_q6k[idx].ql[j] = v;
            }
        }
        for (size_t b = 0; b < nb_32; ++b) {
            size_t idx = r * nb_32 + b;
            w_q4_0[idx].d = 0x2E00;
            for (int j = 0; j < 16; ++j) w_q4_0[idx].qs[j] = (uint8_t)((r * 13 + b * 7 + j) & 0xFF);
        }
    }

    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);
    kt.quantize_q8_K(x, x_q8, n_cols);

    float o_q4k[4], o_q6k[4], o_q40[4];
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_K](w_q4k, nb_k, x_q8, n_cols, o_q4k);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q6_K](w_q6k, nb_k, x_q8, n_cols, o_q6k);
    kt.vec_dot_1x4[LFM2_GGML_TYPE_Q4_0](w_q4_0, nb_32, x_q8, n_cols, o_q40);

    for (int r = 0; r < 4; ++r) {
        float ref4 = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_K](w_q4k + r * nb_k, x_q8, n_cols);
        float ref6 = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q6_K](w_q6k + r * nb_k, x_q8, n_cols);
        float ref0 = kt.vec_dot_1x1[LFM2_GGML_TYPE_Q4_0](w_q4_0 + r * nb_32, x_q8, n_cols);
        assert(fabsf(o_q4k[r] - ref4) < 1e-4f);
        assert(fabsf(o_q6k[r] - ref6) < 1e-4f);
        assert(fabsf(o_q40[r] - ref0) < 1e-4f);
    }

    free(x); free(x_q8); free(w_q4k); free(w_q6k); free(w_q4_0);
    printf("PASSED: 1x4 row-unrolled microkernel verified.\n");
}

static void test_force_isa_override(void) {
    printf("[Test 5/5] Verifying LFM2_FORCE_ISA=scalar Override Hook...\n");
    setenv("LFM2_FORCE_ISA", "scalar", 1);
    LFM2MobileKernelTable kt;
    lfm2_init_mobile_kernels(&kt);
    assert(strcmp(kt.isa_name, "PORTABLE_C11_SCALAR") == 0);
    unsetenv("LFM2_FORCE_ISA");
    printf("PASSED: LFM2_FORCE_ISA=scalar override hook verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 ARM64 Mobile Kernel Test Suite  \n");
    printf("=======================================================\n");

    test_kernel_table_init();
    test_q8_k_quantization();
    test_dot_products_1x1();
    test_dot_products_1x4_unrolled();
    test_force_isa_override();

    printf("=======================================================\n");
    printf("ALL 5 MOBILE SIMD TESTS PASSED WITH 100%% SUCCESS!      \n");
    printf("=======================================================\n");
    return 0;
}
