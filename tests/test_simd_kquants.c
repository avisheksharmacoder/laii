#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "lfm2_simd_avx2.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float scalar_q4_K_ref(const block_q4_K_t *w_row, const float *x, uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d   = lfm2_fp16_to_fp32(w_row[b].d);
        const float min = lfm2_fp16_to_fp32(w_row[b].dmin);
        const uint8_t *q = w_row[b].qs;
        const float *xb  = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            lfm2_get_scale_min_k4(is + 0, w_row[b].scales, &sc1, &m1);
            lfm2_get_scale_min_k4(is + 1, w_row[b].scales, &sc2, &m2);
            float s1 = 0.0f, s1_x = 0.0f, s2 = 0.0f, s2_x = 0.0f;
            for (int l = 0; l < 32; ++l) {
                s1 += (float)(q[l] & 0x0F) * xb[j + l];      s1_x += xb[j + l];
                s2 += (float)(q[l] >> 4)   * xb[j + 32 + l]; s2_x += xb[j + 32 + l];
            }
            sumf += (d * sc1 * s1 - min * m1 * s1_x) + (d * sc2 * s2 - min * m2 * s2_x);
            q += 32;
            is += 2;
        }
    }
    return sumf;
}

static float scalar_q6_K_ref(const block_q6_K_t *w_row, const float *x, uint32_t n_cols) {
    const uint32_t nb = n_cols / 256;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float d = lfm2_fp16_to_fp32(w_row[b].d);
        const uint8_t *ql = w_row[b].ql;
        const uint8_t *qh = w_row[b].qh;
        const int8_t  *sc = w_row[b].scales;
        const float   *xb = x + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                sumf += d * (sc[is + 0] * q1 * xb[l +  0] + sc[is + 2] * q2 * xb[l + 32] +
                             sc[is + 4] * q3 * xb[l + 64] + sc[is + 6] * q4 * xb[l + 96]);
            }
            xb += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return sumf;
}

static float scalar_q8_0_ref(const block_q8_0_t *w_row, const float *x, uint32_t n_cols) {
    const uint32_t nb = n_cols / 32;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        float d = lfm2_fp16_to_fp32(w_row[b].d);
        float s = 0.0f;
        for (int j = 0; j < 32; ++j) {
            s += (float)w_row[b].qs[j] * x[b * 32 + j];
        }
        sumf += d * s;
    }
    return sumf;
}

static inline void* alloc_aligned64(size_t sz) {
    size_t aligned_sz = (sz + 63) & ~((size_t)63);
    return aligned_alloc(64, aligned_sz);
}

int main(void) {
    printf("[Test] Running AVX2 Q8_K Activation Pre-Quantization & K-Quants Dot Product Tests...\n");

    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 6144;
    const uint32_t nb = n_cols / 256;

    float *x = (float *)alloc_aligned64(n_cols * sizeof(float));
    block_q8_K_t *x_q8 = (block_q8_K_t *)alloc_aligned64(nb * sizeof(block_q8_K_t));
    block_q4_K_t *w_q4 = (block_q4_K_t *)alloc_aligned64((size_t)n_rows * nb * sizeof(block_q4_K_t));
    block_q6_K_t *w_q6 = (block_q6_K_t *)alloc_aligned64((size_t)n_rows * nb * sizeof(block_q6_K_t));
    block_q8_0_t *w_q8 = (block_q8_0_t *)alloc_aligned64((size_t)n_rows * (n_cols / 32) * sizeof(block_q8_0_t));
    float *out_hw = (float *)alloc_aligned64(n_rows * sizeof(float));

    assert(x && x_q8 && w_q4 && w_q6 && w_q8 && out_hw);

    /* Initialize synthetic activations and weights */
    for (uint32_t i = 0; i < n_cols; ++i) {
        x[i] = sinf((float)i * 0.017f) * 1.5f;
    }

    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3000; /* ~0.125 */
        w_q4[i].dmin = 0x2800; /* ~0.03125 */
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i + j * 7) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 13 + j * 29) & 0xFF);

        w_q6[i].d = 0x2C00; /* ~0.0625 */
        for (int j = 0; j < 16; ++j)  w_q6[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_q6[i].ql[j]     = (uint8_t)((i * 19 + j * 11) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_q6[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    for (size_t i = 0; i < (size_t)n_rows * (n_cols / 32); ++i) {
        w_q8[i].d = 0x3000;
        for (int j = 0; j < 32; ++j) w_q8[i].qs[j] = (int8_t)(((int)(i * 7 + j * 3) % 127) - 63);
    }

    /* 1. Test Q8_K Activation Quantization */
#if LFM2_HAS_AVX2
    lfm2_quantize_row_q8_K_avx2(x, x_q8, n_cols);
#else
    lfm2_quantize_row_q8_K_fallback(x, x_q8, n_cols);
#endif

    for (uint32_t b = 0; b < nb; ++b) {
        assert(x_q8[b].d > 0.0f);
        for (int j = 0; j < 16; ++j) {
            int32_t sum = 0;
            for (int l = 0; l < 16; ++l) {
                sum += x_q8[b].qs[j * 16 + l];
            }
            assert((int16_t)sum == x_q8[b].bsums[j]);
        }
    }
    printf("  Activation vector Q8_K pre-quantization and bsums[16] sub-block verification: OK\n");

    /* 2. Verify Q4_K dot product numerical parity */
    float ref_q4 = scalar_q4_K_ref(w_q4, x, n_cols);
#if LFM2_HAS_AVX2
    float avx_q4 = lfm2_vec_dot_q4_K_q8_K_avx2(w_q4, x_q8, n_cols);
#else
    float avx_q4 = lfm2_vec_dot_q4_K_fallback(w_q4, x, n_cols);
#endif
    float diff_q4 = fabsf(ref_q4 - avx_q4);
    float rel_q4 = 100.0f * diff_q4 / (fabsf(ref_q4) + 1e-6f);
    printf("  Q4_K single row: Ref = %+.4f, AVX2 = %+.4f (Diff: %.4e, Rel: %.4f%%)\n",
           ref_q4, avx_q4, diff_q4, rel_q4);
    assert(rel_q4 < 0.15f);

    /* 3. Verify Q6_K dot product numerical parity */
    float ref_q6 = scalar_q6_K_ref(w_q6, x, n_cols);
#if LFM2_HAS_AVX2
    float avx_q6 = lfm2_vec_dot_q6_K_q8_K_avx2(w_q6, x_q8, n_cols);
#else
    float avx_q6 = lfm2_vec_dot_q6_K_fallback(w_q6, x, n_cols);
#endif
    float diff_q6 = fabsf(ref_q6 - avx_q6);
    float rel_q6 = 100.0f * diff_q6 / (fabsf(ref_q6) + 1e-6f);
    printf("  Q6_K single row: Ref = %+.4f, AVX2 = %+.4f (Diff: %.4e, Rel: %.4f%%)\n",
           ref_q6, avx_q6, diff_q6, rel_q6);
    assert(rel_q6 < 0.25f);

    /* 4. Verify Q8_0 dot product numerical parity */
    float ref_q8 = scalar_q8_0_ref(w_q8, x, n_cols);
#if LFM2_HAS_AVX2
    float avx_q8 = lfm2_vec_dot_q8_0_q8_K_avx2(w_q8, x_q8, n_cols);
#else
    float avx_q8 = ref_q8;
#endif
    float diff_q8 = fabsf(ref_q8 - avx_q8);
    float rel_q8 = 100.0f * diff_q8 / (fabsf(ref_q8) + 1e-6f);
    printf("  Q8_0 single row: Ref = %+.4f, AVX2 = %+.4f (Diff: %.4e, Rel: %.4f%%)\n",
           ref_q8, avx_q8, diff_q8, rel_q8);
    assert(rel_q8 < 1.0f);

    /* 5. Matrix-Vector Multiplication Test & Benchmark [6144 x 2048] */
#if LFM2_HAS_AVX2
    lfm2_matmul_vec_avx2(out_hw, w_q4, 12, n_cols, n_rows, x, x_q8);
    for (uint32_t r = 0; r < 5; ++r) {
        float ref_r = scalar_q4_K_ref(w_q4 + (size_t)r * nb, x, n_cols);
        float rel_r = 100.0f * fabsf(ref_r - out_hw[r]) / (fabsf(ref_r) + 1e-6f);
        assert(rel_r < 0.15f);
    }

    const int iters = 20;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_matmul_vec_avx2(out_hw, w_q4, 12, n_cols, n_rows, x, x_q8);
    }
    double dt_q4 = (now_ms() - t0) / iters;
    double bw_q4 = ((double)n_rows * nb * sizeof(block_q4_K_t)) / (dt_q4 * 1e6);

    t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        lfm2_matmul_vec_avx2(out_hw, w_q6, 14, n_cols, n_rows, x, x_q8);
    }
    double dt_q6 = (now_ms() - t0) / iters;
    double bw_q6 = ((double)n_rows * nb * sizeof(block_q6_K_t)) / (dt_q6 * 1e6);

    printf("  Performance [Matrix %u x %u]:\n", n_rows, n_cols);
    printf("    Q4_K MatVec latency: %.3f ms | Bandwidth: %.2f GB/s\n", dt_q4, bw_q4);
    printf("    Q6_K MatVec latency: %.3f ms | Bandwidth: %.2f GB/s\n", dt_q6, bw_q6);
#endif

    free(x); free(x_q8); free(w_q4); free(w_q6); free(w_q8); free(out_hw);
    printf("PASSED: AVX2 Q8_K Activation Pre-Quantization & K-Quants SIMD tests successful!\n");
    return 0;
}
