#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "lfm_simd.h"

int main(void) {
    printf("[Test] Running AVX2 SIMD Vectorization unit tests...\n");

#if LFM_HAS_AVX2
    printf("  Host CPU supports AVX2 + FMA: compiling with native vector intrinsics.\n");
#else
    printf("  Compiling with portable fallback SIMD paths.\n");
#endif

    // 1. Test Dot Product across different vector lengths (including unaligned lengths)
    size_t test_lengths[] = { 1, 7, 8, 15, 16, 31, 32, 33, 64, 127, 256, 513 };
    for (size_t l = 0; l < sizeof(test_lengths)/sizeof(test_lengths[0]); ++l) {
        size_t n = test_lengths[l];
        float *a = (float*)malloc(n * sizeof(float));
        float *b = (float*)malloc(n * sizeof(float));

        float scalar_expected = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            a[i] = (float)((int)(i % 7) - 3) * 0.25f;
            b[i] = (float)((int)(i % 5) - 2) * 0.5f;
            scalar_expected += a[i] * b[i];
        }

        float simd_actual = lfm_vec_dot_f32(a, b, n);
        float diff = fabsf(simd_actual - scalar_expected);
        assert(diff < 1e-4f);

        free(a);
        free(b);
    }
    printf("  AVX2 dot product verified across all 12 alignment boundary lengths.\n");

    // 2. Test Vector Multiply & Add
    {
        size_t n = 67; // odd length
        float *a = (float*)malloc(n * sizeof(float));
        float *b = (float*)malloc(n * sizeof(float));
        float *out_mul = (float*)malloc(n * sizeof(float));
        float *out_add = (float*)malloc(n * sizeof(float));

        for (size_t i = 0; i < n; ++i) {
            a[i] = (float)i * 0.1f;
            b[i] = (float)(n - i) * 0.2f;
        }

        lfm_vec_mul_f32(out_mul, a, b, n);
        lfm_vec_add_f32(out_add, a, b, n);

        for (size_t i = 0; i < n; ++i) {
            assert(fabsf(out_mul[i] - (a[i] * b[i])) < 1e-5f);
            assert(fabsf(out_add[i] - (a[i] + b[i])) < 1e-5f);
        }

        free(a); free(b); free(out_mul); free(out_add);
        printf("  AVX2 vector multiply and add verified.\n");
    }

    // 3. Test Vector RMSNorm
    {
        size_t n = 128;
        float *x = (float*)malloc(n * sizeof(float));
        float *w = (float*)malloc(n * sizeof(float));
        float *out = (float*)malloc(n * sizeof(float));

        float sum_sq = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            x[i] = (float)((int)(i % 11) - 5) * 0.3f;
            w[i] = 1.0f + (float)(i % 3) * 0.1f;
            sum_sq += x[i] * x[i];
        }

        float eps = 1e-5f;
        lfm_vec_rmsnorm_f32(out, x, w, n, eps);

        float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)n) + eps);
        for (size_t i = 0; i < n; ++i) {
            float exp_val = x[i] * rsqrt_val * w[i];
            assert(fabsf(out[i] - exp_val) < 1e-5f);
        }

        free(x); free(w); free(out);
        printf("  AVX2 vectorized RMSNorm verified.\n");
    }

    printf("PASSED: AVX2 SIMD Vectorization tests successful!\n");
    return 0;
}
