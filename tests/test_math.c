#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "lfm_forward.h"

static float vec_norm_sq(const float *x, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += x[i] * x[i];
    return sum;
}

int main(void) {
    printf("[Test] Running Mathematical Primitives unit tests...\n");

    // 1. Test RMSNorm
    {
        float x[4] = { 1.0f, -2.0f, 3.0f, -4.0f };
        float w[4] = { 2.0f, 0.5f, 1.0f, 0.25f };
        float out[4] = { 0 };
        float eps = 1e-5f;

        // Mean of squares: (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
        // rsqrt: 1 / sqrt(7.5 + 1e-5) = 1 / 2.738612787 = 0.36514837
        lfm_rmsnorm(out, x, w, 4, eps);

        float expected_rsqrt = 1.0f / sqrtf(7.5f + eps);
        for (int i = 0; i < 4; ++i) {
            float exp_val = x[i] * expected_rsqrt * w[i];
            float diff = fabsf(out[i] - exp_val);
            assert(diff < 1e-6f);
        }
        printf("  RMSNorm numerical precision verified.\n");
    }

    // 2. Test SiLU Activation
    {
        float x[5] = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        float out[5] = { 0 };
        lfm_silu_vec(out, x, 5);

        for (int i = 0; i < 5; ++i) {
            float expected = x[i] / (1.0f + expf(-x[i]));
            float diff = fabsf(out[i] - expected);
            assert(diff < 1e-6f);
        }
        printf("  SiLU activation verified.\n");
    }

    // 3. Test RoPE Orthogonality (Rotations must preserve vector norm!)
    {
        uint32_t head_dim = 16;
        float vec[16];
        for (uint32_t i = 0; i < head_dim; ++i) vec[i] = (float)(i + 1) * 0.1f;

        float norm_before = vec_norm_sq(vec, head_dim);

        // Rotate at position 42
        lfm_rope_neox(vec, head_dim, 42, 1000000.0f);

        float norm_after = vec_norm_sq(vec, head_dim);
        float diff = fabsf(norm_before - norm_after);
        assert(diff < 1e-5f);
        printf("  RoPE rotation orthogonality verified (norm before: %.5f, after: %.5f).\n",
               norm_before, norm_after);
    }

    // 4. Test Matrix-Vector Dot Product (FP32)
    {
        float W[2 * 3] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        float x[3] = { 0.5f, -1.0f, 2.0f };
        float out[2] = { 0 };

        lfm_matvec(out, W, GGML_TYPE_F32, x, 2, 3);
        // Row 0: 1*0.5 + 2*(-1) + 3*2 = 0.5 - 2 + 6 = 4.5
        // Row 1: 4*0.5 + 5*(-1) + 6*2 = 2 - 5 + 12 = 9.0
        assert(fabsf(out[0] - 4.5f) < 1e-5f);
        assert(fabsf(out[1] - 9.0f) < 1e-5f);
        printf("  FP32 MatVec dot-product verified.\n");
    }

    printf("PASSED: All Mathematical Primitives tests successful!\n");
    return 0;
}
