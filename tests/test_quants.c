#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "lfm_simd.h"

// FP32 to FP16 helper for test fixture generation
static uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | frac);
}

int main(void) {
    printf("[Test] Running Quantization (Q8_0 & Q4_0) dot product tests...\n");

    // 1. Test Q8_0 Dot Product (2 blocks = 64 elements)
    {
        size_t n = 64;
        BlockQ8_0_SIMD blocks[2];
        float delta0 = 0.5f;
        float delta1 = 0.25f;
        blocks[0].d = float_to_fp16(delta0);
        blocks[1].d = float_to_fp16(delta1);

        float x[64];
        float expected_sum = 0.0f;

        // Block 0
        for (int i = 0; i < 32; ++i) {
            blocks[0].qs[i] = (int8_t)((i % 15) - 7);
            x[i] = (float)(i + 1) * 0.1f;
            expected_sum += (float)blocks[0].qs[i] * x[i] * delta0;
        }

        // Block 1
        for (int i = 0; i < 32; ++i) {
            blocks[1].qs[i] = (int8_t)((i % 11) - 5);
            x[32 + i] = (float)(i + 1) * 0.2f;
            expected_sum += (float)blocks[1].qs[i] * x[32 + i] * delta1;
        }

        float actual_sum = lfm_vec_dot_q8_0_f32(blocks, x, n);
        float diff = fabsf(actual_sum - expected_sum);
        assert(diff < 1e-4f);
        printf("  Q8_0 SIMD dot product verified (diff: %.2e).\n", diff);
    }

    // 2. Test Q4_0 Dot Product (2 blocks = 64 elements)
    {
        size_t n = 64;
        BlockQ4_0_SIMD blocks[2];
        float delta0 = 0.125f;
        float delta1 = 0.0625f;
        blocks[0].d = float_to_fp16(delta0);
        blocks[1].d = float_to_fp16(delta1);

        float x[64];
        float expected_sum = 0.0f;

        // Block 0
        for (int i = 0; i < 16; ++i) {
            uint8_t low_nibble  = (uint8_t)((i % 8) + 4);  // value: low_nibble - 8
            uint8_t high_nibble = (uint8_t)(((i * 3) % 9) + 4); // value: high_nibble - 8
            blocks[0].qs[i] = (high_nibble << 4) | (low_nibble & 0x0F);

            x[i]      = (float)(i + 1) * 0.1f;
            x[16 + i] = (float)(i + 2) * 0.15f;

            int8_t v0 = (int8_t)low_nibble - 8;
            int8_t v1 = (int8_t)high_nibble - 8;
            expected_sum += ((float)v0 * x[i] + (float)v1 * x[16 + i]) * delta0;
        }

        // Block 1
        for (int i = 0; i < 16; ++i) {
            uint8_t low_nibble  = (uint8_t)((i % 7) + 5);
            uint8_t high_nibble = (uint8_t)(((i * 2) % 8) + 5);
            blocks[1].qs[i] = (high_nibble << 4) | (low_nibble & 0x0F);

            x[32 + i]      = (float)(i + 3) * 0.1f;
            x[32 + 16 + i] = (float)(i + 4) * 0.15f;

            int8_t v0 = (int8_t)low_nibble - 8;
            int8_t v1 = (int8_t)high_nibble - 8;
            expected_sum += ((float)v0 * x[32 + i] + (float)v1 * x[32 + 16 + i]) * delta1;
        }

        float actual_sum = lfm_vec_dot_q4_0_f32(blocks, x, n);
        float diff = fabsf(actual_sum - expected_sum);
        assert(diff < 1e-4f);
        printf("  Q4_0 SIMD dot product verified (diff: %.2e).\n", diff);
    }

    printf("PASSED: Quantization dot product tests successful!\n");
    return 0;
}
