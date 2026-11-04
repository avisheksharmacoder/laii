#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "lfm_fused.h"

int main(void) {
    printf("[Test] Running Fused LIV ShortConv kernel numerical equivalence test...\n");

    uint32_t hidden_dim = 128;

    float *bcx = (float*)malloc(3 * hidden_dim * sizeof(float));
    float *conv_w = (float*)malloc(hidden_dim * 3 * sizeof(float));
    float *conv_bias = (float*)malloc(hidden_dim * sizeof(float));

    // Populate with test values
    for (uint32_t i = 0; i < 3 * hidden_dim; ++i) {
        bcx[i] = (float)((int)(i % 17) - 8) * 0.1f;
    }
    for (uint32_t i = 0; i < hidden_dim * 3; ++i) {
        conv_w[i] = (float)((int)(i % 7) - 3) * 0.2f;
    }
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        conv_bias[i] = (float)(i % 5) * 0.05f;
    }

    // Set up two identical state structures
    float *s0_unfused = (float*)malloc(hidden_dim * sizeof(float));
    float *s1_unfused = (float*)malloc(hidden_dim * sizeof(float));
    float *s0_fused   = (float*)malloc(hidden_dim * sizeof(float));
    float *s1_fused   = (float*)malloc(hidden_dim * sizeof(float));

    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float val0 = (float)(i + 1) * 0.05f;
        float val1 = (float)(i + 2) * 0.07f;
        s0_unfused[i] = s0_fused[i] = val0;
        s1_unfused[i] = s1_fused[i] = val1;
    }

    // 1. Unfused Reference Execution
    float *y_unfused = (float*)malloc(hidden_dim * sizeof(float));
    const float *B = bcx;
    const float *C = bcx + hidden_dim;
    const float *x = bcx + 2 * hidden_dim;

    for (uint32_t c = 0; c < hidden_dim; ++c) {
        float u = B[c] * x[c];
        float w0 = conv_w[c * 3 + 0];
        float w1 = conv_w[c * 3 + 1];
        float w2 = conv_w[c * 3 + 2];
        float b  = conv_bias[c];

        float conv_out = (w0 * s0_unfused[c]) + (w1 * s1_unfused[c]) + (w2 * u) + b;
        s0_unfused[c] = s1_unfused[c];
        s1_unfused[c] = u;

        y_unfused[c] = C[c] * conv_out;
    }

    // 2. Fused Kernel Execution
    float *y_fused = (float*)malloc(hidden_dim * sizeof(float));
    LFMConvState fused_state = { .s0 = s0_fused, .s1 = s1_fused };
    lfm_fused_liv_gate_conv_step(y_fused, bcx, conv_w, conv_bias, &fused_state, hidden_dim);

    // 3. Compare outputs
    float max_diff_y = 0.0f;
    float max_diff_s0 = 0.0f;
    float max_diff_s1 = 0.0f;

    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float dy = fabsf(y_fused[i] - y_unfused[i]);
        if (dy > max_diff_y) max_diff_y = dy;

        float ds0 = fabsf(s0_fused[i] - s0_unfused[i]);
        if (ds0 > max_diff_s0) max_diff_s0 = ds0;

        float ds1 = fabsf(s1_fused[i] - s1_unfused[i]);
        if (ds1 > max_diff_s1) max_diff_s1 = ds1;
    }

    printf("  Max diff Output y:     %.2e\n", max_diff_y);
    printf("  Max diff State s0:     %.2e\n", max_diff_s0);
    printf("  Max diff State s1:     %.2e\n", max_diff_s1);

    assert(max_diff_y < 1e-6f);
    assert(max_diff_s0 < 1e-7f);
    assert(max_diff_s1 < 1e-7f);

    free(bcx); free(conv_w); free(conv_bias);
    free(s0_unfused); free(s1_unfused);
    free(s0_fused); free(s1_fused);
    free(y_unfused); free(y_fused);

    printf("PASSED: Fused LIV ShortConv kernel matches reference exactly!\n");
    return 0;
}
