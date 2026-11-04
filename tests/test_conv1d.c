#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lfm_forward.h"
#include "arena.h"

int main(void) {
    printf("[Test] Running LIV Gated ShortConv numerical parity test...\n");

    FILE *f = fopen("tests/fixtures_conv1d.bin", "rb");
    if (!f) {
        fprintf(stderr, "Cannot open tests/fixtures_conv1d.bin\n");
        return 1;
    }

    uint32_t hidden_dim = 0, k = 0;
    if (fread(&hidden_dim, 4, 1, f) != 1 || fread(&k, 4, 1, f) != 1) {
        fclose(f);
        return 1;
    }

    float *x_norm     = (float*)malloc(hidden_dim * sizeof(float));
    float *in_proj_w  = (float*)malloc(3 * hidden_dim * hidden_dim * sizeof(float));
    float *conv_w     = (float*)malloc(hidden_dim * k * sizeof(float));
    float *conv_b     = (float*)malloc(hidden_dim * sizeof(float));
    float *out_proj_w = (float*)malloc(hidden_dim * hidden_dim * sizeof(float));
    float *s0         = (float*)malloc(hidden_dim * sizeof(float));
    float *s1         = (float*)malloc(hidden_dim * sizeof(float));

    float *expected_out = (float*)malloc(hidden_dim * sizeof(float));
    float *expected_s0  = (float*)malloc(hidden_dim * sizeof(float));
    float *expected_s1  = (float*)malloc(hidden_dim * sizeof(float));

    if (!fread(x_norm, sizeof(float), hidden_dim, f) ||
        !fread(in_proj_w, sizeof(float), 3 * hidden_dim * hidden_dim, f) ||
        !fread(conv_w, sizeof(float), hidden_dim * k, f) ||
        !fread(conv_b, sizeof(float), hidden_dim, f) ||
        !fread(out_proj_w, sizeof(float), hidden_dim * hidden_dim, f) ||
        !fread(s0, sizeof(float), hidden_dim, f) ||
        !fread(s1, sizeof(float), hidden_dim, f) ||
        !fread(expected_out, sizeof(float), hidden_dim, f) ||
        !fread(expected_s0, sizeof(float), hidden_dim, f) ||
        !fread(expected_s1, sizeof(float), hidden_dim, f)) {
        fprintf(stderr, "Failed to read test data\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    LFMLayerWeights lw = {
        .shortconv_in_proj = in_proj_w,
        .shortconv_in_proj_type = GGML_TYPE_F32,
        .shortconv_conv = conv_w,
        .shortconv_conv_bias = conv_b,
        .shortconv_out_proj = out_proj_w,
        .shortconv_out_proj_type = GGML_TYPE_F32
    };

    LFMConvState state = {
        .s0 = s0,
        .s1 = s1
    };

    MemoryArena scratch;
    arena_init(&scratch, 1024 * 1024);

    float *actual_out = (float*)malloc(hidden_dim * sizeof(float));
    lfm_liv_shortconv_step(actual_out, x_norm, &lw, &state, hidden_dim, &scratch);

    // Compute max absolute differences
    float max_diff_out = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float diff = fabsf(actual_out[i] - expected_out[i]);
        if (diff > max_diff_out) max_diff_out = diff;
    }

    float max_diff_s0 = 0.0f, max_diff_s1 = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float d0 = fabsf(state.s0[i] - expected_s0[i]);
        float d1 = fabsf(state.s1[i] - expected_s1[i]);
        if (d0 > max_diff_s0) max_diff_s0 = d0;
        if (d1 > max_diff_s1) max_diff_s1 = d1;
    }

    printf("  Max difference Output:     %.2e\n", max_diff_out);
    printf("  Max difference Next State 0: %.2e\n", max_diff_s0);
    printf("  Max difference Next State 1: %.2e\n", max_diff_s1);

    int status = 0;
    if (max_diff_out > 1e-4f || max_diff_s0 > 1e-5f || max_diff_s1 > 1e-5f) {
        fprintf(stderr, "FAILED: LIV ShortConv tolerance check exceeded!\n");
        status = 1;
    } else {
        printf("PASSED: LIV Gated ShortConv numerical parity verified with PyTorch!\n");
    }

    free(x_norm);
    free(in_proj_w);
    free(conv_w);
    free(conv_b);
    free(out_proj_w);
    free(s0);
    free(s1);
    free(expected_out);
    free(expected_s0);
    free(expected_s1);
    free(actual_out);
    arena_free(&scratch);

    return status;
}
