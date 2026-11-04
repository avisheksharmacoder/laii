#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lfm_forward.h"
#include "arena.h"

int main(void) {
    printf("[Test] Running Grouped Query Attention (GQA) + RoPE numerical parity test...\n");

    FILE *f = fopen("tests/fixtures_attention.bin", "rb");
    if (!f) {
        fprintf(stderr, "Cannot open tests/fixtures_attention.bin\n");
        return 1;
    }

    uint32_t hidden_dim = 0, n_heads_q = 0, n_heads_kv = 0, head_dim = 0, pos = 0, pad = 0;
    if (fread(&hidden_dim, 4, 1, f) != 1 ||
        fread(&n_heads_q, 4, 1, f) != 1 ||
        fread(&n_heads_kv, 4, 1, f) != 1 ||
        fread(&head_dim, 4, 1, f) != 1 ||
        fread(&pos, 4, 1, f) != 1 ||
        fread(&pad, 4, 1, f) != 1) {
        fclose(f);
        return 1;
    }

    float *x_norm    = (float*)malloc(hidden_dim * sizeof(float));
    float *q_proj_w  = (float*)malloc(n_heads_q * head_dim * hidden_dim * sizeof(float));
    float *k_proj_w  = (float*)malloc(n_heads_kv * head_dim * hidden_dim * sizeof(float));
    float *v_proj_w  = (float*)malloc(n_heads_kv * head_dim * hidden_dim * sizeof(float));
    float *q_norm_w  = (float*)malloc(head_dim * sizeof(float));
    float *k_norm_w  = (float*)malloc(head_dim * sizeof(float));
    float *attn_out_w= (float*)malloc(hidden_dim * hidden_dim * sizeof(float));

    size_t hist_size = (size_t)pos * n_heads_kv * head_dim;
    float *k_hist    = (float*)malloc(hist_size * sizeof(float));
    float *v_hist    = (float*)malloc(hist_size * sizeof(float));
    float *expected_out = (float*)malloc(hidden_dim * sizeof(float));

    if (!fread(x_norm, sizeof(float), hidden_dim, f) ||
        !fread(q_proj_w, sizeof(float), n_heads_q * head_dim * hidden_dim, f) ||
        !fread(k_proj_w, sizeof(float), n_heads_kv * head_dim * hidden_dim, f) ||
        !fread(v_proj_w, sizeof(float), n_heads_kv * head_dim * hidden_dim, f) ||
        !fread(q_norm_w, sizeof(float), head_dim, f) ||
        !fread(k_norm_w, sizeof(float), head_dim, f) ||
        !fread(attn_out_w, sizeof(float), hidden_dim * hidden_dim, f) ||
        !fread(k_hist, sizeof(float), hist_size, f) ||
        !fread(v_hist, sizeof(float), hist_size, f) ||
        !fread(expected_out, sizeof(float), hidden_dim, f)) {
        fprintf(stderr, "Failed to read test data\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    LFMLayerWeights lw = {
        .q_proj = q_proj_w,
        .q_proj_type = GGML_TYPE_F32,
        .k_proj = k_proj_w,
        .k_proj_type = GGML_TYPE_F32,
        .v_proj = v_proj_w,
        .v_proj_type = GGML_TYPE_F32,
        .q_norm = q_norm_w,
        .k_norm = k_norm_w,
        .attn_out_proj = attn_out_w,
        .attn_out_proj_type = GGML_TYPE_F32
    };

    LFMConfig cfg = {
        .hidden_dim = hidden_dim,
        .n_heads_q = n_heads_q,
        .n_heads_kv = n_heads_kv,
        .head_dim = head_dim,
        .max_seq_len = pos + 10,
        .norm_eps = 1e-5f,
        .rope_theta = 1000000.0f
    };

    LFMKVCache kv;
    size_t kv_alloc = (size_t)cfg.max_seq_len * n_heads_kv * head_dim * sizeof(float);
    kv.k_cache = (float*)calloc(1, kv_alloc);
    kv.v_cache = (float*)calloc(1, kv_alloc);

    // Pre-fill history into cache
    memcpy(kv.k_cache, k_hist, hist_size * sizeof(float));
    memcpy(kv.v_cache, v_hist, hist_size * sizeof(float));

    MemoryArena scratch;
    arena_init(&scratch, 1024 * 1024);

    float *actual_out = (float*)malloc(hidden_dim * sizeof(float));
    lfm_gqa_step(actual_out, x_norm, &lw, &kv, pos, &cfg, &scratch);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float diff = fabsf(actual_out[i] - expected_out[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  Max difference Attention Output: %.2e\n", max_diff);

    int status = 0;
    if (max_diff > 1e-4f) {
        fprintf(stderr, "FAILED: Attention tolerance check exceeded!\n");
        status = 1;
    } else {
        printf("PASSED: GQA + QK-Norm + RoPE numerical parity verified with PyTorch!\n");
    }

    free(x_norm);
    free(q_proj_w);
    free(k_proj_w);
    free(v_proj_w);
    free(q_norm_w);
    free(k_norm_w);
    free(attn_out_w);
    free(k_hist);
    free(v_hist);
    free(expected_out);
    free(actual_out);
    free(kv.k_cache);
    free(kv.v_cache);
    arena_free(&scratch);

    return status;
}
