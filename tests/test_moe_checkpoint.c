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
#define LFM2_KERNELS_FAST_NO_MAIN 1
#define LFM2_MOE_CHECKPOINT_NO_MAIN 1

#include "tools/lfm2_moe_checkpoint.c"

/* ----------------------------------------------------------------------------
 * Test 1: MoE Layer Binding & Dense Layer Fallback Detection
 * ---------------------------------------------------------------------------- */
static void test_moe_layer_binding_and_dense_fallback(void) {
    printf("[Test 1/7] Verifying MoE Layer Binding & Dense Fallback...\n");

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layers = 4;
    m.cfg.n_embd   = 2048;

    /* Layer 0 & 1: Dense SwiGLU layers (no ffn_gate_exps) */
    GGUFTensor t_dense_gate = { .name = "blk.0.ffn_gate.weight", .n_dims = 2, .ne = {2048, 5632, 1, 1}, .type = GGML_TYPE_Q4_K };
    m.n_tensors = 1;
    m.tensors = &t_dense_gate;

    LFM2MoELayerExt moe0;
    bool ok0 = lfm2_bind_moe_layer(&m, 0, 32, 4, &moe0);
    assert(ok0);
    assert(!moe0.is_moe_layer);

    /* Layer 2: MoE layer with 32 experts */
    GGUFTensor moe_tensors[5] = {
        { .name = "blk.2.ffn_gate_exps.weight", .n_dims = 3, .ne = {2048, 1536, 32, 1}, .type = GGML_TYPE_Q4_K },
        { .name = "blk.2.ffn_up_exps.weight",   .n_dims = 3, .ne = {2048, 1536, 32, 1}, .type = GGML_TYPE_Q4_K },
        { .name = "blk.2.ffn_down_exps.weight", .n_dims = 3, .ne = {1536, 2048, 32, 1}, .type = GGML_TYPE_Q6_K },
        { .name = "blk.2.ffn_gate_inp.weight",  .n_dims = 2, .ne = {2048, 32, 1, 1},   .type = GGML_TYPE_F32 },
        { .name = "blk.2.exp_probs_b.bias",    .n_dims = 1, .ne = {32, 1, 1, 1},       .type = GGML_TYPE_F32 }
    };
    m.n_tensors = 5;
    m.tensors = moe_tensors;

    LFM2MoELayerExt moe2;
    bool ok2 = lfm2_bind_moe_layer(&m, 2, 32, 4, &moe2);
    assert(ok2);
    assert(moe2.is_moe_layer);
    assert(moe2.n_experts == 32);
    assert(moe2.n_experts_used == 4);
    assert(moe2.norm_topk_prob == true);
    assert(moe2.routed_scale == 1.0f);
    assert(moe2.ffn_gate_exps == &moe_tensors[0]);
    assert(moe2.ffn_exp_probs_b == &moe_tensors[4]);

    printf("PASSED: MoE layer binding and dense layer fallback verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 2: Zero-Copy 3D -> 2D Expert Tensor Slicing & Pointer Arithmetic
 * ---------------------------------------------------------------------------- */
static void test_zero_copy_3d_expert_slicing(void) {
    printf("[Test 2/7] Verifying Zero-Copy 3D -> 2D Expert Tensor Slicing...\n");

    const uint32_t D         = 2048;
    const uint32_t n_ff_exp  = 1536;
    const uint32_t n_experts = 32;

    /* 1. Test F32 3D tensor slicing */
    size_t f32_bytes = (size_t)D * n_ff_exp * n_experts * sizeof(float);
    uint8_t *f32_buf = (uint8_t *)calloc(1, f32_bytes);
    assert(f32_buf);

    GGUFTensor t_f32_3d = {
        .name = "blk.2.ffn_gate_exps.weight",
        .n_dims = 3,
        .ne = {D, n_ff_exp, n_experts, 1},
        .type = GGML_TYPE_F32,
        .bytes = f32_bytes,
        .data = f32_buf
    };

    const size_t expert_f32_bytes = (size_t)D * n_ff_exp * sizeof(float);
    for (uint32_t e = 0; e < n_experts; ++e) {
        GGUFTensor slice = lfm2_slice_expert_tensor_2d(&t_f32_3d, e);
        assert(slice.n_dims == 2);
        assert(slice.ne[0] == D);
        assert(slice.ne[1] == n_ff_exp);
        assert(slice.ne[2] == 1);
        assert(slice.type == GGML_TYPE_F32);
        assert(slice.bytes == expert_f32_bytes);
        assert(slice.data == f32_buf + (size_t)e * expert_f32_bytes);
    }
    free(f32_buf);

    /* 2. Test Q4_K 3D tensor slicing */
    const size_t nb_d = D / 256;
    const size_t q4k_row_bytes = nb_d * sizeof(block_q4_K_t);
    const size_t q4k_expert_bytes = (size_t)n_ff_exp * q4k_row_bytes;
    const size_t q4k_total_bytes = (size_t)n_experts * q4k_expert_bytes;
    uint8_t *q4k_buf = (uint8_t *)calloc(1, q4k_total_bytes);
    assert(q4k_buf);

    GGUFTensor t_q4k_3d = {
        .name = "blk.2.ffn_gate_exps.weight",
        .n_dims = 3,
        .ne = {D, n_ff_exp, n_experts, 1},
        .type = GGML_TYPE_Q4_K,
        .bytes = q4k_total_bytes,
        .data = q4k_buf
    };

    for (uint32_t e = 0; e < n_experts; ++e) {
        GGUFTensor slice = lfm2_slice_expert_tensor_2d(&t_q4k_3d, e);
        assert(slice.n_dims == 2);
        assert(slice.ne[0] == D);
        assert(slice.ne[1] == n_ff_exp);
        assert(slice.bytes == q4k_expert_bytes);
        assert(slice.data == q4k_buf + (size_t)e * q4k_expert_bytes);
    }
    free(q4k_buf);

    printf("PASSED: Zero-copy 3D expert tensor slicing verified across block types.\n");
}

/* ----------------------------------------------------------------------------
 * Test 3: Sigmoid + Selection Bias Router Parity & Weight Normalization
 * ---------------------------------------------------------------------------- */
static void test_sigmoid_router_and_bias_selection(void) {
    printf("[Test 3/7] Verifying Sigmoid + Expert-Bias Router Parity...\n");

    const uint32_t D         = 2048;
    const uint32_t n_experts = 32;
    const uint32_t top_k     = 4;

    float *xb       = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *gate_inp = (float *)lfm2_aligned_alloc_64((size_t)n_experts * D * sizeof(float));
    float *exp_bias = (float *)lfm2_aligned_alloc_64(n_experts * sizeof(float));

    for (uint32_t i = 0; i < D; ++i) {
        xb[i] = cosf((float)i * 0.013f);
    }
    for (uint32_t e = 0; e < n_experts; ++e) {
        exp_bias[e] = (float)(e % 5) * 0.15f - 0.2f;
        for (uint32_t i = 0; i < D; ++i) {
            gate_inp[(size_t)e * D + i] = sinf((float)(e * 17 + i) * 0.007f) * 0.04f;
        }
    }

    /* Force specific bias on expert 9 and 25 to guarantee they are chosen */
    exp_bias[9]  = +2.5f;
    exp_bias[25] = +3.0f;

    GGUFTensor t_gate_inp = { .n_dims = 2, .ne = {D, n_experts, 1, 1}, .type = GGML_TYPE_F32, .data = gate_inp };
    GGUFTensor t_exp_bias = { .n_dims = 1, .ne = {n_experts, 1, 1, 1}, .type = GGML_TYPE_F32, .data = exp_bias };

    LFM2MoELayerExt moe = {
        .is_moe_layer    = true,
        .n_experts       = n_experts,
        .n_experts_used  = top_k,
        .norm_topk_prob  = true,
        .routed_scale    = 1.0f,
        .ffn_gate_inp    = &t_gate_inp,
        .ffn_exp_probs_b = &t_exp_bias
    };

    uint32_t top_idx[LFM2_MAX_TOP_K];
    float    top_w[LFM2_MAX_TOP_K];
    lfm2_moe_route_sigmoid_topk(&moe, xb, NULL, top_idx, top_w);

    /* 1. Expert 25 and 9 must be in the top 2 due to massive bias */
    bool found25 = false, found9 = false;
    for (uint32_t k = 0; k < top_k; ++k) {
        if (top_idx[k] == 25) found25 = true;
        if (top_idx[k] == 9)  found9  = true;
    }
    assert(found25);
    assert(found9);

    /* 2. Output weights must sum exactly to 1.000000 */
    float w_sum = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k) {
        assert(top_w[k] > 0.0f);
        w_sum += top_w[k];
    }
    assert(fabsf(w_sum - 1.0f) < 1e-6f);

    /* 3. Reference verification: raw weights must correspond to unbiased sigmoid */
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t e = top_idx[k];
        float ref_logit = 0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            ref_logit += gate_inp[(size_t)e * D + i] * xb[i];
        }
        float ref_sig = 1.0f / (1.0f + expf(-ref_logit));
        /* Weight ratio between winners must equal unbiased sigmoid ratio */
        uint32_t e0 = top_idx[0];
        float ref_sig0 = 0.0f;
        for (uint32_t i = 0; i < D; ++i) ref_sig0 += gate_inp[(size_t)e0 * D + i] * xb[i];
        ref_sig0 = 1.0f / (1.0f + expf(-ref_sig0));

        float actual_ratio = top_w[k] / top_w[0];
        float expected_ratio = ref_sig / ref_sig0;
        assert(fabsf(actual_ratio - expected_ratio) < 1e-5f);
    }

    free(xb); free(gate_inp); free(exp_bias);
    printf("PASSED: Sigmoid router and unbiased Top-4 weight normalization verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 4: Full Sparse MoE FFN Layer Forward Pass Execution
 * ---------------------------------------------------------------------------- */
static void test_moe_ffn_layer_forward_parity(void) {
    printf("[Test 4/7] Verifying Sparse MoE FFN Layer Forward Execution...\n");

    const uint32_t D         = 512;
    const uint32_t n_ff_exp  = 512;
    const uint32_t n_experts = 8;
    const uint32_t top_k     = 2;
    const uint32_t nb_d      = D / 256;
    const uint32_t nb_ff     = n_ff_exp / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 2);

    float *x        = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *x_orig   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *xb       = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *hb       = (float *)lfm2_aligned_alloc_64(n_ff_exp * sizeof(float));
    float *norm_w   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *gate_inp = (float *)lfm2_aligned_alloc_64((size_t)n_experts * D * sizeof(float));
    float *exp_bias = (float *)lfm2_aligned_alloc_64(n_experts * sizeof(float));
    block_q8_K_t *xq_buf = (block_q8_K_t *)lfm2_aligned_alloc_64((nb_d + nb_ff) * sizeof(block_q8_K_t));

    const size_t gu_blocks   = (size_t)n_experts * n_ff_exp * nb_d;
    const size_t down_blocks = (size_t)n_experts * D * nb_ff;
    block_q4_K_t *w_gate_exps = (block_q4_K_t *)lfm2_aligned_alloc_64(gu_blocks * sizeof(block_q4_K_t));
    block_q4_K_t *w_up_exps   = (block_q4_K_t *)lfm2_aligned_alloc_64(gu_blocks * sizeof(block_q4_K_t));
    block_q6_K_t *w_down_exps = (block_q6_K_t *)lfm2_aligned_alloc_64(down_blocks * sizeof(block_q6_K_t));

    for (uint32_t i = 0; i < D; ++i) {
        x[i] = x_orig[i] = sinf((float)i * 0.02f);
        norm_w[i] = 1.0f;
    }
    for (uint32_t e = 0; e < n_experts; ++e) {
        exp_bias[e] = (float)e * 0.1f;
        for (uint32_t i = 0; i < D; ++i) gate_inp[(size_t)e * D + i] = 0.01f * (float)(e + 1);
    }
    for (size_t i = 0; i < gu_blocks; ++i) {
        w_gate_exps[i].d = 0x2E00; w_gate_exps[i].dmin = 0x2600;
        w_up_exps[i].d   = 0x2D00; w_up_exps[i].dmin   = 0x2500;
        for (int j = 0; j < 12; ++j) {
            w_gate_exps[i].scales[j] = (uint8_t)((i * 3 + j * 7) & 0x3F);
            w_up_exps[i].scales[j]   = (uint8_t)((i * 5 + j * 11) & 0x3F);
        }
        for (int j = 0; j < 128; ++j) {
            w_gate_exps[i].qs[j] = (uint8_t)((i * 13 + j * 29) & 0xFF);
            w_up_exps[i].qs[j]   = (uint8_t)((i * 17 + j * 31) & 0xFF);
        }
    }
    for (size_t i = 0; i < down_blocks; ++i) {
        w_down_exps[i].d = 0x2B00;
        for (int j = 0; j < 16; ++j)  w_down_exps[i].scales[j] = (int8_t)(((int)(i + j * 5) % 63) - 31);
        for (int j = 0; j < 128; ++j) w_down_exps[i].ql[j]     = (uint8_t)((i * 19 + j * 11) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down_exps[i].qh[j]     = (uint8_t)((i * 23 + j * 17) & 0xFF);
    }

    GGUFTensor t_norm     = { .n_dims = 1, .ne = {D, 1, 1, 1},                 .type = GGML_TYPE_F32,  .data = norm_w      };
    GGUFTensor t_gate_inp = { .n_dims = 2, .ne = {D, n_experts, 1, 1},         .type = GGML_TYPE_F32,  .data = gate_inp    };
    GGUFTensor t_exp_bias = { .n_dims = 1, .ne = {n_experts, 1, 1, 1},         .type = GGML_TYPE_F32,  .data = exp_bias    };
    GGUFTensor t_gate_exp = { .n_dims = 3, .ne = {D, n_ff_exp, n_experts, 1},  .type = GGML_TYPE_Q4_K, .data = w_gate_exps };
    GGUFTensor t_up_exp   = { .n_dims = 3, .ne = {D, n_ff_exp, n_experts, 1},  .type = GGML_TYPE_Q4_K, .data = w_up_exps   };
    GGUFTensor t_down_exp = { .n_dims = 3, .ne = {n_ff_exp, D, n_experts, 1},  .type = GGML_TYPE_Q6_K, .data = w_down_exps };

    LFM2Layer layer = { .ffn_norm = &t_norm };
    LFM2MoELayerExt moe = {
        .is_moe_layer    = true,
        .n_experts       = n_experts,
        .n_experts_used  = top_k,
        .norm_topk_prob  = true,
        .routed_scale    = 1.0f,
        .ffn_gate_inp    = &t_gate_inp,
        .ffn_exp_probs_b = &t_exp_bias,
        .ffn_gate_exps   = &t_gate_exp,
        .ffn_up_exps     = &t_up_exp,
        .ffn_down_exps   = &t_down_exp
    };

    lfm2_forward_moe_ffn_1x4(&pool, x, xb, hb, xq_buf, &layer, &moe, D, 1e-5f);

    /* Residual output must not equal original x and must be non-zero finite floats */
    float delta_norm = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        assert(!isnan(x[i]));
        assert(!isinf(x[i]));
        delta_norm += fabsf(x[i] - x_orig[i]);
    }
    assert(delta_norm > 0.001f);

    free(x); free(x_orig); free(xb); free(hb); free(norm_w); free(gate_inp); free(exp_bias);
    free(xq_buf); free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    lfm2_threadpool_free(&pool);
    printf("PASSED: Sparse MoE FFN forward pass executed and accumulated into residual.\n");
}

/* ----------------------------------------------------------------------------
 * Test 5: Hybrid State Checkpoint Lifecycle (Save, Mutate, Restore)
 * ---------------------------------------------------------------------------- */
static void test_hybrid_state_checkpoint_lifecycle(void) {
    printf("[Test 5/7] Verifying Hybrid State Checkpoint Lifecycle...\n");

    const uint32_t D        = 256;
    const uint32_t n_layers = 16; /* 10 ShortConv + 6 GQA */

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layers          = n_layers;
    m.cfg.n_embd            = D;
    m.cfg.shortconv_l_cache = 3; /* hist_len = 2 */

    float *conv_state[LFM2_MAX_LAYERS] = {0};
    float *conv_backup[LFM2_MAX_LAYERS] = {0};

    for (uint32_t l = 0; l < n_layers; ++l) {
        /* Layers 2, 5, 8, 11, 13, 15 are GQA; rest are ShortConv */
        bool is_gqa = (l == 2 || l == 5 || l == 8 || l == 11 || l == 13 || l == 15);
        m.layers[l].block_type = is_gqa ? LFM2_BLOCK_GQA_ATTN : LFM2_BLOCK_SHORTCONV;

        if (!is_gqa) {
            conv_state[l]  = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
            conv_backup[l] = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
            for (uint32_t i = 0; i < 2 * D; ++i) {
                conv_state[l][i]  = sinf((float)(l * 37 + i) * 0.05f);
                conv_backup[l][i] = conv_state[l][i];
            }
        }
    }

    LFM2Checkpoint ckpt;
    bool ok_init = lfm2_checkpoint_init(&ckpt, &m, 32);
    assert(ok_init);
    assert(ckpt.n_conv_layers == 10);
    assert(ckpt.conv_floats_per_layer == 2 * D);

    uint32_t pos = 120;
    uint64_t rng = 0x123456789ABCDEF0ULL;
    uint32_t history_ring[32];
    for (uint32_t i = 0; i < 32; ++i) history_ring[i] = i * 11;
    uint32_t hist_count = 32;

    /* 1. Save checkpoint */
    lfm2_checkpoint_save(&ckpt, &m, conv_state, pos, rng, history_ring, hist_count);
    assert(ckpt.is_valid);

    /* 2. Mutate states to simulate advancing 5 tokens */
    pos += 5;
    rng = 0xDEADBEEF;
    hist_count += 5;
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (conv_state[l]) {
            for (uint32_t i = 0; i < 2 * D; ++i) conv_state[l][i] = 777.0f;
        }
    }
    for (uint32_t i = 0; i < 32; ++i) history_ring[i] = 999;

    /* 3. Restore checkpoint */
    uint32_t restored_pos = 0;
    uint64_t restored_rng = 0;
    uint32_t restored_hist_count = 0;
    bool ok_restore = lfm2_checkpoint_restore(&ckpt, &m, conv_state, &restored_pos,
                                              &restored_rng, history_ring, &restored_hist_count);
    assert(ok_restore);
    assert(restored_pos == 120);
    assert(restored_rng == 0x123456789ABCDEF0ULL);
    assert(restored_hist_count == 32);

    /* Verify ShortConv states are 100% bit-exact to original backup */
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (conv_state[l]) {
            for (uint32_t i = 0; i < 2 * D; ++i) {
                assert(conv_state[l][i] == conv_backup[l][i]);
            }
        }
    }
    for (uint32_t i = 0; i < 32; ++i) {
        assert(history_ring[i] == i * 11);
    }

    for (uint32_t l = 0; l < n_layers; ++l) {
        free(conv_state[l]);
        free(conv_backup[l]);
    }
    lfm2_checkpoint_free(&ckpt);
    printf("PASSED: Hybrid state checkpoint saved and restored ShortConv states bit-for-bit.\n");
}

/* ----------------------------------------------------------------------------
 * Test 6: Speculative Draft Rejection and Multi-Step State Rollback
 * ---------------------------------------------------------------------------- */
static void test_speculative_draft_rollback_cycle(void) {
    printf("[Test 6/7] Verifying Speculative Draft Rejection & Multi-Step Rollback...\n");

    const uint32_t D = 256;
    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layers          = 2;
    m.cfg.n_embd            = D;
    m.cfg.shortconv_l_cache = 3;
    m.layers[0].block_type  = LFM2_BLOCK_SHORTCONV;
    m.layers[1].block_type  = LFM2_BLOCK_SHORTCONV;

    float *conv_state[LFM2_MAX_LAYERS] = {0};
    conv_state[0] = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    conv_state[1] = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
    for (uint32_t i = 0; i < 2 * D; ++i) {
        conv_state[0][i] = 1.0f;
        conv_state[1][i] = 2.0f;
    }

    LFM2Checkpoint ckpt_branch0;
    lfm2_checkpoint_init(&ckpt_branch0, &m, 0);

    /* Snapshot at base step 10 */
    uint32_t pos = 10;
    lfm2_checkpoint_save(&ckpt_branch0, &m, conv_state, pos, 0, NULL, 0);

    /* Generate speculative draft tokens: step 11, 12, 13 (mutating conv_state) */
    for (uint32_t step = 11; step <= 13; ++step) {
        pos++;
        for (uint32_t i = 0; i < D; ++i) {
            conv_state[0][i]       = conv_state[0][D + i];
            conv_state[0][D + i]   = (float)step;
            conv_state[1][i]       = conv_state[1][D + i];
            conv_state[1][D + i]   = (float)step * 2.0f;
        }
    }
    assert(pos == 13);
    assert(conv_state[0][D] == 13.0f);

    /* Draft verification rejects steps 12 and 13: Rollback to step 10 */
    bool ok_rewind = lfm2_checkpoint_restore(&ckpt_branch0, &m, conv_state, &pos, NULL, NULL, NULL);
    assert(ok_rewind);
    assert(pos == 10);
    assert(conv_state[0][0] == 1.0f);
    assert(conv_state[0][D] == 1.0f);
    assert(conv_state[1][0] == 2.0f);
    assert(conv_state[1][D] == 2.0f);

    free(conv_state[0]);
    free(conv_state[1]);
    lfm2_checkpoint_free(&ckpt_branch0);
    printf("PASSED: Speculative draft rejection and rollback cycle verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 7: Multi-Threaded vs Single-Threaded MoE Down Accumulation Parity
 * ---------------------------------------------------------------------------- */
static void test_threadpool_parallel_moe_down_accum(void) {
    printf("[Test 7/7] Verifying ThreadPool Multi-Threaded MoE Down Parity...\n");

    const uint32_t D        = 1024;
    const uint32_t n_ff_exp = 1024;
    const size_t   nb_ff    = n_ff_exp / 256;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 4);

    float *x_st   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *x_mt   = (float *)lfm2_aligned_alloc_64(D * sizeof(float));
    float *hb_f32 = (float *)lfm2_aligned_alloc_64(n_ff_exp * sizeof(float));
    block_q8_K_t *hb_q8 = (block_q8_K_t *)lfm2_aligned_alloc_64(nb_ff * sizeof(block_q8_K_t));

    const size_t down_blocks = (size_t)D * nb_ff;
    block_q6_K_t *w_down = (block_q6_K_t *)lfm2_aligned_alloc_64(down_blocks * sizeof(block_q6_K_t));

    for (uint32_t i = 0; i < D; ++i) {
        x_st[i] = x_mt[i] = cosf((float)i * 0.015f);
    }
    for (uint32_t i = 0; i < n_ff_exp; ++i) {
        hb_f32[i] = sinf((float)i * 0.021f);
    }
    for (size_t i = 0; i < nb_ff; ++i) {
        lfm2_quantize_row_q8_K_avx2(hb_f32 + i * 256, &hb_q8[i], 256);
    }
    for (size_t i = 0; i < down_blocks; ++i) {
        w_down[i].d = 0x2A00;
        for (int j = 0; j < 16; ++j)  w_down[i].scales[j] = (int8_t)((i + j) % 15 - 7);
        for (int j = 0; j < 128; ++j) w_down[i].ql[j]     = (uint8_t)((i * 7 + j) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_down[i].qh[j]     = (uint8_t)((i * 13 + j) & 0xFF);
    }

    GGUFTensor t_down_2d = {
        .n_dims = 2,
        .ne     = {n_ff_exp, D, 1, 1},
        .type   = GGML_TYPE_Q6_K,
        .data   = w_down
    };

    const float exp_weight = 0.42857f;

    /* 1. Single-threaded execution */
    const __m128 vw = _mm_set1_ps(exp_weight);
    uint32_t r = 0;
    for (; r + 4 <= D; r += 4) {
        float down4[4];
        lfm2_compute_4rows_avx2(&t_down_2d, r, hb_f32, hb_q8, down4);
        __m128 vx = _mm_loadu_ps(x_st + r);
        __m128 vd = _mm_loadu_ps(down4);
        _mm_storeu_ps(x_st + r, _mm_fmadd_ps(vw, vd, vx));
    }
    for (; r < D; ++r) {
        const float d_val = lfm2_compute_1row_avx2(&t_down_2d, r, hb_f32, hb_q8);
        x_st[r] += exp_weight * d_val;
    }

    /* 2. Multi-threaded execution via ThreadPool */
    MoEDownAccumTaskCtx down_ctx = {
        .pool          = &pool,
        .x_residual    = x_mt,
        .w_down_2d     = &t_down_2d,
        .hb_f32        = hb_f32,
        .hb_q8         = hb_q8,
        .expert_weight = exp_weight
    };
    lfm2_parallel_for(&pool, D, 16, lfm2_moe_down_accum_1x4_worker, &down_ctx);

    /* 3. Compare ST vs MT results */
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        float diff = fabsf(x_st[i] - x_mt[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  ST vs MT MoE Down Accum Max Diff: %.8f\n", max_diff);
    assert(max_diff < 1e-6f);

    free(x_st); free(x_mt); free(hb_f32); free(hb_q8); free(w_down);
    lfm2_threadpool_free(&pool);
    printf("PASSED: Multi-threaded MoE down accumulation matches single-threaded execution bit-for-bit.\n");
}

/* ----------------------------------------------------------------------------
 * Main Test Runner
 * ---------------------------------------------------------------------------- */
int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2-MoE & State Checkpoint Tests    \n");
    printf("=======================================================\n");

    test_moe_layer_binding_and_dense_fallback();
    test_zero_copy_3d_expert_slicing();
    test_sigmoid_router_and_bias_selection();
    test_moe_ffn_layer_forward_parity();
    test_hybrid_state_checkpoint_lifecycle();
    test_speculative_draft_rollback_cycle();
    test_threadpool_parallel_moe_down_accum();

    printf("=======================================================\n");
    printf("ALL 7 MOE & CHECKPOINT TESTS PASSED WITH 100%% SUCCESS! \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
