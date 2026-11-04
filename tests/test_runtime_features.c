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
#define LFM2_CPUID_QUANTS_NO_MAIN 1
#define LFM2_KERNELS_FAST_NO_MAIN 1
#define LFM2_RUNTIME_FEATURES_NO_MAIN 1

#include "tools/lfm2_runtime_features.c"

/* ============================================================================
 * Test 1: Pre-Decoded Raw Byte Vocabulary & GPT-2 Byte Slicing
 * ============================================================================ */
static void test_decoded_vocab_bpe_mapping(void) {
    printf("[Test 1/7] Verifying Pre-Decoded Raw Byte Vocabulary & GPT-2 Byte Slicing...\n");

    /* Test raw GPT-2 byte decoder directly on Unicode-mapped bytes */
    char decoded[16];
    /* Ġ (codepoint 288, UTF-8 0xC4 0xA0) -> raw ASCII space (0x20) */
    const char g_token[] = "\xC4\xA0";
    uint16_t len = decode_gpt2_token_bytes(g_token, decoded, sizeof(decoded));
    assert(len == 1);
    assert(decoded[0] == ' ');

    /* Ċ (codepoint 266, UTF-8 0xC4 0x8A) -> raw ASCII newline (0x0A) */
    const char c_token[] = "\xC4\x8A";
    len = decode_gpt2_token_bytes(c_token, decoded, sizeof(decoded));
    assert(len == 1);
    assert(decoded[0] == '\n');

    /* Build sample vocabulary with special and standard tokens */
    const char *mock_tokens[] = {
        "<|startoftext|>", "<|im_start|>", "<|im_end|>",
        "<|tool_call_start|>", "<|tool_call_end|>", "<think>", "</think>",
        "{", "}", "[", "]", ":", ",", "\"", "name", "value", "123", " ", "\n"
    };
    const uint32_t V = sizeof(mock_tokens) / sizeof(mock_tokens[0]);

    LFM2DecodedVocab dv;
    bool ok = lfm2_decoded_vocab_init(&dv, (char **)mock_tokens, V);
    assert(ok);
    assert(dv.n_vocab == V);

    assert(dv.is_special[0]); /* <|startoftext|> */
    assert(dv.is_special[1]); /* <|im_start|>     */
    assert(dv.is_special[2]); /* <|im_end|>       */
    assert(dv.is_special[3]); /* <|tool_call_start|> */
    assert(dv.is_special[4]); /* <|tool_call_end|>   */
    assert(!dv.is_special[7]); /* "{" */

    assert(dv.id_im_end == 2);
    assert(dv.id_tool_start == 3);
    assert(dv.id_tool_end == 4);
    assert(dv.id_think_open == 5);
    assert(dv.id_think_close == 6);

    lfm2_decoded_vocab_free(&dv);
    printf("PASSED: Pre-decoded vocabulary and GPT-2 byte mapping verified.\n");
}

/* ============================================================================
 * Test 2: RFC 8259 JSON Object Grammar PDA Logit Masking
 * ============================================================================ */
static void test_json_grammar_pda(void) {
    printf("[Test 2/7] Verifying RFC 8259 JSON Object Grammar PDA Logit Masking...\n");

    const char *vocab_strs[] = {
        "<|im_end|>", "{", "}", ":", ",", "\"",
        "model", "fast", "count", "ratio", "null",
        "true", "false", "42", "-3.14", "[", "]",
        "invalid_symbol_!", " "
    };
    const uint32_t V = sizeof(vocab_strs) / sizeof(vocab_strs[0]);

    LFM2DecodedVocab dv;
    bool vok = lfm2_decoded_vocab_init(&dv, (char **)vocab_strs, V);
    assert(vok);

    LFM2Grammar g;
    lfm2_grammar_init(&g, LFM2_GRAMMAR_JSON_OBJECT);
    assert(g.state == G_JSON_START);

    float logits[32];
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;

    /* Step 0: At G_JSON_START, only '{' and whitespace allowed */
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[1]));  /* '{' */
    assert(!isinf(logits[18])); /* ' ' */
    assert(isinf(logits[0]));   /* <|im_end|> blocked */
    assert(isinf(logits[6]));   /* "model" blocked */
    assert(isinf(logits[17]));  /* "invalid_symbol_!" blocked */

    /* Commit: '{' */
    lfm2_grammar_commit_token(&g, &dv, 1);
    assert(g.state == G_JSON_OBJ_KEY_OR_END);

    /* Step 1: Expecting '"' or '}' */
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[5])); /* '"' */
    assert(!isinf(logits[2])); /* '}' */
    assert(isinf(logits[3]));  /* ':' blocked */
    assert(isinf(logits[11])); /* 'true' blocked */

    /* Commit: '"' + "model" + '"' + ':' */
    lfm2_grammar_commit_token(&g, &dv, 5); /* '"' */
    lfm2_grammar_commit_token(&g, &dv, 6); /* "model" */
    lfm2_grammar_commit_token(&g, &dv, 5); /* '"' */
    assert(g.state == G_JSON_AFTER_KEY);

    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[3])); /* ':' allowed */
    assert(isinf(logits[4]));  /* ',' blocked */
    assert(isinf(logits[2]));  /* '}' blocked */
    lfm2_grammar_commit_token(&g, &dv, 3); /* ':' */

    /* Step 2: Expecting JSON value */
    assert(g.state == G_JSON_VAL);
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[5]));  /* '"' string value */
    assert(!isinf(logits[11])); /* 'true' literal */
    assert(!isinf(logits[10])); /* 'null' literal */
    assert(!isinf(logits[13])); /* '42' integer */
    assert(!isinf(logits[14])); /* '-3.14' float */
    assert(!isinf(logits[15])); /* '[' nested array */
    assert(!isinf(logits[1]));  /* '{' nested object */
    assert(isinf(logits[3]));   /* ':' blocked */

    /* Commit: '"' + "fast" + '"' + '}' */
    lfm2_grammar_commit_token(&g, &dv, 5);
    lfm2_grammar_commit_token(&g, &dv, 7);
    lfm2_grammar_commit_token(&g, &dv, 5);
    assert(g.state == G_JSON_AFTER_VAL);
    lfm2_grammar_commit_token(&g, &dv, 2); /* '}' */

    /* Top-level object complete! */
    assert(g.state == G_COMPLETE);
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[0])); /* ONLY <|im_end|> allowed! */
    assert(isinf(logits[1]));  /* '{' blocked */
    assert(isinf(logits[6]));  /* "model" blocked */
    assert(isinf(logits[13])); /* '42' blocked */

    lfm2_decoded_vocab_free(&dv);
    printf("PASSED: RFC 8259 JSON grammar PDA correctly constrained logit masks.\n");
}

/* ============================================================================
 * Test 3: LFM2 Native Pythonic Tool-Call Grammar PDA Logit Masking
 * ============================================================================ */
static void test_tool_call_grammar_pda(void) {
    printf("[Test 3/7] Verifying LFM2 Pythonic Tool-Call Grammar PDA...\n");

    const char *vocab_strs[] = {
        "<|tool_call_start|>", "<|tool_call_end|>", "[", "]", "(", ")",
        "=", ",", "\"", "get_weather", "search_web", "city",
        "Bengaluru", "metric", "True", "False", "42", " ", "@@illegal#"
    };
    const uint32_t V = sizeof(vocab_strs) / sizeof(vocab_strs[0]);

    LFM2DecodedVocab dv;
    lfm2_decoded_vocab_init(&dv, (char **)vocab_strs, V);

    LFM2Grammar g;
    lfm2_grammar_init(&g, LFM2_GRAMMAR_LFM2_TOOL_CALL);
    assert(g.state == G_TOOL_WAIT_TRIGGER);

    float logits[32];

    /* Before <|tool_call_start|>, model is in free generation */
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    uint32_t allowed = lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(allowed == V);

    /* Model emits <|tool_call_start|> (tid = 0) */
    lfm2_grammar_commit_token(&g, &dv, 0);
    assert(g.state == G_TOOL_START);

    /* Inside G_TOOL_START, expecting '[' or '{' */
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[2])); /* '[' allowed */
    assert(isinf(logits[9]));  /* function name blocked before '[' */
    assert(isinf(logits[1]));  /* <|tool_call_end|> blocked before call starts */

    /* Commit: '[' -> G_TOOL_CALL_OR_END */
    lfm2_grammar_commit_token(&g, &dv, 2);
    assert(g.state == G_TOOL_CALL_OR_END);

    /* Step full call: get_weather(city="Bengaluru", metric=True) */
    const uint32_t seq[] = {
        9,  /* get_weather */
        4,  /* (           */
        11, /* city        */
        6,  /* =           */
        8,  /* "           */
        12, /* Bengaluru   */
        8,  /* "           */
        7,  /* ,           */
        13, /* metric      */
        6,  /* =           */
        14, /* True        */
        5,  /* )           */
        3   /* ]           */
    };

    for (size_t s = 0; s < sizeof(seq) / sizeof(seq[0]); ++s) {
        for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
        lfm2_grammar_mask_logits(&g, &dv, logits);
        assert(!isinf(logits[seq[s]]));
        /* Premature <|tool_call_end|> (tid 1) must be blocked while call is incomplete */
        assert(isinf(logits[1]));
        /* Outside string literals, illegal syntax symbols must be blocked */
        if (g.state != G_JSON_STR) {
            assert(isinf(logits[18]));
        }
        lfm2_grammar_commit_token(&g, &dv, seq[s]);
    }

    /* Top-level tool call closed: state must be G_COMPLETE */
    assert(g.state == G_COMPLETE);
    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.0f;
    lfm2_grammar_mask_logits(&g, &dv, logits);
    assert(!isinf(logits[1])); /* ONLY <|tool_call_end|> (tid 1) allowed! */
    assert(isinf(logits[9]));  /* get_weather blocked */
    assert(isinf(logits[2]));  /* '[' blocked */

    /* Commit <|tool_call_end|>: returns to unconstrained G_TOOL_WAIT_TRIGGER */
    lfm2_grammar_commit_token(&g, &dv, 1);
    assert(g.state == G_TOOL_WAIT_TRIGGER);

    lfm2_decoded_vocab_free(&dv);
    printf("PASSED: LFM2 Pythonic tool-call PDA successfully enforced call structure.\n");
}

/* ============================================================================
 * Test 4: `<think>` Reasoning Token Budget Controller Lifecycle
 * ============================================================================ */
static void test_think_budget_controller(void) {
    printf("[Test 4/7] Verifying <think> Reasoning Budget Controller Lifecycle...\n");

    const char *vocab_strs[] = {
        "<|im_end|>", "<think>", "</think>", "\n",
        "step1", "step2", "step3", "final_answer"
    };
    const uint32_t V = sizeof(vocab_strs) / sizeof(vocab_strs[0]);

    LFM2DecodedVocab dv;
    lfm2_decoded_vocab_init(&dv, (char **)vocab_strs, V);

    const uint32_t close_seq[3] = { 3 /* \n */, 2 /* </think> */, 3 /* \n */ };
    LFM2ThinkController tc;
    lfm2_think_controller_init(&tc, 3 /* max 3 reasoning tokens */, close_seq, 3);
    assert(tc.phase == LFM2_THINK_IDLE);

    /* 1. Enter thinking mode */
    lfm2_think_commit_token(&tc, &dv, 1 /* <think> */);
    assert(tc.phase == LFM2_THINK_ACTIVE);
    assert(tc.think_tokens_used == 0);

    /* 2. Emit 2 reasoning tokens (budget = 3) */
    lfm2_think_commit_token(&tc, &dv, 4 /* step1 */);
    lfm2_think_commit_token(&tc, &dv, 5 /* step2 */);
    assert(tc.phase == LFM2_THINK_ACTIVE);
    assert(tc.think_tokens_used == 2);

    /* 3. Emit 3rd reasoning token -> budget hit */
    lfm2_think_commit_token(&tc, &dv, 6 /* step3 */);
    assert(tc.think_tokens_used == 3);

    /* 4. Mask logits: controller must force close sequence \n</think>\n */
    float logits[16];
    for (uint32_t i = 0; i < V; ++i) logits[i] = 1.0f;
    lfm2_think_apply_budget_mask(&tc, logits, V);
    assert(tc.phase == LFM2_THINK_FORCING_CLOSE);
    assert(logits[close_seq[0]] > 1e20f); /* \n forced */
    assert(isinf(logits[4]));             /* step1 blocked */
    assert(isinf(logits[7]));             /* final_answer blocked */

    /* Commit close token 0: \n */
    lfm2_think_commit_token(&tc, &dv, close_seq[0]);
    assert(tc.phase == LFM2_THINK_FORCING_CLOSE);

    /* Commit close token 1: </think> */
    for (uint32_t i = 0; i < V; ++i) logits[i] = 1.0f;
    lfm2_think_apply_budget_mask(&tc, logits, V);
    assert(logits[close_seq[1]] > 1e20f);
    lfm2_think_commit_token(&tc, &dv, close_seq[1]);
    assert(tc.phase == LFM2_THINK_FORCING_CLOSE);

    /* Commit close token 2: \n */
    for (uint32_t i = 0; i < V; ++i) logits[i] = 1.0f;
    lfm2_think_apply_budget_mask(&tc, logits, V);
    assert(logits[close_seq[2]] > 1e20f);
    lfm2_think_commit_token(&tc, &dv, close_seq[2]);

    /* Now transition to ANSWERING! */
    assert(tc.phase == LFM2_THINK_ANSWERING);

    /* 5. Test natural early close before budget */
    LFM2ThinkController tc2;
    lfm2_think_controller_init(&tc2, 100 /* generous budget */, close_seq, 3);
    lfm2_think_commit_token(&tc2, &dv, 1 /* <think> */);
    assert(tc2.phase == LFM2_THINK_ACTIVE);
    lfm2_think_commit_token(&tc2, &dv, 4 /* step1 */);
    lfm2_think_commit_token(&tc2, &dv, 2 /* </think> */); /* early natural close */
    assert(tc2.phase == LFM2_THINK_ANSWERING);

    lfm2_decoded_vocab_free(&dv);
    printf("PASSED: <think> budget controller lifecycle and forced termination verified.\n");
}

/* ============================================================================
 * Test 5: GQA Attention-Sink KV-Cache Shift & RoPE Counter-Rotation Parity
 * ============================================================================ */
static void test_kv_cache_shift_sinks_parity_and_rope(void) {
    printf("[Test 5/7] Verifying GQA Attention-Sink KV-Cache Shift & RoPE Inverse Rotation...\n");

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layers     = 2;
    m.cfg.n_embd       = 2048;
    m.cfg.n_head_q     = 32;
    m.cfg.n_head_kv[0] = 0; /* ShortConv layer */
    m.cfg.n_head_kv[1] = 8; /* GQA layer (8 heads * 64 dim = 512) */
    m.layers[0].block_type = LFM2_BLOCK_SHORTCONV;
    m.layers[1].block_type = LFM2_BLOCK_GQA_ATTN;

    const uint32_t max_seq   = 512;
    const uint32_t kv_dim    = 8 * 64;
    const uint32_t old_pos   = 250;
    const uint32_t n_sink    = 32;
    const uint32_t n_discard = 100;
    const uint32_t new_pos   = old_pos - n_discard; /* 150 */

    LFM2RoPELUT rope_lut;
    lfm2_rope_lut_init(&rope_lut, max_seq, 64, 1000000.0f);

    uint16_t *k_cache[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_cache[LFM2_MAX_LAYERS] = {0};
    k_cache[1] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));
    v_cache[1] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));

    float *k_orig_f32   = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    float *k_direct_f32 = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    uint16_t *k_direct_f16 = (uint16_t *)lfm2_aligned_alloc_64(kv_dim * sizeof(uint16_t));

    for (uint32_t i = 0; i < kv_dim; ++i) {
        k_orig_f32[i] = k_direct_f32[i] = cosf((float)i * 0.023f) * 1.5f;
    }

    /* 1. Rotate k_orig at old_pos (250) and store in k_cache[1][250] */
    lfm2_qk_norm_rope_avx2(k_orig_f32, k_cache[1] + (size_t)old_pos * kv_dim,
                           8, 64, NULL, 1e-5f, &rope_lut, old_pos);

    /* 2. Rotate k_direct directly at new_pos (150) as ground truth */
    lfm2_qk_norm_rope_avx2(k_direct_f32, k_direct_f16,
                           8, 64, NULL, 1e-5f, &rope_lut, new_pos);

    /* 3. Shift KV cache from cur_pos = 300 */
    uint32_t cursor = 300;
    lfm2_kv_cache_shift_sinks_avx2(k_cache, v_cache, &m, &rope_lut, &cursor, n_sink, n_discard);

    assert(cursor == 300 - n_discard); /* 200 */

    /* 4. Verify counter-rotated key at new_pos matches ground-truth */
    const uint16_t *shifted_k = k_cache[1] + (size_t)new_pos * kv_dim;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < kv_dim; ++i) {
        float v_shift  = fp16_to_fp32(shifted_k[i]);
        float v_direct = fp16_to_fp32(k_direct_f16[i]);
        float err = fabsf(v_shift - v_direct);
        if (err > max_err) max_err = err;
    }

    printf("  RoPE Re-Rotation Error: %.6f (FP16 round-trip precision)\n", max_err);
    assert(max_err < 1.5e-3f);

    free(k_cache[1]); free(v_cache[1]);
    free(k_orig_f32); free(k_direct_f32); free(k_direct_f16);
    lfm2_rope_lut_free(&rope_lut);
    printf("PASSED: Attention-sink KV-cache shifting and RoPE counter-rotation verified.\n");
}

/* ============================================================================
 * Test 6: KV-Cache Shifting Scalar vs AVX2 Parity & Multi-ISA Fallback
 * ============================================================================ */
static void test_kv_cache_shift_scalar_vs_avx2_parity(void) {
    printf("[Test 6/7] Verifying KV-Cache Shifting Scalar vs AVX2 Parity & Forced Fallback...\n");

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layers     = 1;
    m.cfg.n_embd       = 2048;
    m.cfg.n_head_q     = 32;
    m.cfg.n_head_kv[0] = 8;
    m.layers[0].block_type = LFM2_BLOCK_GQA_ATTN;

    const uint32_t max_seq = 256;
    const uint32_t kv_dim  = 8 * 64;

    LFM2RoPELUT rope_lut;
    lfm2_rope_lut_init(&rope_lut, max_seq, 64, 1000000.0f);

    uint16_t *kc_avx[LFM2_MAX_LAYERS] = {0};
    uint16_t *vc_avx[LFM2_MAX_LAYERS] = {0};
    uint16_t *kc_sca[LFM2_MAX_LAYERS] = {0};
    uint16_t *vc_sca[LFM2_MAX_LAYERS] = {0};

    kc_avx[0] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));
    vc_avx[0] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));
    kc_sca[0] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));
    vc_sca[0] = (uint16_t *)lfm2_aligned_alloc_64(max_seq * kv_dim * sizeof(uint16_t));

    for (size_t i = 0; i < (size_t)max_seq * kv_dim; ++i) {
        kc_avx[0][i] = kc_sca[0][i] = (uint16_t)((i * 17) & 0x7FFF);
        vc_avx[0][i] = vc_sca[0][i] = (uint16_t)((i * 31) & 0x7FFF);
    }

    uint32_t pos_avx = 200, pos_sca = 200;
    const uint32_t n_sink = 16, n_discard = 64;

    lfm2_kv_cache_shift_sinks_avx2(kc_avx, vc_avx, &m, &rope_lut, &pos_avx, n_sink, n_discard);
    lfm2_kv_cache_shift_sinks_scalar(kc_sca, vc_sca, &m, &rope_lut, &pos_sca, n_sink, n_discard);

    assert(pos_avx == pos_sca);

    /* Verify Value caches match bit-for-bit */
    for (size_t i = 0; i < (size_t)pos_avx * kv_dim; ++i) {
        assert(vc_avx[0][i] == vc_sca[0][i]);
    }

    /* Verify Key caches match within FP16 rounding */
    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)pos_avx * kv_dim; ++i) {
        float f_avx = fp16_to_fp32(kc_avx[0][i]);
        float f_sca = fp16_to_fp32(kc_sca[0][i]);
        float d = fabsf(f_avx - f_sca);
        if (d > max_diff) max_diff = d;
    }
    printf("  AVX2 vs Scalar Shift Max Diff: %.6f\n", max_diff);
    assert(max_diff < 1e-3f);

    /* Verify forced scalar execution via LFM2_FORCE_ISA=scalar */
    setenv("LFM2_FORCE_ISA", "scalar", 1);
    uint32_t pos_forced = 120;
    lfm2_kv_cache_shift_sinks(kc_sca, vc_sca, &m, &rope_lut, &pos_forced, 8, 32);
    assert(pos_forced == 88);
    unsetenv("LFM2_FORCE_ISA");

    free(kc_avx[0]); free(vc_avx[0]); free(kc_sca[0]); free(vc_sca[0]);
    lfm2_rope_lut_free(&rope_lut);
    printf("PASSED: KV-cache shift scalar fallback verified with parity.\n");
}

/* ============================================================================
 * Test 7: Full Autoregressive Decode Loop Simulation
 * ============================================================================ */
static void test_end_to_end_runtime_control_decode_simulation(void) {
    printf("[Test 7/7] Verifying Full Autoregressive Decode Loop Runtime Integration...\n");

    const char *vocab_strs[] = {
        "<|im_end|>", "<think>", "</think>", "\n",
        "{", "}", ":", "\"", "answer", "42", " ",
        "reasoning_token_a", "reasoning_token_b"
    };
    const uint32_t V = sizeof(vocab_strs) / sizeof(vocab_strs[0]);

    LFM2DecodedVocab dv;
    lfm2_decoded_vocab_init(&dv, (char **)vocab_strs, V);

    /* 1. Initialize Runtime Controllers */
    const uint32_t close_seq[3] = { 3 /* \n */, 2 /* </think> */, 3 /* \n */ };
    LFM2ThinkController think_ctrl;
    lfm2_think_controller_init(&think_ctrl, 4 /* budget: 4 think tokens */, close_seq, 3);

    LFM2Grammar grammar;
    lfm2_grammar_init(&grammar, LFM2_GRAMMAR_JSON_OBJECT);

    float logits[32];
    uint32_t pos = 0;
    const uint32_t max_seq_len = 64;

    /* Simulation script:
     * 1. <think>
     * 2-5. 4 reasoning tokens (exhausts budget)
     * 6-8. forced \n </think> \n
     * 9-15. JSON response: { "answer": 42 }
     * 16. <|im_end|> */
    const uint32_t script[] = {
        1,  /* <think> */
        11, /* reasoning_token_a */
        12, /* reasoning_token_b */
        11, /* reasoning_token_a */
        12, /* reasoning_token_b -> hits budget! */
        3,  /* \n (forced) */
        2,  /* </think> (forced) */
        3,  /* \n (forced) */
        4,  /* { */
        7,  /* " */
        8,  /* answer */
        7,  /* " */
        6,  /* : */
        9,  /* 42 */
        5,  /* } */
        0   /* <|im_end|> */
    };

    for (size_t step = 0; step < sizeof(script) / sizeof(script[0]); ++step) {
        /* A. Context Capacity Check (dummy test) */
        if (pos >= max_seq_len) {
            pos -= 16;
        }

        /* B. Initialize candidate logits */
        for (uint32_t i = 0; i < V; ++i) logits[i] = 1.0f;

        /* C. Apply Think Budget Mask */
        lfm2_think_apply_budget_mask(&think_ctrl, logits, V);

        /* D. Apply Grammar Mask if outside thinking phase */
        if (think_ctrl.phase == LFM2_THINK_ANSWERING) {
            lfm2_grammar_mask_logits(&grammar, &dv, logits);
        }

        /* E. Pick token from script */
        uint32_t next_id = script[step];

        /* Verify next_id is legal according to active masks */
        assert(!isinf(logits[next_id]));

        /* F. Commit to controllers */
        lfm2_think_commit_token(&think_ctrl, &dv, next_id);
        if (think_ctrl.phase == LFM2_THINK_ANSWERING) {
            lfm2_grammar_commit_token(&grammar, &dv, next_id);
        }

        pos++;
    }

    assert(think_ctrl.phase == LFM2_THINK_ANSWERING);
    assert(grammar.state == G_COMPLETE);

    lfm2_decoded_vocab_free(&dv);
    printf("PASSED: Full autoregressive runtime control loop verified end-to-end.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 Runtime Control Features Tests  \n");
    printf("=======================================================\n");

    test_decoded_vocab_bpe_mapping();
    test_json_grammar_pda();
    test_tool_call_grammar_pda();
    test_think_budget_controller();
    test_kv_cache_shift_sinks_parity_and_rope();
    test_kv_cache_shift_scalar_vs_avx2_parity();
    test_end_to_end_runtime_control_decode_simulation();

    printf("=======================================================\n");
    printf("ALL 7 RUNTIME CONTROL TESTS PASSED WITH 100%% SUCCESS!  \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
