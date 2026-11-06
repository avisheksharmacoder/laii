#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "include/lfm2_speculative_pld.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define LFM2_SPECULATIVE_PLD_NO_MAIN 1
#include "tools/lfm2_speculative_pld.c"

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static uint32_t greedy_sampler(float *logits, uint32_t n_vocab, void *user_data) {
    (void)user_data;
    uint32_t best_id = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI Prompt-Lookup Speculative TDD Suite  \n");
    printf("=======================================================\n");

    /* ------------------------------------------------------------------------
     * Test 1: N-gram Index Lifecycle, Hashing & Exact Retrieval
     * ------------------------------------------------------------------------ */
    printf("[Test 1/5] Verifying 3-Gram & 2-Gram Hash Index Precision...\n");
    LFM2NgramIndex idx;
    TEST_ASSERT(lfm2_ngram_index_init(&idx, 256, 6, 2)); /* EOS=2 */
    TEST_ASSERT(idx.history_len == 0);
    TEST_ASSERT(idx.cur_draft_k == 6);

    /* Ingest sequence: [1, 10, 20, 30, 40, 50, 60, 10, 20] */
    const uint32_t seq[] = {1, 10, 20, 30, 40, 50, 60, 10, 20};
    for (size_t i = 0; i < sizeof(seq)/sizeof(seq[0]); ++i) {
        lfm2_ngram_index_ingest(&idx, seq[i]);
    }
    TEST_ASSERT(idx.history_len == 9);

    /* Query 3-gram draft when pending_token is 30 (preceded by [10, 20]) */
    uint32_t drafts[LFM2_PLD_MAX_DRAFT];
    uint32_t n_draft = lfm2_ngram_draft(&idx, 30, drafts);
    TEST_ASSERT(n_draft >= 3);
    TEST_ASSERT(drafts[0] == 40);
    TEST_ASSERT(drafts[1] == 50);
    TEST_ASSERT(drafts[2] == 60);
    printf("  Draft query for [10, 20] + 30 -> drafted %u tokens: [%u, %u, %u]\n",
           n_draft, drafts[0], drafts[1], drafts[2]);
    printf("PASSED: 3-Gram & 2-Gram retrieval verified.\n");

    /* ------------------------------------------------------------------------
     * Test 2: Stop Token Protection & Adaptive Window Dynamics
     * ------------------------------------------------------------------------ */
    printf("[Test 2/5] Verifying Stop Token Boundary & Adaptive Windowing...\n");
    lfm2_ngram_index_reset(&idx);
    /* Ingest prompt with EOS: [1, 5, 6, 7, 2, 8, 9] where 2 is EOS.
     * Then append context ending with token 5 so pending_token 6 matches (5, 6). */
    const uint32_t eos_seq[] = {1, 5, 6, 7, 2, 8, 9, 100, 5};
    for (size_t i = 0; i < sizeof(eos_seq)/sizeof(eos_seq[0]); ++i) {
        lfm2_ngram_index_ingest(&idx, eos_seq[i]);
    }
    /* Pending token 6 preceded by 5 -> matches 2-gram (5, 6) -> must draft 7 and 2, but stop before 8 */
    n_draft = lfm2_ngram_draft(&idx, 6, drafts);
    TEST_ASSERT(n_draft == 2);
    TEST_ASSERT(drafts[0] == 7);
    TEST_ASSERT(drafts[1] == 2); /* EOS terminates draft */

    /* Verify adaptive window throttle on consecutive zero-accepts */
    lfm2_ngram_update_adaptive_window(&idx, 4, 0);
    lfm2_ngram_update_adaptive_window(&idx, 4, 0);
    lfm2_ngram_update_adaptive_window(&idx, 4, 0);
    TEST_ASSERT(idx.cur_draft_k == 2); /* Throttled to 2 */

    /* Verify window restore on hit */
    lfm2_ngram_update_adaptive_window(&idx, 2, 2);
    TEST_ASSERT(idx.cur_draft_k == idx.max_draft_k); /* Restored to 6 */
    printf("  Adaptive window: throttled to %u on misses, restored to %u on hit.\n", 2, idx.cur_draft_k);
    lfm2_ngram_index_free(&idx);
    printf("PASSED: Stop token protection and adaptive windowing verified.\n");

    /* ------------------------------------------------------------------------
     * Test 3: LFM2SpecState Arena & ShortConv Trace Rollback Precision
     * ------------------------------------------------------------------------ */
    printf("[Test 3/5] Verifying LFM2SpecState Arena & conv_trace Rollback...\n");
    LFM2Model model;
    memset(&model, 0, sizeof(model));
    TEST_ASSERT(lfm2_load_gguf("tests/lfm2_toy.gguf", &model));

    LFM2SpecState spec_state;
    TEST_ASSERT(lfm2_spec_state_init(&spec_state, &model, 128, 2));
    TEST_ASSERT(spec_state.logits_batch != NULL);

    /* Verify conv_trace slots are non-NULL for ShortConv layers */
    for (uint32_t b = 0; b < LFM2_MAX_BATCH; ++b) {
        for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
            if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
                TEST_ASSERT(spec_state.conv_trace[b][l] != NULL);
            }
        }
    }

    /* Seed slot 2 with known test pattern */
    for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            spec_state.conv_trace[2][l][0] = 42.5f;
            spec_state.conv_trace[2][l][1] = 84.25f;
        }
    }

    /* Roll back to slot 2 and verify state restoration */
    lfm2_restore_shortconv_trace_slot(&spec_state, &model, 2);
    for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            TEST_ASSERT(spec_state.base.conv_state[l][0] == 42.5f);
            TEST_ASSERT(spec_state.base.conv_state[l][1] == 84.25f);
        }
    }
    printf("  conv_trace[2] successfully snapped into live ShortConv state.\n");
    printf("PASSED: conv_trace rollback verified.\n");

    /* ------------------------------------------------------------------------
     * Test 4: Verification Batch Forward Pass on Real Checkpoint
     * ------------------------------------------------------------------------ */
    printf("[Test 4/5] Verifying Speculative Batched Forward Pass on GGUF Checkpoint...\n");
    const uint32_t batch_tokens[4] = {1, 3, 4, 5};
    float *batch_logits = lfm2_forward_speculative_batch(&model, &spec_state, batch_tokens, 4, 0);
    TEST_ASSERT(batch_logits != NULL);

    /* Check that all 4 batch slots produced finite logits across all vocab tokens */
    for (uint32_t b = 0; b < 4; ++b) {
        float *slot_l = batch_logits + (size_t)b * model.cfg.n_vocab;
        for (uint32_t v = 0; v < model.cfg.n_vocab; ++v) {
            TEST_ASSERT(!isnan(slot_l[v]) && !isinf(slot_l[v]));
        }
    }
    printf("  Batch forward pass verified: 4 slots x %u vocab logits computed.\n", model.cfg.n_vocab);
    printf("PASSED: Speculative batch forward pass verified.\n");

    /* ------------------------------------------------------------------------
     * Test 5: End-to-End Sequence Parity vs. Autoregressive M=1 Baseline
     * ------------------------------------------------------------------------ */
    printf("[Test 5/5] Verifying 100%% Bit-Exact Parity vs. M=1 Autoregressive Decode...\n");
    LFM2SpecState base_state;
    TEST_ASSERT(lfm2_spec_state_init(&base_state, &model, 128, 2));

    const uint32_t num_steps = 12;
    uint32_t base_tokens[16];
    uint32_t pos_base = 0;
    uint32_t tok_base = 1; /* BOS */

    /* Generate baseline sequence */
    for (uint32_t s = 0; s < num_steps; ++s) {
        float *logits = lfm2_forward_batch(&model, &base_state.base, &tok_base, 1, pos_base++, true);
        tok_base = greedy_sampler(logits, model.cfg.n_vocab, NULL);
        base_tokens[s] = tok_base;
    }

    /* Reset model state for speculative decode */
    lfm2_spec_state_free(&spec_state);
    TEST_ASSERT(lfm2_spec_state_init(&spec_state, &model, 128, 2));

    LFM2NgramIndex spec_ngram;
    TEST_ASSERT(lfm2_ngram_index_init(&spec_ngram, 128, 4, 2));

    /* Pre-seed N-gram history with prompt context containing part of the sequence */
    lfm2_ngram_index_ingest(&spec_ngram, 1);
    for (uint32_t s = 0; s < 8; ++s) {
        lfm2_ngram_index_ingest(&spec_ngram, base_tokens[s]);
    }

    uint32_t spec_tokens[16];
    uint32_t n_spec_emitted = 0;
    uint32_t pos_spec = 0;
    uint32_t tok_spec = 1; /* BOS */

    while (n_spec_emitted < num_steps) {
        uint32_t em[LFM2_MAX_BATCH];
        uint32_t cnt = lfm2_speculative_decode_step(
            &model, &spec_state, &spec_ngram, &tok_spec, &pos_spec,
            greedy_sampler, NULL, em);
        TEST_ASSERT(cnt >= 1);
        for (uint32_t i = 0; i < cnt && n_spec_emitted < num_steps; ++i) {
            spec_tokens[n_spec_emitted++] = em[i];
        }
    }

    /* Assert 100% bit-exact parity between baseline and speculative decode */
    for (uint32_t i = 0; i < num_steps; ++i) {
        TEST_ASSERT(base_tokens[i] == spec_tokens[i]);
    }
    printf("  Parity certified: all %u tokens match bit-for-bit with M=1 baseline!\n", num_steps);
    printf("PASSED: 100%% bit-exact speculative sequence parity certified.\n");

    lfm2_ngram_index_free(&spec_ngram);
    lfm2_spec_state_free(&base_state);
    lfm2_spec_state_free(&spec_state);
    lfm2_free_model(&model);

    printf("=======================================================\n");
    printf("ALL 5 SPECULATIVE PLD TESTS PASSED WITH 100%% SUCCESS!  \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
