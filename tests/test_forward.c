#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lfm_engine.h"

static void dummy_token_cb(uint32_t token_id, const char *piece, void *user_data) {
    (void)user_data;
    printf("[Token %u: \"%s\"] ", token_id, piece ? piece : "");
}

int main(void) {
    printf("[Test] Running end-to-end forward pass on toy LFM2 model...\n");

    LFMEngine engine;
    if (!lfm_engine_init(&engine, "tests/lfm2_toy.gguf", 2)) {
        fprintf(stderr, "FAILED: Failed to init engine on tests/lfm2_toy.gguf\n");
        return 1;
    }

    const LFMConfig *cfg = &engine.model.config;
    printf("  Model: %u layers (%u conv, %u attn), hidden_dim=%u, vocab_size=%u\n",
           cfg->n_layers,
           cfg->is_conv_layer[0] + cfg->is_conv_layer[1],
           !cfg->is_conv_layer[0] + !cfg->is_conv_layer[1],
           cfg->hidden_dim,
           cfg->vocab_size);

    LFMSamplerConfig sampler = {
        .temperature = 0.0f, // greedy
        .top_k = 1,
        .max_new_tokens = 8,
        .seed = 42,
        .echo_prompt = false
    };

    printf("  Generating 8 tokens for prompt 'Liquid Foundation Models'...\n  Generated: ");
    size_t count = lfm_engine_generate(&engine, "Liquid Foundation Models", &sampler, dummy_token_cb, NULL);
    printf("\n");

    printf("  Tokens generated: %zu\n", count);
    if (count != 8) {
        fprintf(stderr, "FAILED: Expected 8 generated tokens, got %zu\n", count);
        lfm_engine_free(&engine);
        return 1;
    }

    lfm_engine_free(&engine);
    printf("PASSED: End-to-end forward pass test successful!\n");
    return 0;
}
