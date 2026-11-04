#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lfm_c_api.h"

static void stream_callback(uint32_t token_id, const char *piece, void *user_data) {
    (void)token_id;
    int *count = (int*)user_data;
    (*count)++;
    if (piece) {
        // piece received successfully
    }
}

int main(void) {
    printf("[Test] Running C ABI lifecycle, metadata, tokenizer & generation tests...\n");

    const char *model_path = "tests/lfm2_toy.gguf";
    lfm_engine_t *engine = lfm_create(model_path, 2);
    assert(engine != NULL);
    printf("  Engine handle created successfully via C ABI.\n");

    // 1. Metadata getters
    uint32_t hidden_dim = lfm_get_hidden_dim(engine);
    uint32_t num_layers = lfm_get_num_layers(engine);
    uint32_t vocab_size = lfm_get_vocab_size(engine);
    const char *arch = lfm_get_architecture(engine);

    printf("  Model Metadata: arch=%s, layers=%u, hidden_dim=%u, vocab=%u\n",
           arch, num_layers, hidden_dim, vocab_size);

    assert(hidden_dim == 64);
    assert(num_layers == 2);
    assert(vocab_size == 32);
    assert(strcmp(arch, "lfm2") == 0);

    // 2. Tokenize
    int32_t tokens[16];
    int n_tokens = lfm_tokenize(engine, "Liquid Foundation Models", tokens, 16);
    assert(n_tokens == 4);
    assert(tokens[0] == 1); // BOS
    assert(tokens[1] == 3); // "Liquid"
    assert(tokens[2] == 4); // " Foundation"
    assert(tokens[3] == 5); // " Models"
    printf("  Tokenization verified: 4 tokens [1, 3, 4, 5].\n");

    // 3. Detokenize
    char piece_buf[64];
    int bytes = lfm_detokenize(engine, 3, piece_buf, sizeof(piece_buf));
    assert(bytes > 0);
    assert(strcmp(piece_buf, "Liquid") == 0);
    printf("  Detokenize token 3 -> '%s' verified.\n", piece_buf);

    // 4. Synchronous generation
    char out_buf[512];
    lfm_sampler_params_t params = {
        .temperature = 0.7f,
        .top_k = 20,
        .max_new_tokens = 8,
        .seed = 1234,
        .echo_prompt = false
    };
    int n_gen = lfm_generate(engine, "Liquid Foundation Models", out_buf, sizeof(out_buf), &params);
    assert(n_gen > 0);
    assert(strlen(out_buf) > 0);
    printf("  Generated %d tokens synchronously: \"%s\"\n", n_gen, out_buf);

    // 5. Streaming generation
    int stream_count = 0;
    int n_stream = lfm_generate_stream(engine, "Liquid", &params, stream_callback, &stream_count);
    assert(n_stream > 0);
    assert(stream_count == n_stream);
    printf("  Streamed %d tokens via C callback.\n", stream_count);

    // 6. Clean destruction
    lfm_destroy(engine);
    printf("  Engine cleanly destroyed via C ABI.\n");

    printf("PASSED: C ABI tests successful!\n");
    return 0;
}
