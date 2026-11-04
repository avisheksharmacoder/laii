#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lfm_tokenizer.h"
#include "gguf.h"

int main(void) {
    printf("[Test] Running BPE Tokenizer unit tests...\n");

    GGUFFile gguf;
    if (!gguf_open(&gguf, "tests/lfm2_toy.gguf")) {
        fprintf(stderr, "FAILED: Failed to open tests/lfm2_toy.gguf\n");
        return 1;
    }

    LFMTokenizer tok;
    if (!lfm_tokenizer_load(&tok, &gguf)) {
        fprintf(stderr, "FAILED: Failed to load tokenizer\n");
        gguf_close(&gguf);
        return 1;
    }

    assert(tok.vocab_size == 32);
    assert(tok.bos_id == 1);
    assert(tok.eos_id == 2);
    printf("  Tokenizer loaded: %u tokens (BOS=%u, EOS=%u)\n",
           tok.vocab_size, tok.bos_id, tok.eos_id);

    // 1. Decode tests
    const char *tok3 = lfm_tokenizer_decode_id(&tok, 3);
    assert(strcmp(tok3, "Liquid") == 0);
    printf("  Token 3 decoded: \"%s\"\n", tok3);

    const char *tok4 = lfm_tokenizer_decode_id(&tok, 4);
    assert(strcmp(tok4, " Foundation") == 0);
    printf("  Token 4 decoded: \"%s\"\n", tok4);

    // 2. Encode tests
    uint32_t tokens[16];
    size_t n = lfm_tokenizer_encode(&tok, "Liquid Foundation Models", tokens, 16, true);
    // Expected: BOS(1), Liquid(3), Foundation(4), Models(5)
    assert(n == 4);
    assert(tokens[0] == 1);
    assert(tokens[1] == 3);
    assert(tokens[2] == 4);
    assert(tokens[3] == 5);
    printf("  Encode 'Liquid Foundation Models' -> [%u, %u, %u, %u] verified.\n",
           tokens[0], tokens[1], tokens[2], tokens[3]);

    lfm_tokenizer_free(&tok);
    gguf_close(&gguf);

    printf("PASSED: Tokenizer unit tests successful!\n");
    return 0;
}
