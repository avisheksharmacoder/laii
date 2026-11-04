#ifndef LFM_TOKENIZER_H
#define LFM_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "gguf.h"

typedef struct {
    char *token;
    float score;
    uint32_t id;
} TokenEntry;

typedef struct {
    uint32_t vocab_size;
    TokenEntry *entries;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;
} LFMTokenizer;

// Initialize tokenizer from parsed GGUF metadata
bool lfm_tokenizer_load(LFMTokenizer *tok, const GGUFFile *file);

// Free tokenizer allocations
void lfm_tokenizer_free(LFMTokenizer *tok);

// Decode token ID to string representation
const char* lfm_tokenizer_decode_id(const LFMTokenizer *tok, uint32_t id);

// Encode a prompt string into token IDs
// Returns the number of tokens encoded
size_t lfm_tokenizer_encode(
    const LFMTokenizer *tok,
    const char *text,
    uint32_t *out_tokens,
    size_t max_tokens,
    bool add_bos
);

#endif // LFM_TOKENIZER_H
