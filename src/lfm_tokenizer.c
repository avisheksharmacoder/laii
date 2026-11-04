#define _POSIX_C_SOURCE 200809L
#include "lfm_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool lfm_tokenizer_load(LFMTokenizer *tok, const GGUFFile *file) {
    if (!tok || !file) return false;
    memset(tok, 0, sizeof(LFMTokenizer));

    tok->bos_id = gguf_get_u32(file, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = gguf_get_u32(file, "tokenizer.ggml.eos_token_id", 2);
    tok->pad_id = gguf_get_u32(file, "tokenizer.ggml.padding_token_id", 0);

    const GGUFValue *tokens_val = gguf_get_val(file, "tokenizer.ggml.tokens");
    if (!tokens_val || tokens_val->type != GGUF_TYPE_ARRAY || tokens_val->val_arr.type != GGUF_TYPE_STRING) {
        // Fallback: check models without explicit tokenizer in metadata
        tok->vocab_size = 0;
        return true;
    }

    tok->vocab_size = (uint32_t)tokens_val->val_arr.len;
    tok->entries = (TokenEntry*)calloc(tok->vocab_size, sizeof(TokenEntry));
    if (!tok->entries) return false;

    GGUFString *strs = (GGUFString*)tokens_val->val_arr.data;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        tok->entries[i].id = i;
        if (strs[i].data) {
            tok->entries[i].token = strdup(strs[i].data);
        } else {
            tok->entries[i].token = strdup("");
        }
    }

    // Load scores if available
    const GGUFValue *scores_val = gguf_get_val(file, "tokenizer.ggml.scores");
    if (scores_val && scores_val->type == GGUF_TYPE_ARRAY && scores_val->val_arr.type == GGUF_TYPE_FLOAT32) {
        const float *scores = (const float*)scores_val->val_arr.data;
        for (uint32_t i = 0; i < tok->vocab_size && i < scores_val->val_arr.len; ++i) {
            tok->entries[i].score = scores[i];
        }
    }

    return true;
}

void lfm_tokenizer_free(LFMTokenizer *tok) {
    if (!tok) return;
    if (tok->entries) {
        for (uint32_t i = 0; i < tok->vocab_size; ++i) {
            free(tok->entries[i].token);
        }
        free(tok->entries);
        tok->entries = NULL;
    }
    tok->vocab_size = 0;
}

const char* lfm_tokenizer_decode_id(const LFMTokenizer *tok, uint32_t id) {
    if (!tok || id >= tok->vocab_size || !tok->entries) return "";

    const char *str = tok->entries[id].token;
    if (!str) return "";

    // Check for byte token format: "<0xHH>"
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>' && str[6] == '\0') {
        static char byte_buf[2];
        char hex[3] = { str[3], str[4], '\0' };
        byte_buf[0] = (char)strtol(hex, NULL, 16);
        byte_buf[1] = '\0';
        return byte_buf;
    }

    return str;
}

static int find_token_exact(const LFMTokenizer *tok, const char *text, size_t len) {
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        if (tok->entries[i].token && strlen(tok->entries[i].token) == len) {
            if (strncmp(tok->entries[i].token, text, len) == 0) {
                return (int)i;
            }
        }
    }
    return -1;
}

size_t lfm_tokenizer_encode(
    const LFMTokenizer *tok,
    const char *text,
    uint32_t *out_tokens,
    size_t max_tokens,
    bool add_bos
) {
    if (!tok || !text || !out_tokens || max_tokens == 0) return 0;

    size_t count = 0;
    if (add_bos && count < max_tokens) {
        out_tokens[count++] = tok->bos_id;
    }

    size_t text_len = strlen(text);
    size_t i = 0;

    while (i < text_len && count < max_tokens) {
        // Greedy longest match
        int best_id = -1;
        size_t best_len = 0;

        size_t max_match = (text_len - i < 64) ? (text_len - i) : 64;
        for (size_t l = max_match; l >= 1; --l) {
            int id = find_token_exact(tok, text + i, l);
            if (id >= 0) {
                best_id = id;
                best_len = l;
                break;
            }
        }

        if (best_id >= 0) {
            out_tokens[count++] = (uint32_t)best_id;
            i += best_len;
        } else {
            // Byte fallback: try matching byte hex "<0xHH>"
            char byte_str[7];
            snprintf(byte_str, sizeof(byte_str), "<0x%02X>", (unsigned char)text[i]);
            int byte_id = find_token_exact(tok, byte_str, 6);
            if (byte_id >= 0) {
                out_tokens[count++] = (uint32_t)byte_id;
            }
            i++;
        }
    }

    return count;
}
