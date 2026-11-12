#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define LFM2_LOADER_NO_MAIN 1
#if __has_include("../tools/lfm2_bpe.c")
#include "../tools/lfm2_bpe.c"
#elif __has_include("tools/lfm2_bpe.c")
#include "tools/lfm2_bpe.c"
#endif

/* Helper to setup a controlled synthetic BPE vocabulary */
static void setup_mock_bpe_vocab(LFM2Vocab *v) {
    memset(v, 0, sizeof(*v));
    v->id_pad      = 0;
    v->id_bos      = 1;
    v->id_im_start = 2;
    v->id_im_end   = 3;

    /* Vocabulary tokens */
    const char *tok_list[] = {
        "<|pad|>", "<|startoftext|>", "<|im_start|>", "<|im_end|>",
        "H", "e", "l", "o",
        "He", "ll", "llo", "Hello",
        "a", "aa",
        "Ġ", "Ġworld", "world"
    };
    uint32_t n_tok = sizeof(tok_list) / sizeof(tok_list[0]);
    v->n_vocab = n_tok;
    v->tokens = (char **)calloc(n_tok, sizeof(char *));
    v->token_lens = (uint32_t *)calloc(n_tok, sizeof(uint32_t));
    v->str_ht_cap = 512;
    v->str_ht = (TokenStrSlot *)calloc(v->str_ht_cap, sizeof(TokenStrSlot));
    for (uint32_t i = 0; i < v->str_ht_cap; ++i) v->str_ht[i].token_id = UINT32_MAX;

    for (uint32_t i = 0; i < n_tok; ++i) {
        v->tokens[i] = strdup(tok_list[i]);
        v->token_lens[i] = (uint32_t)strlen(tok_list[i]);
        vocab_str_ht_insert(v, v->tokens[i], v->token_lens[i], i);
    }

    /* Initialize byte_to_id lookup table */
    for (int b = 0; b < 256; ++b) v->byte_to_id[b] = UINT32_MAX;
    v->byte_to_id['H'] = vocab_lookup_str(v, "H", 1);
    v->byte_to_id['e'] = vocab_lookup_str(v, "e", 1);
    v->byte_to_id['l'] = vocab_lookup_str(v, "l", 1);
    v->byte_to_id['o'] = vocab_lookup_str(v, "o", 1);
    v->byte_to_id['a'] = vocab_lookup_str(v, "a", 1);
    v->byte_to_id['w'] = UINT32_MAX; /* simulate partial */
    /* 0x20 space maps to Ġ (UTF-8: 0xC4 0xA0) */
    v->byte_to_id[' '] = vocab_lookup_str(v, "Ġ", 2);

    /* Specials */
    v->n_specials = 4;
    for (uint32_t i = 0; i < 4; ++i) {
        v->specials[i] = (SpecialToken){
            .str = v->tokens[i], .len = v->token_lens[i], .id = i
        };
    }

    /* Merges in specific rank order:
     * Rank 0: 'l' + 'l' -> 'll'
     * Rank 1: 'H' + 'e' -> 'He'
     * Rank 2: 'll' + 'o' -> 'llo'
     * Rank 3: 'He' + 'llo' -> 'Hello'
     * Rank 4: 'a' + 'a' -> 'aa'
     */
    v->merge_ht_cap = 512;
    v->merge_ht = (BPEMergeSlot *)calloc(v->merge_ht_cap, sizeof(BPEMergeSlot));
    for (uint32_t i = 0; i < v->merge_ht_cap; ++i) v->merge_ht[i].pair_key = UINT64_MAX;

    vocab_merge_ht_insert(v, v->byte_to_id['l'], v->byte_to_id['l'], 0, vocab_lookup_str(v, "ll", 2));
    vocab_merge_ht_insert(v, v->byte_to_id['H'], v->byte_to_id['e'], 1, vocab_lookup_str(v, "He", 2));
    vocab_merge_ht_insert(v, vocab_lookup_str(v, "ll", 2), v->byte_to_id['o'], 2, vocab_lookup_str(v, "llo", 3));
    vocab_merge_ht_insert(v, vocab_lookup_str(v, "He", 2), vocab_lookup_str(v, "llo", 3), 3, vocab_lookup_str(v, "Hello", 5));
    vocab_merge_ht_insert(v, v->byte_to_id['a'], v->byte_to_id['a'], 4, vocab_lookup_str(v, "aa", 2));
    v->n_merges = 5;
}

static void test_exact_bpe_merging(void) {
    printf("[Test 1/5] Verifying exact priority-ordered BPE rank merging...\n");
    LFM2Vocab v;
    setup_mock_bpe_vocab(&v);

    uint32_t out[16];
    /* "Hello" with bytes 'H', 'e', 'l', 'l', 'o' */
    uint32_t n = bpe_encode_word(&v, (const uint8_t *)"Hello", 5, out, 16);
    assert(n == 1);
    assert(out[0] == vocab_lookup_str(&v, "Hello", 5));
    printf("  Encoded 'Hello' into single token ID %u (\"%s\") successfully.\n",
           out[0], v.tokens[out[0]]);

    /* "aaaa" should merge leftmost 'aa' + 'aa' -> 2 tokens of 'aa' */
    n = bpe_encode_word(&v, (const uint8_t *)"aaaa", 4, out, 16);
    assert(n == 2);
    assert(out[0] == vocab_lookup_str(&v, "aa", 2));
    assert(out[1] == vocab_lookup_str(&v, "aa", 2));
    printf("  Encoded 'aaaa' into 2 'aa' tokens successfully (leftmost tie-breaking verified).\n");

    lfm2_vocab_free(&v);
    printf("PASSED: Priority-ordered BPE merge logic verified.\n");
}

static void test_gpt2_byte_mapping(void) {
    printf("[Test 2/5] Verifying GPT-2 Byte-to-Unicode mapping and round-trip detokenizer...\n");
    /* Verify all 256 bytes map to valid 1-byte or 2-byte UTF-8 sequences and round-trip */
    char *one_tok = NULL;
    LFM2Vocab roundtrip_v = {0};
    roundtrip_v.n_vocab = 1;
    roundtrip_v.tokens = &one_tok;

    for (int b = 0; b < 256; ++b) {
        char u8[4];
        size_t len = gpt2_single_byte_to_utf8((uint8_t)b, u8);
        assert(len == 1 || len == 2);
        assert(u8[len] == '\0');

        if (b == 0x20) {
            /* Space must map to Ġ (\xc4\xa0) */
            assert(len == 2);
            assert((uint8_t)u8[0] == 0xC4 && (uint8_t)u8[1] == 0xA0);
        } else if (b == 0x0A) {
            /* Newline must map to Ċ (\xc4\x8a) */
            assert(len == 2);
            assert((uint8_t)u8[0] == 0xC4 && (uint8_t)u8[1] == 0x8A);
        } else if (b >= 33 && b <= 126) {
            /* ASCII printable characters map directly */
            assert(len == 1);
            assert((uint8_t)u8[0] == (uint8_t)b);
        }

        one_tok = u8;
        char raw_out[4];
        size_t rlen = lfm2_detokenize_token(&roundtrip_v, 0, raw_out, sizeof(raw_out));
        assert(rlen == 1 && (uint8_t)raw_out[0] == (uint8_t)b);
    }

    /* Test detokenize on space, newline, and multi-byte UTF-8 sequences */
    LFM2Vocab v;
    memset(&v, 0, sizeof(v));
    const char *toks[] = {
        "<pad>", "Ġ", "Ċ", "Hello", "Ġworld",
        "\xC3\xA2\xC4\xA2\xC4\xB5",         /* âĢĵ -> en-dash "–" (\xE2\x80\x93) */
        "\xC4\xA0\xC3\xA2\xC4\xA2\xC4\xB5"  /* ĠâĢĵ -> " –" */
    };
    v.n_vocab = 7;
    v.tokens = (char **)calloc(v.n_vocab, sizeof(char *));
    for (uint32_t i = 0; i < v.n_vocab; ++i) v.tokens[i] = strdup(toks[i]);

    char decoded[32];
    /* Token 1: Ġ -> ' ' */
    size_t dlen = lfm2_detokenize_token(&v, 1, decoded, sizeof(decoded));
    assert(dlen == 1 && decoded[0] == ' ');

    /* Token 2: Ċ -> '\n' */
    dlen = lfm2_detokenize_token(&v, 2, decoded, sizeof(decoded));
    assert(dlen == 1 && decoded[0] == '\n');

    /* Token 3: Hello -> Hello */
    dlen = lfm2_detokenize_token(&v, 3, decoded, sizeof(decoded));
    assert(dlen == 5 && strcmp(decoded, "Hello") == 0);

    /* Token 4: Ġworld -> " world" */
    dlen = lfm2_detokenize_token(&v, 4, decoded, sizeof(decoded));
    assert(dlen == 6 && strcmp(decoded, " world") == 0);

    /* Token 5: âĢĵ -> "–" */
    dlen = lfm2_detokenize_token(&v, 5, decoded, sizeof(decoded));
    assert(dlen == 3 && strcmp(decoded, "\xE2\x80\x93") == 0);

    /* Token 6: ĠâĢĵ -> " –" */
    dlen = lfm2_detokenize_token(&v, 6, decoded, sizeof(decoded));
    assert(dlen == 4 && strcmp(decoded, " \xE2\x80\x93") == 0);

    lfm2_vocab_free(&v);
    printf("PASSED: GPT-2 Byte-to-Unicode mapping and detokenization verified.\n");
}

static void test_pretokenizer_regex(void) {
    printf("[Test 3/5] Verifying GPT-2 pre-tokenizer regex state machine...\n");
    const char *text = "We're 123456 testing, 'll right!";
    size_t len = strlen(text);
    size_t off = 0;

    /* Piece 1: "We" */
    size_t p1 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p1 == 2);
    off += p1;

    /* Piece 2: "'re" (contraction) */
    size_t p2 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p2 == 3);
    off += p2;

    /* Piece 3: " 123" (space + up to 3 digits) */
    size_t p3 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p3 == 4);
    off += p3;

    /* Piece 4: "456" (up to 3 digits) */
    size_t p4 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p4 == 3);
    off += p4;

    /* Piece 5: " testing" (space + letters) */
    size_t p5 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p5 == 8);
    off += p5;

    /* Piece 6: "," (punctuation) */
    size_t p6 = next_gpt2_pretoken_len((const uint8_t *)(text + off), len - off);
    assert(p6 == 1);
    off += p6;

    printf("PASSED: GPT-2 pre-tokenization piece splitting verified.\n");
}

static void test_special_tokens_preservation(void) {
    printf("[Test 4/5] Verifying special control tokens pre-splitting...\n");
    LFM2Vocab v;
    setup_mock_bpe_vocab(&v);

    const char *prompt = "<|startoftext|><|im_start|>Hello<|im_end|>";
    uint32_t out[16];
    uint32_t n = lfm2_tokenize(&v, prompt, out, 16);

    assert(n == 4);
    assert(out[0] == v.id_bos);
    assert(out[1] == v.id_im_start);
    assert(out[2] == vocab_lookup_str(&v, "Hello", 5));
    assert(out[3] == v.id_im_end);

    printf("  Tokenized '%s' -> [%u, %u, %u, %u] with exact boundary tags.\n",
           prompt, out[0], out[1], out[2], out[3]);

    lfm2_vocab_free(&v);
    printf("PASSED: Special token pre-splitting verified.\n");
}

static void test_toy_gguf_loader_compatibility(void) {
    printf("[Test 5/5] Verifying backward-compatibility loader on tests/lfm2_toy.gguf...\n");

    LFM2Model model;
    if (!lfm2_load_gguf("tests/lfm2_toy.gguf", &model)) {
        fprintf(stderr, "FAILED: Could not load tests/lfm2_toy.gguf\n");
        exit(1);
    }

    LFM2Vocab vocab;
    if (!lfm2_vocab_load(&vocab, &model)) {
        fprintf(stderr, "FAILED: lfm2_vocab_load failed on tests/lfm2_toy.gguf\n");
        exit(1);
    }

    assert(vocab.n_vocab == 32);
    assert(vocab.id_bos == 1);
    assert(vocab.id_im_end == 2);
    printf("  GGUF Vocab: %u tokens, BOS=%u, EOS=%u\n",
           vocab.n_vocab, vocab.id_bos, vocab.id_im_end);

    /* Test tokenization on prompt containing vocabulary words */
    const char *test_prompt = "Liquid Foundation Models are fast";
    uint32_t out[16];
    uint32_t n = lfm2_tokenize(&vocab, test_prompt, out, 16);
    assert(n >= 4);
    printf("  Tokenized '%s' into %u tokens successfully.\n", test_prompt, n);

    lfm2_vocab_free(&vocab);
    lfm2_free_model(&model);
    printf("PASSED: GGUF model loader and fallback prefix tokenization verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI Exact GGUF BPE Tokenizer Test Suite   \n");
    printf("=======================================================\n");

    test_exact_bpe_merging();
    test_gpt2_byte_mapping();
    test_pretokenizer_regex();
    test_special_tokens_preservation();
    test_toy_gguf_loader_compatibility();

    printf("=======================================================\n");
    printf("ALL 5 BPE TOKENIZER UNIT TESTS PASSED WITH 100%% SUCCESS!\n");
    printf("=======================================================\n");
    return 0;
}
