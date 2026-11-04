#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <unistd.h>

#define LFM2_CLI_STREAM_NO_MAIN 1
#include "tools/lfm2_cli_stream.c"

typedef struct {
    char   captured[4096];
    size_t len;
    int    emit_count;
} CaptureCtx;

static void test_emit_callback(const char *utf8_bytes, size_t len, void *user_data) {
    CaptureCtx *c = (CaptureCtx *)user_data;
    assert(c->len + len < sizeof(c->captured));
    memcpy(c->captured + c->len, utf8_bytes, len);
    c->len += len;
    c->captured[c->len] = '\0';
    c->emit_count++;
}

/* ----------------------------------------------------------------------------
 * Test 1: UTF-8 2-Byte, 3-Byte and 4-Byte Split Boundary Assembly
 * ---------------------------------------------------------------------------- */
static void test_utf8_split_assembly(void) {
    printf("[Test 1/6] Verifying UTF-8 2-Byte, 3-Byte & 4-Byte Split Assembly...\n");

    CaptureCtx cap = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, test_emit_callback, &cap);

    /* 1. Cyrillic 2-byte char 'Д' (0xD0 0x94) split across 2 tokens */
    const char tok_cyr1[] = { 'A', (char)0xD0 };
    const char tok_cyr2[] = { (char)0x94, 'B' };
    lfm2_stream_filter_feed(&sf, tok_cyr1, sizeof(tok_cyr1));
    assert(strcmp(cap.captured, "A") == 0); /* 0xD0 held back */
    lfm2_stream_filter_feed(&sf, tok_cyr2, sizeof(tok_cyr2));
    assert(strcmp(cap.captured, "AДB") == 0);

    /* 2. Japanese Kanji 3-byte char '気' (0xE6 0xB0 0x97) split 1 byte per token */
    const char tok_kanji1[] = { (char)0xE6 };
    const char tok_kanji2[] = { (char)0xB0 };
    const char tok_kanji3[] = { (char)0x97, '!' };
    lfm2_stream_filter_feed(&sf, tok_kanji1, sizeof(tok_kanji1));
    assert(strcmp(cap.captured, "AДB") == 0); /* 0xE6 held back */
    lfm2_stream_filter_feed(&sf, tok_kanji2, sizeof(tok_kanji2));
    assert(strcmp(cap.captured, "AДB") == 0); /* 0xE6 0xB0 held back */
    lfm2_stream_filter_feed(&sf, tok_kanji3, sizeof(tok_kanji3));
    assert(strcmp(cap.captured, "AДB気!") == 0);

    /* 3. Rocket Emoji 4-byte char '🚀' (0xF0 0x9F 0x9A 0x80) split 1 byte per token */
    const char tok_r1[] = { ' ', (char)0xF0 };
    const char tok_r2[] = { (char)0x9F };
    const char tok_r3[] = { (char)0x9A };
    const char tok_r4[] = { (char)0x80, ' ' };
    lfm2_stream_filter_feed(&sf, tok_r1, sizeof(tok_r1));
    assert(strcmp(cap.captured, "AДB気! ") == 0);
    lfm2_stream_filter_feed(&sf, tok_r2, sizeof(tok_r2));
    assert(strcmp(cap.captured, "AДB気! ") == 0);
    lfm2_stream_filter_feed(&sf, tok_r3, sizeof(tok_r3));
    assert(strcmp(cap.captured, "AДB気! ") == 0);
    lfm2_stream_filter_feed(&sf, tok_r4, sizeof(tok_r4));
    assert(strcmp(cap.captured, "AДB気! 🚀 ") == 0);

    lfm2_stream_filter_flush(&sf);
    assert(strcmp(cap.captured, "AДB気! 🚀 ") == 0);

    printf("PASSED: UTF-8 split boundary assembly verified across 2, 3, and 4 byte codepoints.\n");
}

/* ----------------------------------------------------------------------------
 * Test 2: Multi-Token Stop String Matching & Suffix Stripping
 * ---------------------------------------------------------------------------- */
static void test_multi_token_stop_string(void) {
    printf("[Test 2/6] Verifying Multi-Token Stop String Matching & Truncation...\n");

    CaptureCtx cap = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, test_emit_callback, &cap);

    assert(lfm2_stream_filter_add_stop(&sf, "\nUser:"));
    assert(lfm2_stream_filter_add_stop(&sf, "<|im_end|>"));
    assert(lfm2_stream_filter_add_stop(&sf, "```"));

    /* Feed tokens that end with a multi-token stop string "\nUser:" split across 3 tokens */
    const char *tokens[] = {
        "This is ",
        "the AI response.",
        "\n",
        "User",
        ": Tell me more",
        "Should not be seen"
    };

    lfm2_stream_status_t st = LFM2_STREAM_OK;
    for (size_t i = 0; i < sizeof(tokens)/sizeof(tokens[0]); ++i) {
        st = lfm2_stream_filter_feed(&sf, tokens[i], strlen(tokens[i]));
        if (st == LFM2_STREAM_STOPPED) {
            break;
        }
    }

    assert(st == LFM2_STREAM_STOPPED);
    assert(sf.matched_stop_idx == 0);
    assert(strcmp(cap.captured, "This is the AI response.") == 0);

    printf("PASSED: Multi-token stop string correctly matched and stripped.\n");
}

/* ----------------------------------------------------------------------------
 * Test 3: False-Positive Partial Stop Prefix Divergence
 * ---------------------------------------------------------------------------- */
static void test_false_positive_stop_prefix(void) {
    printf("[Test 3/6] Verifying False-Positive Partial Stop Prefix Resolution...\n");

    CaptureCtx cap = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, test_emit_callback, &cap);

    assert(lfm2_stream_filter_add_stop(&sf, "\nUser:"));
    assert(lfm2_stream_filter_add_stop(&sf, "<|im_end|>"));

    /* Feed: "\n" (held), "Use" (held), "rs" -> diverges from "\nUser:", so entire "\nUsers" flushes! */
    lfm2_stream_filter_feed(&sf, "Good news:\n", 11);
    assert(strcmp(cap.captured, "Good news:") == 0); /* '\n' held back */

    lfm2_stream_filter_feed(&sf, "Use", 3);
    assert(strcmp(cap.captured, "Good news:") == 0); /* "\nUse" held back */

    lfm2_stream_filter_feed(&sf, "rs are happy.", 13);
    /* Now divergent! Flushes "\nUsers are happy." */
    assert(strcmp(cap.captured, "Good news:\nUsers are happy.") == 0);

    lfm2_stream_filter_flush(&sf);
    assert(strcmp(cap.captured, "Good news:\nUsers are happy.") == 0);

    printf("PASSED: False positive stop prefixes cleanly flushed on divergence.\n");
}

/* ----------------------------------------------------------------------------
 * Test 4: End-of-Stream Partial Prefix Flush & Trailing Incomplete Discard
 * ---------------------------------------------------------------------------- */
static void test_stream_filter_flush_behavior(void) {
    printf("[Test 4/6] Verifying End-of-Stream Partial Prefix Flush & Incomplete Discard...\n");

    CaptureCtx cap = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, test_emit_callback, &cap);

    assert(lfm2_stream_filter_add_stop(&sf, "\nUser:"));

    /* Case A: Sentence ends with "\n" (a valid partial prefix of "\nUser:") */
    lfm2_stream_filter_feed(&sf, "Line 1\n", 7);
    assert(strcmp(cap.captured, "Line 1") == 0); /* '\n' held */

    /* Normal generation finishes: flush must output the trailing '\n' */
    lfm2_stream_filter_flush(&sf);
    assert(strcmp(cap.captured, "Line 1\n") == 0);

    /* Case B: Sentence ends with an incomplete multi-byte UTF-8 lead byte (0xF0) */
    lfm2_stream_filter_reset(&sf);
    cap.len = 0;
    cap.captured[0] = '\0';

    const char broken_tail[] = { 'O', 'K', (char)0xF0 };
    lfm2_stream_filter_feed(&sf, broken_tail, sizeof(broken_tail));
    assert(strcmp(cap.captured, "OK") == 0);

    /* Flush must NOT output orphaned 0xF0 lead byte (preventing invalid UTF-8) */
    lfm2_stream_filter_flush(&sf);
    assert(strcmp(cap.captured, "OK") == 0);

    printf("PASSED: End-of-stream flush correctly handles partial stops and discards incomplete UTF-8.\n");
}

/* ----------------------------------------------------------------------------
 * Test 5: Two-Stage SIGINT Handler Transitions
 * ---------------------------------------------------------------------------- */
static void test_two_stage_sigint_state_machine(void) {
    printf("[Test 5/6] Verifying Two-Stage SIGINT State Machine...\n");

    lfm2_install_sigint_guard();
    lfm2_sigint_clear_abort();
    lfm2_sigint_clear_exit();

    /* 1. Stage 1: Active generation -> Ctrl+C sets abort_gen = 1, exit_process = 0 */
    lfm2_sigint_begin_generation();
    assert(!lfm2_sigint_should_abort());
    assert(!lfm2_sigint_should_exit());

    raise(SIGINT);

    assert(lfm2_sigint_should_abort());
    assert(!lfm2_sigint_should_exit());
    lfm2_sigint_end_generation();

    /* Clear abort for the next turn */
    lfm2_sigint_clear_abort();
    assert(!lfm2_sigint_should_abort());

    /* 2. Stage 2: Idle at prompt -> Ctrl+C sets exit_process = 1 */
    raise(SIGINT);
    assert(lfm2_sigint_should_exit());

    lfm2_sigint_clear_exit();
    printf("PASSED: Two-stage SIGINT transitions verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 6: Terminal Detection & Large Buffer Overflow Safety
 * ---------------------------------------------------------------------------- */
static void test_terminal_and_buffer_safety(void) {
    printf("[Test 6/6] Verifying Terminal Detection & Large Buffer Safety...\n");

    LFM2TerminalEnv term;
    lfm2_detect_terminal_env(&term);
    assert(term.log_stream != NULL);

    /* Test feeding an oversized chunk (> 1024 bytes) */
    CaptureCtx cap = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, test_emit_callback, &cap);
    lfm2_stream_filter_add_stop(&sf, "###STOP###");

    char big_chunk[1500];
    memset(big_chunk, 'X', sizeof(big_chunk) - 1);
    big_chunk[sizeof(big_chunk) - 1] = '\0';

    lfm2_stream_status_t st = lfm2_stream_filter_feed(&sf, big_chunk, strlen(big_chunk));
    assert(st == LFM2_STREAM_OK);
    lfm2_stream_filter_flush(&sf);

    assert(cap.len == strlen(big_chunk));
    assert(memcmp(cap.captured, big_chunk, strlen(big_chunk)) == 0);

    /* Test stop string at the end of a big chunk */
    lfm2_stream_filter_reset(&sf);
    cap.len = 0;
    cap.captured[0] = '\0';

    memcpy(big_chunk + 1200, "###STOP###", 10);
    st = lfm2_stream_filter_feed(&sf, big_chunk, 1210);
    assert(st == LFM2_STREAM_STOPPED);
    assert(sf.matched_stop_idx == 0);
    assert(cap.len == 1200);

    printf("PASSED: Terminal environment detection and large buffer safety verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI CLI Stream & SIGINT Unit Test Suite   \n");
    printf("=======================================================\n");

    test_utf8_split_assembly();
    test_multi_token_stop_string();
    test_false_positive_stop_prefix();
    test_stream_filter_flush_behavior();
    test_two_stage_sigint_state_machine();
    test_terminal_and_buffer_safety();

    printf("=======================================================\n");
    printf("ALL CLI STREAM & SIGINT TESTS PASSED SUCCESSFULLY!     \n");
    printf("=======================================================\n");

    return EXIT_SUCCESS;
}

