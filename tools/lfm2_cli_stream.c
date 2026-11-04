#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef LFM2_CLI_STREAM_INCLUDED
#define LFM2_CLI_STREAM_INCLUDED 1

#define LFM2_MAX_STOP_STRINGS 16
#define LFM2_MAX_STOP_LEN     64
#define LFM2_STREAM_BUF_CAP   1024

/* ============================================================================
 * 1. Two-Stage SIGINT Handler & UNIX Pipe / TTY Detector
 * ============================================================================ */
static volatile sig_atomic_t g_lfm2_is_generating = 0;
static volatile sig_atomic_t g_lfm2_abort_gen     = 0;
static volatile sig_atomic_t g_lfm2_exit_process  = 0;

static void lfm2_sigint_handler(int sig) {
    (void)sig;
    if (g_lfm2_is_generating) {
        /* Stage 1: User pressed Ctrl+C during token generation -> abort turn only */
        g_lfm2_abort_gen = 1;
    } else {
        /* Stage 2: User pressed Ctrl+C while idle at prompt -> exit process */
        g_lfm2_exit_process = 1;
    }
}

/* Installs POSIX sigaction WITHOUT SA_RESTART so Ctrl+C immediately interrupts blocking fgets() */
static inline void lfm2_install_sigint_guard(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = lfm2_sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* Critical: omit SA_RESTART so fgets() returns on EINTR */
    sigaction(SIGINT, &sa, NULL);
}

static inline void lfm2_sigint_begin_generation(void) {
    g_lfm2_abort_gen     = 0;
    g_lfm2_is_generating = 1;
}

static inline void lfm2_sigint_end_generation(void) {
    g_lfm2_is_generating = 0;
}

static inline bool lfm2_sigint_should_abort(void) {
    return g_lfm2_abort_gen != 0;
}

static inline bool lfm2_sigint_should_exit(void) {
    return g_lfm2_exit_process != 0;
}

static inline void lfm2_sigint_clear_abort(void) {
    g_lfm2_abort_gen = 0;
}

static inline void lfm2_sigint_clear_exit(void) {
    g_lfm2_exit_process = 0;
}

typedef struct {
    bool stdin_is_tty;
    bool stdout_is_tty;
    bool use_ansi_colors;
    FILE *log_stream; /* stderr when stdout is piped/redirected so stdout stays pure model text */
} LFM2TerminalEnv;

static inline void lfm2_detect_terminal_env(LFM2TerminalEnv *env) {
    env->stdin_is_tty    = (isatty(STDIN_FILENO) != 0);
    env->stdout_is_tty   = (isatty(STDOUT_FILENO) != 0);
    const char *no_color = getenv("NO_COLOR");
    env->use_ansi_colors = env->stdout_is_tty && (no_color == NULL || no_color[0] == '\0');
    env->log_stream      = env->stdout_is_tty ? stdout : stderr;
}

/* ============================================================================
 * 2. UTF-8 Incomplete Tail Inspector (Zero-Mojibake Token Boundary Guard)
 * ============================================================================ */

/* Returns the expected byte length (1..4) of a UTF-8 codepoint from its lead byte,
 * or 0 if `b` is a continuation byte (10xxxxxx) or invalid byte (0xF8..0xFF). */
static inline size_t utf8_sequence_len(uint8_t lead) {
    if ((lead & 0x80u) == 0x00u) return 1; /* 0xxxxxxx (ASCII 1-byte) */
    if ((lead & 0xE0u) == 0xC0u) return 2; /* 110xxxxx (2-byte)        */
    if ((lead & 0xF0u) == 0xE0u) return 3; /* 1110xxxx (3-byte)        */
    if ((lead & 0xF8u) == 0xF0u) return 4; /* 11110xxx (4-byte)        */
    return 0;                              /* 10xxxxxx continuation or invalid */
}

static inline bool utf8_is_continuation(uint8_t b) {
    return (b & 0xC0u) == 0x80u; /* 10xxxxxx */
}

/* Given a buffer `buf[0 .. len-1]`, returns how many trailing bytes (0..3) belong
 * to an INCOMPLETE multi-byte UTF-8 sequence that is waiting for the next token. */
static inline size_t utf8_incomplete_tail_bytes(const uint8_t *buf, size_t len) {
    if (len == 0) return 0;

    /* Scan backwards up to 3 continuation bytes to find the last lead byte */
    size_t cont_count = 0;
    while (cont_count < len && cont_count < 3 &&
           utf8_is_continuation(buf[len - 1 - cont_count])) {
        cont_count++;
    }

    /* If we didn't find a lead byte within 3 bytes, the sequence is complete or malformed;
     * do not hold back forever. */
    if (cont_count == len || cont_count == 3) {
        return 0;
    }

    const size_t  lead_idx = len - 1 - cont_count;
    const uint8_t lead     = buf[lead_idx];
    const size_t  expected = utf8_sequence_len(lead);
    const size_t  present  = cont_count + 1;

    if (expected > present) {
        /* Multi-byte character is split across token boundary! Hold `present` bytes. */
        return present;
    }
    return 0;
}

/* ============================================================================
 * 3. Unified Streaming UTF-8 & Multi-Token Stop-String Filter
 * ============================================================================ */
typedef enum {
    LFM2_STREAM_OK      = 0, /* Normal streaming; safe prefix flushed       */
    LFM2_STREAM_STOPPED = 1  /* Stop string matched; pre-stop bytes flushed */
} lfm2_stream_status_t;

/* Callback invoked whenever verified-complete UTF-8 bytes are ready for output */
typedef void (*lfm2_stream_emit_fn)(const char *utf8_bytes, size_t len, void *user_data);

typedef struct {
    /* Registered multi-token stop strings (e.g. "\nUser:", "<|im_end|>") */
    char     stops[LFM2_MAX_STOP_STRINGS][LFM2_MAX_STOP_LEN];
    size_t   stop_lens[LFM2_MAX_STOP_STRINGS];
    uint32_t n_stops;
    int32_t  matched_stop_idx; /* Index of stop string that triggered halt (-1 if none) */

    /* Sliding holdback buffer for incomplete UTF-8 & partial stop prefixes */
    char     pending[LFM2_STREAM_BUF_CAP];
    size_t   pending_len;

    lfm2_stream_emit_fn emit_cb;
    void               *user_data;
} LFM2StreamFilter;

/* Default emitter for CLI: writes directly to stdout/stream and flushes immediately */
static inline void lfm2_stdout_emit_cb(const char *utf8_bytes, size_t len, void *user_data) {
    FILE *out = user_data ? (FILE *)user_data : stdout;
    if (len > 0) {
        fwrite(utf8_bytes, 1, len, out);
        fflush(out);
    }
}

static inline void lfm2_stream_filter_init(LFM2StreamFilter *sf,
                                          lfm2_stream_emit_fn emit_cb,
                                          void *user_data) {
    memset(sf, 0, sizeof(*sf));
    sf->matched_stop_idx = -1;
    sf->emit_cb          = emit_cb ? emit_cb : lfm2_stdout_emit_cb;
    sf->user_data        = user_data;
}

static inline void lfm2_stream_filter_reset(LFM2StreamFilter *sf) {
    sf->pending_len      = 0;
    sf->matched_stop_idx = -1;
    sf->pending[0]       = '\0';
}

static inline bool lfm2_stream_filter_add_stop(LFM2StreamFilter *sf, const char *stop_str) {
    if (!stop_str || !stop_str[0] || sf->n_stops >= LFM2_MAX_STOP_STRINGS) {
        return false;
    }
    const size_t len = strlen(stop_str);
    if (len >= LFM2_MAX_STOP_LEN) return false;

    memcpy(sf->stops[sf->n_stops], stop_str, len + 1);
    sf->stop_lens[sf->n_stops] = len;
    sf->n_stops++;
    return true;
}

/* Computes the longest suffix of `buf[0..buf_len-1]` that is a proper prefix
 * of ANY registered stop string. Those bytes must be held back until the next token. */
static inline size_t compute_stop_prefix_holdback(const LFM2StreamFilter *sf,
                                                 const char *buf,
                                                 size_t buf_len) {
    size_t max_hold = 0;

    for (uint32_t i = 0; i < sf->n_stops; ++i) {
        const char  *stop = sf->stops[i];
        const size_t slen = sf->stop_lens[i];
        if (slen <= 1) continue;

        /* Check candidate overlap lengths k = min(buf_len, slen - 1) down to max_hold + 1 */
        const size_t max_k = (buf_len < slen - 1) ? buf_len : (slen - 1);
        for (size_t k = max_k; k > max_hold; --k) {
            if (memcmp(buf + buf_len - k, stop, k) == 0) {
                max_hold = k;
                break;
            }
        }
    }
    return max_hold;
}

/* Internal helper to feed a chunk of raw token bytes through the filter */
static inline lfm2_stream_status_t lfm2_stream_filter_feed_chunk(LFM2StreamFilter *sf,
                                                                 const char *token_bytes,
                                                                 size_t token_len) {
    if (token_len == 0) return LFM2_STREAM_OK;

    /* Guard against buffer overflow by flushing verified safe prefix if necessary */
    if (sf->pending_len + token_len >= LFM2_STREAM_BUF_CAP) {
        size_t safe = sf->pending_len;
        size_t utf8_hold = utf8_incomplete_tail_bytes((const uint8_t *)sf->pending, safe);
        safe -= utf8_hold;
        if (safe > 0) {
            sf->emit_cb(sf->pending, safe, sf->user_data);
            memmove(sf->pending, sf->pending + safe, sf->pending_len - safe);
            sf->pending_len -= safe;
            sf->pending[sf->pending_len] = '\0';
        }
    }

    memcpy(sf->pending + sf->pending_len, token_bytes, token_len);
    sf->pending_len += token_len;
    sf->pending[sf->pending_len] = '\0';

    /* 1. Check for a complete Stop-String match anywhere in `pending`
     *    (If multiple match, pick the earliest match position in the stream) */
    size_t  earliest_pos = SIZE_MAX;
    int32_t matched_idx  = -1;

    for (uint32_t i = 0; i < sf->n_stops; ++i) {
        const char *hit = strstr(sf->pending, sf->stops[i]);
        if (hit != NULL) {
            const size_t pos = (size_t)(hit - sf->pending);
            if (pos < earliest_pos) {
                earliest_pos = pos;
                matched_idx  = (int32_t)i;
            }
        }
    }

    if (matched_idx >= 0) {
        /* Flush everything strictly BEFORE the stop string, discard the rest, and halt! */
        if (earliest_pos > 0) {
            size_t flush_len = earliest_pos;
            size_t utf8_hold = utf8_incomplete_tail_bytes((const uint8_t *)sf->pending, flush_len);
            flush_len -= utf8_hold;
            if (flush_len > 0) {
                sf->emit_cb(sf->pending, flush_len, sf->user_data);
            }
        }
        sf->pending_len      = 0;
        sf->pending[0]       = '\0';
        sf->matched_stop_idx = matched_idx;
        return LFM2_STREAM_STOPPED;
    }

    /* 2. Compute how many trailing bytes must be held back for partial stop prefixes */
    const size_t stop_hold = compute_stop_prefix_holdback(sf, sf->pending, sf->pending_len);
    size_t flush_len       = sf->pending_len - stop_hold;

    /* 3. Ensure `pending[0 .. flush_len - 1]` does not cut a multi-byte UTF-8 codepoint */
    if (flush_len > 0) {
        const size_t utf8_hold = utf8_incomplete_tail_bytes(
            (const uint8_t *)sf->pending, flush_len);
        flush_len -= utf8_hold;
    }

    /* 4. Emit the verified-safe prefix and shift remaining held bytes to index 0 */
    if (flush_len > 0) {
        sf->emit_cb(sf->pending, flush_len, sf->user_data);
        const size_t remain = sf->pending_len - flush_len;
        if (remain > 0) {
            memmove(sf->pending, sf->pending + flush_len, remain);
        }
        sf->pending_len = remain;
        sf->pending[sf->pending_len] = '\0';
    }

    return LFM2_STREAM_OK;
}

/* Feeds raw bytes from a newly decoded token into the stream filter.
 * Automatically handles chunks of any size, holds back partial stop strings and incomplete UTF-8 sequences,
 * and flushes verified bytes via `sf->emit_cb`. */
static inline lfm2_stream_status_t lfm2_stream_filter_feed(LFM2StreamFilter *sf,
                                                          const char *token_bytes,
                                                          size_t token_len) {
    if (!token_bytes || token_len == 0) return LFM2_STREAM_OK;

    const size_t max_chunk = LFM2_STREAM_BUF_CAP / 2;
    size_t offset = 0;
    while (offset < token_len) {
        size_t chunk = token_len - offset;
        if (chunk > max_chunk) chunk = max_chunk;
        lfm2_stream_status_t st = lfm2_stream_filter_feed_chunk(sf, token_bytes + offset, chunk);
        if (st == LFM2_STREAM_STOPPED) return LFM2_STREAM_STOPPED;
        offset += chunk;
    }
    return LFM2_STREAM_OK;
}

/* Called when generation ends normally (via <|im_end|> or Ctrl+C) to flush any
 * held-back partial stop prefix bytes that never turned into a full stop string. */
static inline void lfm2_stream_filter_flush(LFM2StreamFilter *sf) {
    if (sf->pending_len > 0) {
        const size_t utf8_hold = utf8_incomplete_tail_bytes(
            (const uint8_t *)sf->pending, sf->pending_len);
        const size_t valid_len = sf->pending_len - utf8_hold;
        if (valid_len > 0) {
            sf->emit_cb(sf->pending, valid_len, sf->user_data);
        }
        sf->pending_len = 0;
        sf->pending[0]  = '\0';
    }
}

#endif /* LFM2_CLI_STREAM_INCLUDED */

/* ============================================================================
 * 4. Standalone Verification Suite (UTF-8 Splits, Stop Holdback & SIGINT)
 * ============================================================================ */
#ifndef LFM2_CLI_STREAM_NO_MAIN

typedef struct {
    char   captured[2048];
    size_t len;
    int    flush_calls;
} LFM2TestCaptureBuf;

static void lfm2_test_capture_emit_cb(const char *utf8_bytes, size_t len, void *user_data) {
    LFM2TestCaptureBuf *cap = (LFM2TestCaptureBuf *)user_data;
    if (cap->len + len < sizeof(cap->captured)) {
        memcpy(cap->captured + cap->len, utf8_bytes, len);
        cap->len += len;
        cap->captured[cap->len] = '\0';
        cap->flush_calls++;
    }
}

int main(void) {
    lfm2_install_sigint_guard();

    LFM2TerminalEnv term;
    lfm2_detect_terminal_env(&term);

    printf("=================================================================\n");
    printf("   Liquid AI (LFM2) Streaming UTF-8 & Stop-String Verifier       \n");
    printf("=================================================================\n");
    printf("Terminal Environment: stdin_tty=%d | stdout_tty=%d | ansi_colors=%d\n\n",
           term.stdin_is_tty, term.stdout_is_tty, term.use_ansi_colors);

    /* ------------------------------------------------------------------------
     * Test 1: 4-Byte Emoji (🔥 = F0 9F 94 A5) & 3-Byte Box Char (─ = E2 94 80)
     *         Split Across Multiple BPE Tokens
     * ------------------------------------------------------------------------ */
    LFM2TestCaptureBuf cap1 = {0};
    LFM2StreamFilter sf;
    lfm2_stream_filter_init(&sf, lfm2_test_capture_emit_cb, &cap1);

    /* Token 1: "Fast " + first 2 bytes of 🔥 (0xF0, 0x9F) */
    const char tok1[] = { 'F', 'a', 's', 't', ' ', (char)0xF0, (char)0x9F };
    /* Token 2: last 2 bytes of 🔥 (0x94, 0xA5) + " C11 " + first byte of ─ (0xE2) */
    const char tok2[] = { (char)0x94, (char)0xA5, ' ', 'C', '1', '1', ' ', (char)0xE2 };
    /* Token 3: last 2 bytes of ─ (0x94, 0x80) + " engine!" */
    const char tok3[] = { (char)0x94, (char)0x80, ' ', 'e', 'n', 'g', 'i', 'n', 'e', '!' };

    lfm2_stream_filter_feed(&sf, tok1, sizeof(tok1));
    char after_tok1[sizeof(cap1.captured)] = {0};
    snprintf(after_tok1, sizeof(after_tok1), "%s", cap1.captured);
    bool step1_clean = (strcmp(cap1.captured, "Fast ") == 0); /* 0xF0 0x9F held back! */

    lfm2_stream_filter_feed(&sf, tok2, sizeof(tok2));
    char after_tok2[sizeof(cap1.captured)] = {0};
    snprintf(after_tok2, sizeof(after_tok2), "%s", cap1.captured);
    bool step2_clean = (strcmp(cap1.captured, "Fast 🔥 C11 ") == 0); /* 0xE2 held back! */

    lfm2_stream_filter_feed(&sf, tok3, sizeof(tok3));
    lfm2_stream_filter_flush(&sf);
    bool step3_clean = (strcmp(cap1.captured, "Fast 🔥 C11 ─ engine!") == 0);

    printf("[Test 1/3] Split Multi-Byte UTF-8 Assembly Test:\n");
    printf("  After Token 1 (held 0xF0 0x9F) : \"%s\"\n", after_tok1);
    printf("  After Token 2 (held 0xE2)      : \"%s\"\n", after_tok2);
    printf("  Final Assembled Output         : \"%s\"\n", cap1.captured);
    if (step1_clean && step2_clean && step3_clean) {
        printf("PASSED: Multi-byte UTF-8 split assembly verified with zero mojibake!\n\n");
    } else {
        fprintf(stderr, "FAILED: UTF-8 assembly mismatch: step1=%d step2=%d step3=%d\n",
                step1_clean, step2_clean, step3_clean);
        return EXIT_FAILURE;
    }

    /* ------------------------------------------------------------------------
     * Test 2: False-Positive Partial Stop Prefix Followed by Real Multi-Token Stop
     *         Registered stops: "\nUser:" and "<|im_end|>"
     * ------------------------------------------------------------------------ */
    LFM2TestCaptureBuf cap2 = {0};
    lfm2_stream_filter_init(&sf, lfm2_test_capture_emit_cb, &cap2);
    lfm2_stream_filter_add_stop(&sf, "\nUser:");
    lfm2_stream_filter_add_stop(&sf, "<|im_end|>");

    const char *stream_tokens[] = {
        "Note:\n",        /* Ends with "\n" -> partial prefix of "\nUser:", held back! */
        "Users",          /* "\nUsers" -> NOT "\nUser:", so "\nUsers" flushes immediately! */
        " love Linux.",   /* Normal flush */
        "\n",             /* Held back (matches prefix of "\nUser:") */
        "User",           /* Held back (matches "\nUser" prefix of "\nUser:") */
        ": What is next?",/* Completes "\nUser:"! Must halt and strip ": What is next?" */
        "NEVER_PRINTED"
    };

    lfm2_stream_status_t st = LFM2_STREAM_OK;
    for (size_t i = 0; i < sizeof(stream_tokens) / sizeof(stream_tokens[0]); ++i) {
        st = lfm2_stream_filter_feed(&sf, stream_tokens[i], strlen(stream_tokens[i]));
        if (st == LFM2_STREAM_STOPPED) break;
    }

    const bool stop_ok = (st == LFM2_STREAM_STOPPED) &&
                         (sf.matched_stop_idx == 0) &&
                         (strcmp(cap2.captured, "Note:\nUsers love Linux.") == 0);

    printf("[Test 2/3] Multi-Token Rolling Stop-String Matcher:\n");
    printf("  Emitted Before Stop : \"%s\"\n", cap2.captured);
    printf("  Matched Stop String : \"%s\" (index=%d)\n",
           sf.stops[sf.matched_stop_idx], sf.matched_stop_idx);
    if (stop_ok) {
        printf("PASSED: Multi-token stop holdback and truncation verified!\n\n");
    } else {
        fprintf(stderr, "FAILED: Stop-string mismatch: st=%d idx=%d cap=\"%s\"\n",
                st, sf.matched_stop_idx, cap2.captured);
        return EXIT_FAILURE;
    }

    /* ------------------------------------------------------------------------
     * Test 3: Simulate Two-Stage SIGINT Transition
     * ------------------------------------------------------------------------ */
    lfm2_sigint_clear_abort();
    lfm2_sigint_clear_exit();

    lfm2_sigint_begin_generation();
    raise(SIGINT); /* Simulate user pressing Ctrl+C during generation */
    const bool stage1_ok = lfm2_sigint_should_abort() && !lfm2_sigint_should_exit();
    lfm2_sigint_end_generation();

    raise(SIGINT); /* Simulate user pressing Ctrl+C while idle at prompt */
    const bool stage2_ok = lfm2_sigint_should_exit();

    printf("[Test 3/3] Two-Stage SIGINT (Ctrl+C) State Machine:\n");
    printf("  Ctrl+C During Decode : abort_turn=%d, exit_proc=%d\n",
           lfm2_sigint_should_abort() ? 1 : 0, 0);
    printf("  Ctrl+C At Idle Prompt: exit_proc=%d\n", stage2_ok ? 1 : 0);
    if (stage1_ok && stage2_ok) {
        printf("PASSED: Two-stage SIGINT handler verified successfully!\n\n");
    } else {
        fprintf(stderr, "FAILED: SIGINT stage mismatch: stage1=%d stage2=%d\n",
                stage1_ok, stage2_ok);
        return EXIT_FAILURE;
    }

    printf("=================================================================\n");
    printf("ALL TESTS PASSED WITH 100%% SUCCESS RATE!                        \n");
    printf("=================================================================\n");

    return EXIT_SUCCESS;
}

#endif /* LFM2_CLI_STREAM_NO_MAIN */
