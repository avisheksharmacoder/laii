#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/lfm2_gpt2_decode.h"

#ifndef LFM2_RUNTIME_FEATURES_INCLUDED
#define LFM2_RUNTIME_FEATURES_INCLUDED 1

#ifndef LFM2_KERNELS_FAST_INCLUDED
#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1
#define LFM2_DECODE_1X4_NO_MAIN 1
#define LFM2_CPUID_QUANTS_NO_MAIN 1
#define LFM2_KERNELS_FAST_NO_MAIN 1

#if __has_include("tools/lfm2_kernels_fast.c")
#include "tools/lfm2_kernels_fast.c"
#elif __has_include("lfm2_kernels_fast.c")
#include "lfm2_kernels_fast.c"
#else
#include "../tools/lfm2_kernels_fast.c"
#endif
#endif

#ifndef LFM2_TARGET_AVX2
#if defined(__x86_64__) || defined(_M_X64)
#define LFM2_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define LFM2_TARGET_AVX2
#endif
#endif

#ifndef LFM2_TARGET_ARM_FP16
#if defined(__aarch64__)
#define LFM2_TARGET_ARM_FP16 __attribute__((target("arch=armv8.2-a+dotprod+fp16")))
#else
#define LFM2_TARGET_ARM_FP16
#endif
#endif

#ifndef fp16_to_fp32
#define fp16_to_fp32 lfm2_fp16_to_fp32
#endif

#ifndef fp32_to_fp16_scalar
static inline uint16_t fp32_to_fp16_scalar(float f) {
#if defined(__F16C__)
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x00001000) >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    return (uint16_t)(sign | (exp << 10) | ((mant + 0x00001000) >> 13));
#endif
}
#endif

/* ============================================================================
 * 1. Pre-Decoded Raw Byte Vocabulary Table (For Fast O(1) Grammar Stepping)
 * ============================================================================ */
typedef struct {
    uint32_t  n_vocab;
    char     *arena;        /* Contiguous byte arena holding all decoded tokens */
    char    **bytes;        /* [n_vocab] pointer to decoded raw bytes           */
    uint16_t *lens;         /* [n_vocab] byte length of each decoded token      */
    bool     *is_special;   /* [n_vocab] true for <|...|> control tokens        */

    uint32_t  id_im_end;
    uint32_t  id_tool_start;
    uint32_t  id_tool_end;
    int32_t   id_think_open;  /* Optional single-token <think> ID (-1 if text)  */
    int32_t   id_think_close; /* Optional single-token </think> ID (-1 if text) */
} LFM2DecodedVocab;

/* Decodes GPT-2 byte-level UTF-8 string into raw ASCII/UTF-8 bytes */
static inline uint16_t decode_gpt2_token_bytes(const char *src_utf8, char *dst, size_t dst_cap) {
    return (uint16_t)lfm2_decode_gpt2_bytes(src_utf8, dst, dst_cap);
}

bool lfm2_decoded_vocab_init(LFM2DecodedVocab *dv, char **raw_tokens, uint32_t n_vocab) {
    memset(dv, 0, sizeof(*dv));
    dv->n_vocab        = n_vocab;
    dv->id_im_end      = 7;
    dv->id_tool_start  = 4;
    dv->id_tool_end    = 5;
    dv->id_think_open  = -1;
    dv->id_think_close = -1;

    dv->bytes      = (char **)calloc(n_vocab, sizeof(char *));
    dv->lens       = (uint16_t *)calloc(n_vocab, sizeof(uint16_t));
    dv->is_special = (bool *)calloc(n_vocab, sizeof(bool));

    size_t total_bytes = 0;
    for (uint32_t tid = 0; tid < n_vocab; ++tid) {
        total_bytes += strlen(raw_tokens[tid]) + 2;
    }
    if (total_bytes < (size_t)n_vocab * 32) total_bytes = (size_t)n_vocab * 32;

    dv->arena = (char *)malloc(total_bytes);
    if (!dv->bytes || !dv->lens || !dv->is_special || !dv->arena) return false;

    size_t arena_off = 0;
    char tmp[1024];

    for (uint32_t tid = 0; tid < n_vocab; ++tid) {
        const char *tok = raw_tokens[tid];
        size_t tlen = strlen(tok);

        if (tlen >= 4 && tok[0] == '<' && tok[1] == '|' &&
            tok[tlen - 2] == '|' && tok[tlen - 1] == '>') {
            dv->is_special[tid] = true;
            if (strcmp(tok, "<|im_end|>") == 0)          dv->id_im_end     = tid;
            if (strcmp(tok, "<|tool_call_start|>") == 0) dv->id_tool_start = tid;
            if (strcmp(tok, "<|tool_call_end|>") == 0)   dv->id_tool_end   = tid;
        }
        if (strcmp(tok, "<think>") == 0)  dv->id_think_open  = (int32_t)tid;
        if (strcmp(tok, "</think>") == 0) dv->id_think_close = (int32_t)tid;

        uint16_t dlen = decode_gpt2_token_bytes(tok, tmp, sizeof(tmp));
        dv->bytes[tid] = dv->arena + arena_off;
        dv->lens[tid]  = dlen;
        memcpy(dv->bytes[tid], tmp, dlen + 1);
        arena_off += (size_t)dlen + 1;
    }
    return true;
}

void lfm2_decoded_vocab_free(LFM2DecodedVocab *dv) {
    if (!dv) return;
    free(dv->bytes);
    free(dv->lens);
    free(dv->is_special);
    free(dv->arena);
    memset(dv, 0, sizeof(*dv));
}

/* ============================================================================
 * 2. Zero-Allocation Pushdown Automaton for JSON & LFM2 Pythonic Tool Calls
 * ============================================================================
 * Supports two grammar modes:
 *   1. LFM2_GRAMMAR_JSON_OBJECT: Strict RFC 8259 JSON (`{"key": [1, true, "v"]}`)
 *   2. LFM2_GRAMMAR_LFM2_TOOL_CALL: LFM2 native Pythonic or JSON tool calls inside
 *      `<|tool_call_start|>...<|tool_call_end|>`, e.g.:
 *      `[get_weather(city="Bengaluru", metric=True)]<|tool_call_end|>`
 * ============================================================================ */
#ifndef LFM2_H
typedef enum {
    LFM2_GRAMMAR_DISABLED       = 0,
    LFM2_GRAMMAR_JSON_OBJECT    = 1,
    LFM2_GRAMMAR_LFM2_TOOL_CALL = 2
} lfm2_grammar_mode_t;
#else
typedef lfm2_grammar_type_t lfm2_grammar_mode_t;
#define LFM2_GRAMMAR_DISABLED       LFM2_GRAMMAR_NONE
#define LFM2_GRAMMAR_LFM2_TOOL_CALL LFM2_GRAMMAR_TOOL_CALL
#endif

typedef enum {
    /* Shared & JSON states */
    G_JSON_START = 0,          /* Expecting '{' (or '[' in tool mode)          */
    G_JSON_OBJ_KEY_OR_END,     /* Inside '{': expecting '"' or '}'             */
    G_JSON_OBJ_KEY,            /* Expecting '"' for object key                 */
    G_JSON_AFTER_KEY,          /* Expecting ':' after object key               */
    G_JSON_VAL,                /* Expecting JSON value: '"', '{', '[', num, lit*/
    G_JSON_AFTER_VAL,          /* Expecting ',' or closing '}' / ']' / ')'     */
    G_JSON_STR,                /* Inside "..." string literal                  */
    G_JSON_STR_ESC,            /* Immediately after '\' inside string          */
    G_JSON_STR_HEX1,           /* \uXXXX hex digit 1                           */
    G_JSON_STR_HEX2,           /* \uXXXX hex digit 2                           */
    G_JSON_STR_HEX3,           /* \uXXXX hex digit 3                           */
    G_JSON_STR_HEX4,           /* \uXXXX hex digit 4                           */
    G_JSON_NUM_SIGN,           /* After '-': expecting '0' or '1'..'9'         */
    G_JSON_NUM_ZERO,           /* After leading '0': expecting '.', 'e', delim */
    G_JSON_NUM_INT,            /* After '1'..'9': expecting digits, '.', 'e'   */
    G_JSON_NUM_FRAC_START,     /* After '.': expecting '0'..'9'                */
    G_JSON_NUM_FRAC,           /* Fractional digits: expecting '0'..'9', 'e'   */
    G_JSON_NUM_EXP_START,      /* After 'e'/'E': expecting '+', '-', '0'..'9'  */
    G_JSON_NUM_EXP_SIGN,       /* After 'e+'/'e-': expecting '0'..'9'          */
    G_JSON_NUM_EXP,            /* Exponent digits: expecting '0'..'9' or delim */
    G_JSON_LIT,                /* Matching literal ("true", "false", "null")   */

    /* LFM2 Pythonic Tool-Call States: [func_name(arg1=val1, arg2=val2)] */
    G_TOOL_WAIT_TRIGGER,       /* Unconstrained until <|tool_call_start|>      */
    G_TOOL_START,              /* After <|tool_call_start|>: '[' or '{'        */
    G_TOOL_CALL_OR_END,        /* Inside '[': expecting func identifier or ']' */
    G_TOOL_FUNC_NAME,          /* Reading [a-zA-Z0-9_.] function name          */
    G_TOOL_AFTER_FUNC,         /* Expecting '('                                */
    G_TOOL_ARG_OR_CLOSE,       /* Inside '(': expecting arg_name or ')'        */
    G_TOOL_ARG_NAME,           /* Reading [a-zA-Z0-9_] keyword argument name   */
    G_TOOL_AFTER_ARG_NAME,     /* Expecting '='                                */
    G_TOOL_AFTER_CALL,         /* After ')': expecting ',' or ']'              */

    G_COMPLETE,                /* Top-level object/tool call balanced & closed */
    G_ERROR
} lfm2_gstate_t;

typedef struct {
    lfm2_grammar_mode_t mode;
    lfm2_gstate_t       state;
    lfm2_gstate_t       after_str_state; /* Target state when closing '"'      */
    const char         *lit_target;      /* Remaining literal chars to match   */
    uint8_t             depth;           /* Nesting depth of {}, [], ()        */
    uint8_t             stack[32];       /* Stack of '{', '[', '(' containers  */
} LFM2Grammar;

void lfm2_grammar_init(LFM2Grammar *g, lfm2_grammar_mode_t mode) {
    memset(g, 0, sizeof(*g));
    g->mode = mode;
    if (mode == LFM2_GRAMMAR_JSON_OBJECT) {
        g->state = G_JSON_START;
    } else if (mode == LFM2_GRAMMAR_LFM2_TOOL_CALL) {
        g->state = G_TOOL_WAIT_TRIGGER;
    } else {
        g->state = G_COMPLETE;
    }
}

static inline bool is_json_ws(char c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
}

static inline bool is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static inline bool is_ident_char(char c) {
    return is_ident_start(c) || (c >= '0' && c <= '9') || c == '.';
}

static inline bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

/* Advances the Pushdown Automaton by a single character `c`.
 * Returns `false` immediately if `c` violates the active JSON or Tool-Call grammar. */
static bool lfm2_grammar_step_char(LFM2Grammar *g, char c) {
reprocess:
    switch (g->state) {
        case G_TOOL_WAIT_TRIGGER:
            return true; /* Unconstrained text before <|tool_call_start|> */

        case G_JSON_START:
            if (is_json_ws(c)) return true;
            if (c == '{') {
                g->stack[g->depth++] = '{';
                g->state = G_JSON_OBJ_KEY_OR_END;
                return true;
            }
            return false;

        case G_TOOL_START:
            if (is_json_ws(c)) return true;
            if (c == '[') {
                g->stack[g->depth++] = 'T'; /* 'T' = top-level Pythonic tool list */
                g->state = G_TOOL_CALL_OR_END;
                return true;
            }
            if (c == '{') {
                g->stack[g->depth++] = '{'; /* JSON-style tool call */
                g->state = G_JSON_OBJ_KEY_OR_END;
                return true;
            }
            return false;

        case G_TOOL_CALL_OR_END:
            if (is_json_ws(c)) return true;
            if (c == ']' && g->depth == 1 && g->stack[0] == 'T') {
                g->depth = 0;
                g->state = G_COMPLETE;
                return true;
            }
            if (is_ident_start(c)) {
                g->state = G_TOOL_FUNC_NAME;
                return true;
            }
            return false;

        case G_TOOL_FUNC_NAME:
            if (is_ident_char(c)) return true;
            if (is_json_ws(c)) { g->state = G_TOOL_AFTER_FUNC; return true; }
            if (c == '(') {
                if (g->depth >= 31) return false;
                g->stack[g->depth++] = '(';
                g->state = G_TOOL_ARG_OR_CLOSE;
                return true;
            }
            return false;

        case G_TOOL_AFTER_FUNC:
            if (is_json_ws(c)) return true;
            if (c == '(') {
                if (g->depth >= 31) return false;
                g->stack[g->depth++] = '(';
                g->state = G_TOOL_ARG_OR_CLOSE;
                return true;
            }
            return false;

        case G_TOOL_ARG_OR_CLOSE:
            if (is_json_ws(c)) return true;
            if (c == ')') {
                g->depth--;
                g->state = G_TOOL_AFTER_CALL;
                return true;
            }
            if (is_ident_start(c)) {
                g->state = G_TOOL_ARG_NAME;
                return true;
            }
            return false;

        case G_TOOL_ARG_NAME:
            if (is_ident_start(c) || (c >= '0' && c <= '9')) return true;
            if (is_json_ws(c)) { g->state = G_TOOL_AFTER_ARG_NAME; return true; }
            if (c == '=') { g->state = G_JSON_VAL; return true; }
            return false;

        case G_TOOL_AFTER_ARG_NAME:
            if (is_json_ws(c)) return true;
            if (c == '=') { g->state = G_JSON_VAL; return true; }
            return false;

        case G_TOOL_AFTER_CALL:
            if (is_json_ws(c)) return true;
            if (c == ',') { g->state = G_TOOL_CALL_OR_END; return true; }
            if (c == ']' && g->depth == 1 && g->stack[0] == 'T') {
                g->depth = 0;
                g->state = G_COMPLETE;
                return true;
            }
            return false;

        case G_JSON_OBJ_KEY_OR_END:
            if (is_json_ws(c)) return true;
            if (c == '}') {
                g->depth--;
                g->state = (g->depth == 0) ? G_COMPLETE : G_JSON_AFTER_VAL;
                return true;
            }
            if (c == '"') {
                g->after_str_state = G_JSON_AFTER_KEY;
                g->state = G_JSON_STR;
                return true;
            }
            return false;

        case G_JSON_OBJ_KEY:
            if (is_json_ws(c)) return true;
            if (c == '"') {
                g->after_str_state = G_JSON_AFTER_KEY;
                g->state = G_JSON_STR;
                return true;
            }
            return false;

        case G_JSON_AFTER_KEY:
            if (is_json_ws(c)) return true;
            if (c == ':') { g->state = G_JSON_VAL; return true; }
            return false;

        case G_JSON_VAL:
            if (is_json_ws(c)) return true;
            if (c == '"') {
                g->after_str_state = G_JSON_AFTER_VAL;
                g->state = G_JSON_STR;
                return true;
            }
            if (c == '{') {
                if (g->depth >= 31) return false;
                g->stack[g->depth++] = '{';
                g->state = G_JSON_OBJ_KEY_OR_END;
                return true;
            }
            if (c == '[') {
                if (g->depth >= 31) return false;
                g->stack[g->depth++] = '[';
                g->state = G_JSON_VAL;
                return true;
            }
            /* Allow empty array `[]` */
            if (c == ']' && g->depth > 0 && g->stack[g->depth - 1] == '[') {
                g->depth--;
                g->state = (g->depth == 0) ? G_COMPLETE : G_JSON_AFTER_VAL;
                return true;
            }
            if (c == '-') { g->state = G_JSON_NUM_SIGN; return true; }
            if (c == '0') { g->state = G_JSON_NUM_ZERO; return true; }
            if (c >= '1' && c <= '9') { g->state = G_JSON_NUM_INT; return true; }
            if (c == 't') { g->lit_target = "rue";  g->state = G_JSON_LIT; return true; }
            if (c == 'f') { g->lit_target = "alse"; g->state = G_JSON_LIT; return true; }
            if (c == 'n') { g->lit_target = "ull";  g->state = G_JSON_LIT; return true; }
            /* Pythonic literals inside LFM2 tool calls: True, False, None */
            if (g->mode == LFM2_GRAMMAR_LFM2_TOOL_CALL) {
                if (c == 'T') { g->lit_target = "rue";  g->state = G_JSON_LIT; return true; }
                if (c == 'F') { g->lit_target = "alse"; g->state = G_JSON_LIT; return true; }
                if (c == 'N') { g->lit_target = "one";  g->state = G_JSON_LIT; return true; }
            }
            return false;

        case G_JSON_STR:
            if ((uint8_t)c < 0x20) return false; /* Unescaped control chars forbidden */
            if (c == '\\') { g->state = G_JSON_STR_ESC; return true; }
            if (c == '"')  { g->state = g->after_str_state; return true; }
            return true;

        case G_JSON_STR_ESC:
            if (c == '"' || c == '\\' || c == '/' || c == 'b' ||
                c == 'f' || c == 'n'  || c == 'r' || c == 't') {
                g->state = G_JSON_STR;
                return true;
            }
            if (c == 'u') { g->state = G_JSON_STR_HEX1; return true; }
            return false;

        case G_JSON_STR_HEX1:
            if (!is_hex_digit(c)) return false;
            g->state = G_JSON_STR_HEX2; return true;
        case G_JSON_STR_HEX2:
            if (!is_hex_digit(c)) return false;
            g->state = G_JSON_STR_HEX3; return true;
        case G_JSON_STR_HEX3:
            if (!is_hex_digit(c)) return false;
            g->state = G_JSON_STR_HEX4; return true;
        case G_JSON_STR_HEX4:
            if (!is_hex_digit(c)) return false;
            g->state = G_JSON_STR; return true;

        case G_JSON_NUM_SIGN:
            if (c == '0') { g->state = G_JSON_NUM_ZERO; return true; }
            if (c >= '1' && c <= '9') { g->state = G_JSON_NUM_INT; return true; }
            return false;

        case G_JSON_NUM_ZERO:
            if (c == '.') { g->state = G_JSON_NUM_FRAC_START; return true; }
            if (c == 'e' || c == 'E') { g->state = G_JSON_NUM_EXP_START; return true; }
            g->state = G_JSON_AFTER_VAL;
            goto reprocess;

        case G_JSON_NUM_INT:
            if (c >= '0' && c <= '9') return true;
            if (c == '.') { g->state = G_JSON_NUM_FRAC_START; return true; }
            if (c == 'e' || c == 'E') { g->state = G_JSON_NUM_EXP_START; return true; }
            g->state = G_JSON_AFTER_VAL;
            goto reprocess;

        case G_JSON_NUM_FRAC_START:
            if (c >= '0' && c <= '9') { g->state = G_JSON_NUM_FRAC; return true; }
            return false;

        case G_JSON_NUM_FRAC:
            if (c >= '0' && c <= '9') return true;
            if (c == 'e' || c == 'E') { g->state = G_JSON_NUM_EXP_START; return true; }
            g->state = G_JSON_AFTER_VAL;
            goto reprocess;

        case G_JSON_NUM_EXP_START:
            if (c == '+' || c == '-') { g->state = G_JSON_NUM_EXP_SIGN; return true; }
            if (c >= '0' && c <= '9') { g->state = G_JSON_NUM_EXP; return true; }
            return false;

        case G_JSON_NUM_EXP_SIGN:
            if (c >= '0' && c <= '9') { g->state = G_JSON_NUM_EXP; return true; }
            return false;

        case G_JSON_NUM_EXP:
            if (c >= '0' && c <= '9') return true;
            g->state = G_JSON_AFTER_VAL;
            goto reprocess;

        case G_JSON_LIT:
            if (c == *g->lit_target) {
                g->lit_target++;
                if (*g->lit_target == '\0') g->state = G_JSON_AFTER_VAL;
                return true;
            }
            return false;

        case G_JSON_AFTER_VAL: {
            if (is_json_ws(c)) return true;
            if (g->depth == 0) return false;
            const uint8_t top = g->stack[g->depth - 1];
            if (top == '{') {
                if (c == ',') { g->state = G_JSON_OBJ_KEY; return true; }
                if (c == '}') {
                    g->depth--;
                    g->state = (g->depth == 0) ? G_COMPLETE : G_JSON_AFTER_VAL;
                    return true;
                }
            } else if (top == '[') {
                if (c == ',') { g->state = G_JSON_VAL; return true; }
                if (c == ']') {
                    g->depth--;
                    g->state = (g->depth == 0) ? G_COMPLETE : G_JSON_AFTER_VAL;
                    return true;
                }
            } else if (top == '(') {
                if (c == ',') { g->state = G_TOOL_ARG_OR_CLOSE; return true; }
                if (c == ')') {
                    g->depth--;
                    g->state = G_TOOL_AFTER_CALL;
                    return true;
                }
            }
            return false;
        }

        case G_COMPLETE:
            return is_json_ws(c);

        default:
            return false;
    }
}

/* Tests whether `tid` is syntactically valid from the current `LFM2Grammar` state */
static inline bool lfm2_grammar_can_accept_token(const LFM2Grammar *g,
                                                 const LFM2DecodedVocab *dv,
                                                 uint32_t tid) {
    if (g->mode == LFM2_GRAMMAR_DISABLED || g->state == G_TOOL_WAIT_TRIGGER) {
        return true;
    }

    /* Once the root JSON object or tool call is complete, ONLY allow end tokens */
    if (g->state == G_COMPLETE) {
        if (g->mode == LFM2_GRAMMAR_LFM2_TOOL_CALL) {
            return (tid == dv->id_tool_end);
        }
        return (tid == dv->id_im_end);
    }

    /* Inside an incomplete JSON object or tool call, block all <|...|> special tokens */
    if (dv->is_special[tid]) {
        return false;
    }

    LFM2Grammar copy = *g;
    const char *s    = dv->bytes[tid];
    const uint16_t n = dv->lens[tid];
    for (uint16_t i = 0; i < n; ++i) {
        if (!lfm2_grammar_step_char(&copy, s[i])) {
            return false;
        }
    }
    return true;
}

/* Applies grammar logit mask in-place across `logits[0 .. n_vocab-1]` */
uint32_t lfm2_grammar_mask_logits(const LFM2Grammar *g,
                                  const LFM2DecodedVocab *dv,
                                  float * restrict logits) {
    if (g->mode == LFM2_GRAMMAR_DISABLED || g->state == G_TOOL_WAIT_TRIGGER) {
        return dv->n_vocab;
    }

    uint32_t n_allowed = 0;
    for (uint32_t tid = 0; tid < dv->n_vocab; ++tid) {
        if (lfm2_grammar_can_accept_token(g, dv, tid)) {
            n_allowed++;
        } else {
            logits[tid] = -INFINITY;
        }
    }
    return n_allowed;
}

/* Updates the persistent grammar state after `sampled_tid` is chosen */
void lfm2_grammar_commit_token(LFM2Grammar *g, const LFM2DecodedVocab *dv, uint32_t sampled_tid) {
    if (g->mode == LFM2_GRAMMAR_DISABLED) return;

    if (g->mode == LFM2_GRAMMAR_LFM2_TOOL_CALL) {
        if (sampled_tid == dv->id_tool_start) {
            g->depth = 0;
            g->state = G_TOOL_START;
            return;
        }
        if (sampled_tid == dv->id_tool_end) {
            g->depth = 0;
            g->state = G_TOOL_WAIT_TRIGGER;
            return;
        }
        if (g->state == G_TOOL_WAIT_TRIGGER) return;
    }

    if (dv->is_special[sampled_tid]) return;

    const char *s    = dv->bytes[sampled_tid];
    const uint16_t n = dv->lens[sampled_tid];
    for (uint16_t i = 0; i < n; ++i) {
        lfm2_grammar_step_char(g, s[i]);
    }
}

/* ============================================================================
 * 3. LFM2.5 `<think>` Reasoning Token Budget Controller
 * ============================================================================ */
typedef enum {
    LFM2_THINK_IDLE          = 0, /* Before <think> (or non-thinking model)    */
    LFM2_THINK_ACTIVE        = 1, /* Currently generating inside <think>       */
    LFM2_THINK_FORCING_CLOSE = 2, /* Budget reached: emitting \n</think>\n     */
    LFM2_THINK_ANSWERING     = 3  /* Past </think>, generating final response  */
} lfm2_think_phase_t;

typedef struct {
    lfm2_think_phase_t phase;
    uint32_t           think_tokens_used;
    uint32_t           max_think_tokens;   /* e.g. 512; 0 = unlimited          */

    /* Pre-tokenized sequence for "\n</think>\n" forced budget termination */
    uint32_t           close_seq[8];
    uint32_t           close_seq_len;
    uint32_t           close_seq_cursor;

    /* Rolling tail window to detect multi-token "<think>" / "</think>" */
    char               recent_tail[32];
    uint32_t           tail_len;
} LFM2ThinkController;

void lfm2_think_controller_init(LFM2ThinkController *tc,
                                uint32_t max_think_tokens,
                                const uint32_t *close_think_ids,
                                uint32_t n_close_ids) {
    memset(tc, 0, sizeof(*tc));
    tc->phase            = LFM2_THINK_IDLE;
    tc->max_think_tokens = max_think_tokens;
    tc->close_seq_len    = (n_close_ids <= 8) ? n_close_ids : 8;
    for (uint32_t i = 0; i < tc->close_seq_len; ++i) {
        tc->close_seq[i] = close_think_ids[i];
    }
}

/* If the thinking budget is exhausted, masks all logits except the next token
 * in the `\n</think>\n` closing sequence so the forward pass closes cleanly. */
void lfm2_think_apply_budget_mask(LFM2ThinkController *tc,
                                  float * restrict logits,
                                  uint32_t n_vocab) {
    if (tc->phase == LFM2_THINK_ACTIVE &&
        tc->max_think_tokens > 0 &&
        tc->think_tokens_used >= tc->max_think_tokens &&
        tc->close_seq_len > 0) {
        tc->phase            = LFM2_THINK_FORCING_CLOSE;
        tc->close_seq_cursor = 0;
    }

    if (tc->phase == LFM2_THINK_FORCING_CLOSE) {
        const uint32_t forced_tid = tc->close_seq[tc->close_seq_cursor];
        for (uint32_t i = 0; i < n_vocab; ++i) {
            logits[i] = -INFINITY;
        }
        if (forced_tid < n_vocab) {
            logits[forced_tid] = 1e30f;
        }
    }
}

/* Updates the `<think>` state machine after `sampled_tid` is selected */
void lfm2_think_commit_token(LFM2ThinkController *tc,
                             const LFM2DecodedVocab *dv,
                             uint32_t sampled_tid) {
    if (tc->phase == LFM2_THINK_FORCING_CLOSE) {
        tc->close_seq_cursor++;
        if (tc->close_seq_cursor >= tc->close_seq_len) {
            tc->phase = LFM2_THINK_ANSWERING;
        }
        return;
    }

    /* Append decoded bytes into rolling 31-byte tail buffer */
    const char *b    = dv->bytes[sampled_tid];
    const uint16_t n = dv->lens[sampled_tid];
    for (uint16_t i = 0; i < n; ++i) {
        if (tc->tail_len >= sizeof(tc->recent_tail) - 1) {
            memmove(tc->recent_tail, tc->recent_tail + 1, sizeof(tc->recent_tail) - 2);
            tc->tail_len = sizeof(tc->recent_tail) - 2;
        }
        tc->recent_tail[tc->tail_len++] = b[i];
        tc->recent_tail[tc->tail_len]   = '\0';
    }

    if (tc->phase == LFM2_THINK_IDLE) {
        if ((dv->id_think_open >= 0 && (int32_t)sampled_tid == dv->id_think_open) ||
            strstr(tc->recent_tail, "<think>") != NULL) {
            tc->phase             = LFM2_THINK_ACTIVE;
            tc->think_tokens_used = 0;
            tc->tail_len          = 0;
            tc->recent_tail[0]    = '\0';
        }
    } else if (tc->phase == LFM2_THINK_ACTIVE) {
        tc->think_tokens_used++;
        if ((dv->id_think_close >= 0 && (int32_t)sampled_tid == dv->id_think_close) ||
            strstr(tc->recent_tail, "</think>") != NULL) {
            tc->phase = LFM2_THINK_ANSWERING;
        }
    }
}

/* ============================================================================
 * 4. GQA Attention-Sink KV-Cache Shifting with Incremental RoPE Re-Rotation
 * ============================================================================
 * When `pos` reaches `max_seq_len`, keeps the first `n_sink` tokens (e.g. 64),
 * discards `n_discard` old tokens (e.g. 1984), shifts the recent `n_keep`
 * tokens down to `n_sink`, and counter-rotates every shifted FP16 Key vector
 * by `-n_discard` positions using the precomputed `LFM2RoPELUT`!
 * Note: LFM2's 10 ShortConv layers are 100% untouched because their 2-token
 * rolling state already represents the exact immediate tail at `pos - 1`!
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64)
/* AVX2 Hardware Accelerated Kernel */
LFM2_TARGET_AVX2
void lfm2_kv_cache_shift_sinks_avx2(uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                    uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                    const LFM2Model * restrict m,
                                    const LFM2RoPELUT * restrict rope_lut,
                                    uint32_t * restrict inout_pos,
                                    uint32_t n_sink,
                                    uint32_t n_discard) {
    const uint32_t cur_pos = *inout_pos;
    if (n_discard == 0 || cur_pos <= n_sink + n_discard) return;

    const uint32_t head_dim = m->cfg.n_embd / (m->cfg.n_head_q ? m->cfg.n_head_q : 1);
    if (head_dim != 64) {
        *inout_pos = cur_pos;
        return;
    }
    const uint32_t n_keep = cur_pos - (n_sink + n_discard);

    /* Load cos(n_discard * freq) and sin(n_discard * freq) from RoPE LUT */
    const float * restrict cos_d = rope_lut->cos_lut + (size_t)n_discard * 32;
    const float * restrict sin_d = rope_lut->sin_lut + (size_t)n_discard * 32;

    const __m256 c0 = _mm256_loadu_ps(cos_d +  0), s0 = _mm256_loadu_ps(sin_d +  0);
    const __m256 c1 = _mm256_loadu_ps(cos_d +  8), s1 = _mm256_loadu_ps(sin_d +  8);
    const __m256 c2 = _mm256_loadu_ps(cos_d + 16), s2 = _mm256_loadu_ps(sin_d + 16);
    const __m256 c3 = _mm256_loadu_ps(cos_d + 24), s3 = _mm256_loadu_ps(sin_d + 24);

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        /* ShortConv layers need ZERO adjustment during context shifting */
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) continue;

        const uint32_t n_head_kv = m->cfg.n_head_kv[l];
        const size_t   kv_dim    = (size_t)n_head_kv * head_dim;

        uint16_t *kc = k_cache_f16[l];
        uint16_t *vc = v_cache_f16[l];
        if (!kc || !vc) continue;

        /* 1. Shift Value cache down by `n_discard` steps (Values have no RoPE) */
        memmove(vc + (size_t)n_sink * kv_dim,
                vc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        /* 2. Shift Key cache down by `n_discard` steps */
        memmove(kc + (size_t)n_sink * kv_dim,
                kc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        /* 3. Counter-rotate shifted Keys [n_sink .. n_sink + n_keep - 1] by -n_discard:
         *    k'_lo = k_lo * cos(d) + k_hi * sin(d)
         *    k'_hi = k_hi * cos(d) - k_lo * sin(d) */
        for (uint32_t t = n_sink; t < n_sink + n_keep; ++t) {
            uint16_t *k_step = kc + (size_t)t * kv_dim;
            for (uint32_t h = 0; h < n_head_kv; ++h) {
                uint16_t *kh = k_step + (size_t)h * 64;

                __m256 lo0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh +  0)));
                __m256 lo1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh +  8)));
                __m256 lo2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 16)));
                __m256 lo3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 24)));

                __m256 hi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 32)));
                __m256 hi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 40)));
                __m256 hi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 48)));
                __m256 hi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + 56)));

                const __m256 n_lo0 = _mm256_fmadd_ps(lo0, c0, _mm256_mul_ps(hi0, s0));
                const __m256 n_hi0 = _mm256_fmsub_ps(hi0, c0, _mm256_mul_ps(lo0, s0));
                const __m256 n_lo1 = _mm256_fmadd_ps(lo1, c1, _mm256_mul_ps(hi1, s1));
                const __m256 n_hi1 = _mm256_fmsub_ps(hi1, c1, _mm256_mul_ps(lo1, s1));
                const __m256 n_lo2 = _mm256_fmadd_ps(lo2, c2, _mm256_mul_ps(hi2, s2));
                const __m256 n_hi2 = _mm256_fmsub_ps(hi2, c2, _mm256_mul_ps(lo2, s2));
                const __m256 n_lo3 = _mm256_fmadd_ps(lo3, c3, _mm256_mul_ps(hi3, s3));
                const __m256 n_hi3 = _mm256_fmsub_ps(hi3, c3, _mm256_mul_ps(lo3, s3));

                _mm_storeu_si128((__m128i *)(kh +  0), _mm256_cvtps_ph(n_lo0, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh +  8), _mm256_cvtps_ph(n_lo1, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 16), _mm256_cvtps_ph(n_lo2, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 24), _mm256_cvtps_ph(n_lo3, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 32), _mm256_cvtps_ph(n_hi0, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 40), _mm256_cvtps_ph(n_hi1, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 48), _mm256_cvtps_ph(n_hi2, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(kh + 56), _mm256_cvtps_ph(n_hi3, _MM_FROUND_TO_NEAREST_INT));
            }
        }
    }

    *inout_pos = n_sink + n_keep;
}
#endif

/* Portable Scalar Fallback Kernel */
static inline void lfm2_kv_cache_shift_sinks_scalar(uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                                     uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                                     const LFM2Model * restrict m,
                                                     const LFM2RoPELUT * restrict rope_lut,
                                                     uint32_t * restrict inout_pos,
                                                     uint32_t n_sink,
                                                     uint32_t n_discard) {
    const uint32_t cur_pos = *inout_pos;
    if (n_discard == 0 || cur_pos <= n_sink + n_discard) return;

    const uint32_t head_dim = m->cfg.n_embd / (m->cfg.n_head_q ? m->cfg.n_head_q : 1);
    if (head_dim != 64) {
        *inout_pos = cur_pos;
        return;
    }
    const uint32_t n_keep = cur_pos - (n_sink + n_discard);

    const float * restrict cos_d = rope_lut->cos_lut + (size_t)n_discard * 32;
    const float * restrict sin_d = rope_lut->sin_lut + (size_t)n_discard * 32;

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) continue;

        const uint32_t n_head_kv = m->cfg.n_head_kv[l];
        const size_t   kv_dim    = (size_t)n_head_kv * head_dim;

        uint16_t *kc = k_cache_f16[l];
        uint16_t *vc = v_cache_f16[l];
        if (!kc || !vc) continue;

        /* Shift Value cache down by n_discard steps */
        memmove(vc + (size_t)n_sink * kv_dim,
                vc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        /* Shift Key cache down by n_discard steps */
        memmove(kc + (size_t)n_sink * kv_dim,
                kc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        /* Counter-rotate shifted Keys [n_sink .. n_sink + n_keep - 1] by -n_discard */
        for (uint32_t t = n_sink; t < n_sink + n_keep; ++t) {
            uint16_t *k_step = kc + (size_t)t * kv_dim;
            for (uint32_t h = 0; h < n_head_kv; ++h) {
                uint16_t *kh = k_step + (size_t)h * 64;
                for (uint32_t i = 0; i < 32; ++i) {
                    float lo = fp16_to_fp32(kh[i]);
                    float hi = fp16_to_fp32(kh[32 + i]);
                    float c  = cos_d[i];
                    float s  = sin_d[i];
                    float n_lo = lo * c + hi * s;
                    float n_hi = hi * c - lo * s;
                    kh[i]      = fp32_to_fp16_scalar(n_lo);
                    kh[32 + i] = fp32_to_fp16_scalar(n_hi);
                }
            }
        }
    }

    *inout_pos = n_sink + n_keep;
}

#if defined(__aarch64__)
/* ARM64 NEON + FP16 GQA Attention-Sink KV-Cache Shifting & RoPE Re-Rotation */
LFM2_TARGET_ARM_FP16
void lfm2_kv_cache_shift_sinks_neon(uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                    uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                    const LFM2Model * restrict m,
                                    const LFM2RoPELUT * restrict rope_lut,
                                    uint32_t * restrict inout_pos,
                                    uint32_t n_sink,
                                    uint32_t n_discard) {
    const uint32_t cur_pos = *inout_pos;
    if (n_discard == 0 || cur_pos <= n_sink + n_discard) return;

    const uint32_t head_dim = m->cfg.n_embd / (m->cfg.n_head_q ? m->cfg.n_head_q : 1);
    if (head_dim != 64) {
        *inout_pos = cur_pos;
        return;
    }
    const uint32_t n_keep = cur_pos - (n_sink + n_discard);

    const float * restrict cos_d = rope_lut->cos_lut + (size_t)n_discard * 32;
    const float * restrict sin_d = rope_lut->sin_lut + (size_t)n_discard * 32;

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) continue;

        const uint32_t n_head_kv = m->cfg.n_head_kv[l];
        const size_t   kv_dim    = (size_t)n_head_kv * head_dim;

        uint16_t *kc = k_cache_f16[l];
        uint16_t *vc = v_cache_f16[l];
        if (!kc || !vc) continue;

        /* 1. Shift Value & Key caches down by n_discard tokens */
        memmove(vc + (size_t)n_sink * kv_dim,
                vc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        memmove(kc + (size_t)n_sink * kv_dim,
                kc + (size_t)(n_sink + n_discard) * kv_dim,
                (size_t)n_keep * kv_dim * sizeof(uint16_t));

        /* 2. Counter-rotate shifted Keys [n_sink .. n_sink + n_keep - 1] by -n_discard:
         *    n_lo = (lo * c) + (hi * s)  -> vfmaq_f32(vmulq_f32(lo, c), hi, s)
         *    n_hi = (hi * c) - (lo * s)  -> vfmsq_f32(vmulq_f32(hi, c), lo, s) */
        for (uint32_t t = n_sink; t < n_sink + n_keep; ++t) {
            uint16_t *k_step = kc + (size_t)t * kv_dim;
            for (uint32_t h = 0; h < n_head_kv; ++h) {
                uint16_t * restrict kh = k_step + (size_t)h * 64;

                for (uint32_t i = 0; i < 32; i += 8) {
                    float32x4_t lo0, lo1, hi0, hi1;
                    neon_load_fp16x8_to_fp32(kh + i +  0, &lo0, &lo1);
                    neon_load_fp16x8_to_fp32(kh + i + 32, &hi0, &hi1);

                    const float32x4_t c0 = vld1q_f32(cos_d + i + 0), c1 = vld1q_f32(cos_d + i + 4);
                    const float32x4_t s0 = vld1q_f32(sin_d + i + 0), s1 = vld1q_f32(sin_d + i + 4);

                    const float32x4_t n_lo0 = vfmaq_f32(vmulq_f32(lo0, c0), hi0, s0);
                    const float32x4_t n_lo1 = vfmaq_f32(vmulq_f32(lo1, c1), hi1, s1);
                    const float32x4_t n_hi0 = vfmsq_f32(vmulq_f32(hi0, c0), lo0, s0);
                    const float32x4_t n_hi1 = vfmsq_f32(vmulq_f32(hi1, c1), lo1, s1);

                    neon_store_fp32_to_fp16x8(kh + i +  0, n_lo0, n_lo1);
                    neon_store_fp32_to_fp16x8(kh + i + 32, n_hi0, n_hi1);
                }
            }
        }
    }

    *inout_pos = n_sink + n_keep;
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define lfm2_kv_cache_shift_sinks_simd lfm2_kv_cache_shift_sinks_avx2
#elif defined(__aarch64__)
    #define lfm2_kv_cache_shift_sinks_simd lfm2_kv_cache_shift_sinks_neon
#else
    #define lfm2_kv_cache_shift_sinks_simd lfm2_kv_cache_shift_sinks_scalar
#endif

/* Unified Dispatcher */
static inline void lfm2_kv_cache_shift_sinks(uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                             uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                             const LFM2Model * restrict m,
                                             const LFM2RoPELUT * restrict rope_lut,
                                             uint32_t * restrict inout_pos,
                                             uint32_t n_sink,
                                             uint32_t n_discard) {
#if defined(__x86_64__) || defined(_M_X64)
    const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
    if (kt->has_avx2 && kt->has_fma && kt->has_f16c) {
        const char *force_isa = getenv("LFM2_FORCE_ISA");
        if (!force_isa || strcmp(force_isa, "scalar") != 0) {
            lfm2_kv_cache_shift_sinks_avx2(k_cache_f16, v_cache_f16, m, rope_lut, inout_pos, n_sink, n_discard);
            return;
        }
    }
#elif defined(__aarch64__)
    const char *force_isa = getenv("LFM2_FORCE_ISA");
    if (!force_isa || strcmp(force_isa, "scalar") != 0) {
        lfm2_kv_cache_shift_sinks_neon(k_cache_f16, v_cache_f16, m, rope_lut, inout_pos, n_sink, n_discard);
        return;
    }
#endif
    lfm2_kv_cache_shift_sinks_scalar(k_cache_f16, v_cache_f16, m, rope_lut, inout_pos, n_sink, n_discard);
}

/* ============================================================================
 * 4B. LFM3 Sink-Shift Dispatcher (Global Partial-RoPE 128-Dim + SWA Ring)
 * ============================================================================ */
static inline void lfm3_counter_rotate_key_head_scalar(uint16_t * restrict kh,
                                                       uint32_t head_dim,
                                                       const float * restrict cos_d,
                                                       const float * restrict sin_d) {
    const uint32_t half_dim = head_dim / 2U;
    for (uint32_t i = 0; i < half_dim; ++i) {
        const float c  = cos_d[i];
        const float s  = sin_d[i];
        if (s == 0.0f && c == 1.0f) continue; /* Partial RoPE unrotated dims preserved identically */
        const float lo = fp16_to_fp32(kh[i]);
        const float hi = fp16_to_fp32(kh[i + half_dim]);
        kh[i]            = fp32_to_fp16(lo * c + hi * s);
        kh[i + half_dim] = fp32_to_fp16(hi * c - lo * s);
    }
}

#if LFM2_HAS_AVX2
LFM2_TARGET_AVX2
static inline void lfm3_counter_rotate_key_head_avx2(uint16_t * restrict kh,
                                                     uint32_t head_dim,
                                                     const float * restrict cos_d,
                                                     const float * restrict sin_d) {
    const uint32_t half_dim = head_dim / 2U;
    uint32_t i = 0;
    for (; i + 8U <= half_dim; i += 8U) {
        const __m256 c  = _mm256_loadu_ps(cos_d + i);
        const __m256 s  = _mm256_loadu_ps(sin_d + i);
        const __m256 lo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + i)));
        const __m256 hi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + i + half_dim)));
        const __m256 nlo = _mm256_fmadd_ps(lo, c, _mm256_mul_ps(hi, s));
        const __m256 nhi = _mm256_fmsub_ps(hi, c, _mm256_mul_ps(lo, s));
        _mm_storeu_si128((__m128i *)(kh + i),            _mm256_cvtps_ph(nlo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_storeu_si128((__m128i *)(kh + i + half_dim), _mm256_cvtps_ph(nhi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    for (; i < half_dim; ++i) {
        const float c = cos_d[i], s = sin_d[i];
        const float lo = fp16_to_fp32(kh[i]);
        const float hi = fp16_to_fp32(kh[i + half_dim]);
        kh[i]            = fp32_to_fp16(lo * c + hi * s);
        kh[i + half_dim] = fp32_to_fp16(hi * c - lo * s);
    }
}
#endif

static inline void lfm3_kv_cache_shift_sinks(uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                             uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                             const LFM2Model * restrict m,
                                             const LFM2RoPELUT * restrict rope_lut_full,
                                             const LFM2RoPELUT * restrict rope_lut_swa,
                                             uint32_t swa_cap,
                                             uint32_t * restrict inout_pos,
                                             uint32_t n_sink,
                                             uint32_t n_discard) {
    if (!m || !inout_pos || !rope_lut_full || !rope_lut_swa) return;
    const uint32_t cur_pos = *inout_pos;
    if (n_discard == 0U || cur_pos <= n_sink + n_discard) return;
    if (n_discard >= rope_lut_full->max_seq_len || n_discard >= rope_lut_swa->max_seq_len) return;

    const uint32_t head_dim = (m->cfg.head_dim > 0U) ? m->cfg.head_dim
                              : (m->cfg.n_head_q > 0U ? (m->cfg.n_embd / m->cfg.n_head_q) : 128U);
    if ((head_dim % 2U) != 0U) return;
    const uint32_t half_dim = head_dim / 2U;
    const uint32_t n_keep   = cur_pos - (n_sink + n_discard);
    const uint32_t new_pos  = n_sink + n_keep;
    const uint32_t swa_win  = (m->cfg.sliding_window > 0U) ? m->cfg.sliding_window : 256U;
    const uint32_t eff_cap  = (swa_cap > 0U) ? swa_cap : (swa_win + 8U);

    bool use_avx2 = false;
#if LFM2_HAS_AVX2
    {
        const LFM2KernelTable *kt = lfm2_get_cpu_kernel_table();
        const char *force_isa = getenv("LFM2_FORCE_ISA");
        use_avx2 = kt->has_avx2 && kt->has_fma && kt->has_f16c &&
                   (!force_isa || strcmp(force_isa, "scalar") != 0);
    }
#endif

    const float *cos_full = rope_lut_full->cos_lut + (size_t)n_discard * half_dim;
    const float *sin_full = rope_lut_full->sin_lut + (size_t)n_discard * half_dim;
    const float *cos_swa  = rope_lut_swa->cos_lut  + (size_t)n_discard * half_dim;
    const float *sin_swa  = rope_lut_swa->sin_lut  + (size_t)n_discard * half_dim;

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) continue;
        uint16_t *kc = k_cache_f16[l];
        uint16_t *vc = v_cache_f16[l];
        if (!kc || !vc) continue;

        const uint32_t n_head_kv = m->cfg.n_head_kv[l];
        const size_t   kv_dim    = (size_t)n_head_kv * head_dim;

        if (m->layers[l].block_type == LFM2_BLOCK_SWA_ATTN) {
            /* Repack valid SWA window [max(0, new_pos - swa_win) .. new_pos - 1] into new ring slots */
            const uint32_t t_new_start = (new_pos > swa_win) ? (new_pos - swa_win) : 0U;
            const uint32_t n_valid     = new_pos - t_new_start;
            const size_t   tmp_bytes   = (size_t)n_valid * kv_dim * sizeof(uint16_t);
            uint16_t *tmp_k = (uint16_t *)malloc(tmp_bytes);
            uint16_t *tmp_v = (uint16_t *)malloc(tmp_bytes);
            if (!tmp_k || !tmp_v) {
                free(tmp_k); free(tmp_v);
                continue;
            }
            for (uint32_t idx = 0; idx < n_valid; ++idx) {
                const uint32_t t_new   = t_new_start + idx;
                const bool     is_kept = (t_new >= n_sink);
                const uint32_t t_old   = is_kept ? (t_new + n_discard) : t_new;
                const uint32_t s_old   = t_old % eff_cap;
                uint16_t *dk = tmp_k + (size_t)idx * kv_dim;
                uint16_t *dv = tmp_v + (size_t)idx * kv_dim;
                memcpy(dk, kc + (size_t)s_old * kv_dim, kv_dim * sizeof(uint16_t));
                memcpy(dv, vc + (size_t)s_old * kv_dim, kv_dim * sizeof(uint16_t));
                if (is_kept) {
                    for (uint32_t h = 0; h < n_head_kv; ++h) {
#if LFM2_HAS_AVX2
                        if (use_avx2) {
                            lfm3_counter_rotate_key_head_avx2(dk + (size_t)h * head_dim, head_dim, cos_swa, sin_swa);
                            continue;
                        }
#endif
                        lfm3_counter_rotate_key_head_scalar(dk + (size_t)h * head_dim, head_dim, cos_swa, sin_swa);
                    }
                }
            }
            for (uint32_t idx = 0; idx < n_valid; ++idx) {
                const uint32_t s_new = (t_new_start + idx) % eff_cap;
                memcpy(kc + (size_t)s_new * kv_dim, tmp_k + (size_t)idx * kv_dim, kv_dim * sizeof(uint16_t));
                memcpy(vc + (size_t)s_new * kv_dim, tmp_v + (size_t)idx * kv_dim, kv_dim * sizeof(uint16_t));
            }
            free(tmp_k);
            free(tmp_v);
        } else {
            memmove(vc + (size_t)n_sink * kv_dim,
                    vc + (size_t)(n_sink + n_discard) * kv_dim,
                    (size_t)n_keep * kv_dim * sizeof(uint16_t));
            memmove(kc + (size_t)n_sink * kv_dim,
                    kc + (size_t)(n_sink + n_discard) * kv_dim,
                    (size_t)n_keep * kv_dim * sizeof(uint16_t));

            for (uint32_t t = n_sink; t < n_sink + n_keep; ++t) {
                uint16_t *k_step = kc + (size_t)t * kv_dim;
                for (uint32_t h = 0; h < n_head_kv; ++h) {
#if LFM2_HAS_AVX2
                    if (use_avx2) {
                        lfm3_counter_rotate_key_head_avx2(k_step + (size_t)h * head_dim, head_dim, cos_full, sin_full);
                        continue;
                    }
#endif
                    lfm3_counter_rotate_key_head_scalar(k_step + (size_t)h * head_dim, head_dim, cos_full, sin_full);
                }
            }
        }
    }

    *inout_pos = new_pos;
}

#endif /* LFM2_RUNTIME_FEATURES_INCLUDED */

/* ============================================================================
 * 5. Standalone Verification Suite (Grammar + Think Budget + RoPE KV-Shift)
 * ============================================================================ */
#ifndef LFM2_RUNTIME_FEATURES_NO_MAIN
int main(void) {
    printf("=== LFM2 Grammar Enforcer, <think> Controller & KV-Shift Verifier ===\n\n");

    /* 1. Build a synthetic mini-vocabulary to test Grammar & <think> masking */
    const char *mock_tokens[] = {
        "<|pad|>", "<|startoftext|>", "<|im_start|>", "<|im_end|>",
        "<|tool_call_start|>", "<|tool_call_end|>", "<think>", "</think>",
        "{", "}", "[", "]", "(", ")", ":", ",", "=", "\"city\"", "\"Bengaluru\"",
        "get_weather", "city", "metric", "True", "42", "\n", "invalid_syntax$$"
    };
    const uint32_t V = sizeof(mock_tokens) / sizeof(mock_tokens[0]);

    LFM2DecodedVocab dv;
    bool vok = lfm2_decoded_vocab_init(&dv, (char **)mock_tokens, V);
    if (!vok) {
        fprintf(stderr, "Failed to initialize decoded vocab!\n");
        return EXIT_FAILURE;
    }

    /* Test A: LFM2 Pythonic Tool-Call Grammar Enforcer */
    LFM2Grammar g_tool;
    lfm2_grammar_init(&g_tool, LFM2_GRAMMAR_LFM2_TOOL_CALL);

    const uint32_t valid_tool_seq[] = {
        4,  /* <|tool_call_start|> */
        10, /* [                   */
        19, /* get_weather         */
        12, /* (                   */
        20, /* city                */
        16, /* =                   */
        18, /* "Bengaluru"         */
        15, /* ,                   */
        21, /* metric              */
        16, /* =                   */
        22, /* True                */
        13, /* )                   */
        11, /* ]                   */
        5   /* <|tool_call_end|>   */
    };

    printf("1. Testing LFM2 Pythonic Tool-Call Grammar PDA:\n   ");
    float logits[32];
    bool tool_ok = true;
    for (size_t step = 0; step < sizeof(valid_tool_seq) / sizeof(valid_tool_seq[0]); ++step) {
        for (uint32_t i = 0; i < V; ++i) logits[i] = 1.0f;
        lfm2_grammar_mask_logits(&g_tool, &dv, logits);

        uint32_t tid = valid_tool_seq[step];
        if (isinf(logits[tid]) || !isinf(logits[25])) { /* token 25 ("invalid_syntax$$") must be -INF */
            if (g_tool.state != G_TOOL_WAIT_TRIGGER) tool_ok = false;
        }
        printf("%s", dv.bytes[tid]);
        lfm2_grammar_commit_token(&g_tool, &dv, tid);
    }
    printf("\n   Result: %s (invalid tokens masked to -INF at every step)\n\n",
           tool_ok ? "PASS" : "FAIL");

    /* Test B: <think> Reasoning Token Budget Controller */
    const uint32_t close_seq[3] = { 24 /* \n */, 7 /* </think> */, 24 /* \n */ };
    LFM2ThinkController tc;
    lfm2_think_controller_init(&tc, 3 /* max_think_tokens = 3 */, close_seq, 3);

    lfm2_think_commit_token(&tc, &dv, 6 /* <think> */);
    lfm2_think_commit_token(&tc, &dv, 19); /* think token 1 */
    lfm2_think_commit_token(&tc, &dv, 20); /* think token 2 */
    lfm2_think_commit_token(&tc, &dv, 21); /* think token 3 -> hits budget! */

    for (uint32_t i = 0; i < V; ++i) logits[i] = 0.5f;
    lfm2_think_apply_budget_mask(&tc, logits, V);
    printf("2. <think> Budget Controller (max_think_tokens = 3):\n");
    printf("   Phase after 3 tokens : %s\n",
           (tc.phase == LFM2_THINK_FORCING_CLOSE) ? "LFM2_THINK_FORCING_CLOSE (PASS)" : "FAIL");
    printf("   Forced close token 0 : id=%u (logit=%+.1e, others=-inf)\n\n",
           close_seq[0], logits[close_seq[0]]);

    /* Test C: GQA Attention-Sink KV-Cache Shifting & RoPE Inverse Rotation */
    LFM2Model mock_model;
    memset(&mock_model, 0, sizeof(mock_model));
    mock_model.cfg.n_layers     = 1;
    mock_model.cfg.n_embd       = 2048;
    mock_model.cfg.n_head_q     = 32;
    mock_model.cfg.n_head_kv[0] = 8;
    mock_model.layers[0].block_type = LFM2_BLOCK_GQA_ATTN;

    LFM2RoPELUT rope_lut;
    lfm2_rope_lut_init(&rope_lut, 512, 64, 1000000.0f);

    const uint32_t kv_dim    = 8 * 64;
    const uint32_t old_pos   = 200;
    const uint32_t n_sink    = 16;
    const uint32_t n_discard = 100;
    const uint32_t new_pos   = old_pos - n_discard; /* 100 */

    uint16_t *k_cache_f16[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_cache_f16[LFM2_MAX_LAYERS] = {0};
    k_cache_f16[0] = (uint16_t *)lfm2_aligned_alloc_64(512 * kv_dim * sizeof(uint16_t));
    v_cache_f16[0] = (uint16_t *)lfm2_aligned_alloc_64(512 * kv_dim * sizeof(uint16_t));

    float *k_orig_f32   = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    float *k_direct_f32 = (float *)lfm2_aligned_alloc_64(kv_dim * sizeof(float));
    uint16_t *k_direct_f16 = (uint16_t *)lfm2_aligned_alloc_64(kv_dim * sizeof(uint16_t));

    for (uint32_t i = 0; i < kv_dim; ++i) {
        k_orig_f32[i] = k_direct_f32[i] = sinf((float)i * 0.037f);
    }

    /* Rotate k_orig_f32 at `old_pos = 200` and store in k_cache_f16[0][200] */
    lfm2_qk_norm_rope_simd(k_orig_f32, k_cache_f16[0] + (size_t)old_pos * kv_dim,
                           8, 64, NULL, 1e-5f, &rope_lut, old_pos);

    /* Rotate k_direct_f32 directly at `new_pos = 100` as ground-truth reference */
    lfm2_qk_norm_rope_simd(k_direct_f32, k_direct_f16,
                           8, 64, NULL, 1e-5f, &rope_lut, new_pos);

    /* Shift KV cache by discarding `n_discard = 100` tokens after `n_sink = 16` */
    uint32_t cursor = 256;
    lfm2_kv_cache_shift_sinks_simd(k_cache_f16, v_cache_f16, &mock_model,
                                   &rope_lut, &cursor, n_sink, n_discard);

    float max_shift_err = 0.0f;
    const uint16_t *shifted_k = k_cache_f16[0] + (size_t)new_pos * kv_dim;
    for (uint32_t i = 0; i < kv_dim; ++i) {
        float v_shift  = fp16_to_fp32(shifted_k[i]);
        float v_direct = fp16_to_fp32(k_direct_f16[i]);
        float err = fabsf(v_shift - v_direct);
        if (err > max_shift_err) max_shift_err = err;
    }

    printf("3. GQA Attention-Sink KV-Cache Shift (old_pos=200 -> new_pos=100):\n");
    printf("   Updated Cursor       : %u (expected 156)\n", cursor);
    printf("   RoPE Re-Rotation Err : %.6f (FP16 round-trip precision)\n", max_shift_err);

    free(k_cache_f16[0]); free(v_cache_f16[0]);
    free(k_orig_f32); free(k_direct_f32); free(k_direct_f16);
    lfm2_rope_lut_free(&rope_lut);
    lfm2_decoded_vocab_free(&dv);
    return EXIT_SUCCESS;
}
#endif
