#ifndef LFM_DEFAULTS_H
#define LFM_DEFAULTS_H

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Default maximum number of tokens generated per turn/request across LFM engines.
 * Override at compile time with -DLFM_DEFAULT_MAX_TOKENS=<N> (0 disables the
 * token-count cap while still enforcing the model context window limit).
 */
#ifndef LFM_DEFAULT_MAX_TOKENS
#define LFM_DEFAULT_MAX_TOKENS 2000U
#endif

/**
 * Default runtime context window (max_seq / KV cache slots) when not explicitly
 * configured by the developer/user. Override via -DLFM_DEFAULT_CTX_LEN=<N>.
 * Clamped automatically to the loaded model's native context length (n_ctx).
 */
#ifndef LFM_DEFAULT_CTX_LEN
#define LFM_DEFAULT_CTX_LEN 10000U
#endif

/**
 * Resolve effective runtime context length: use requested when > 0, else
 * LFM_DEFAULT_CTX_LEN, clamped to model_n_ctx when model_n_ctx > 0.
 */
static inline uint32_t lfm_resolve_ctx_len(uint32_t requested, uint32_t model_n_ctx) {
    uint32_t ctx = (requested > 0U) ? requested : (uint32_t)LFM_DEFAULT_CTX_LEN;
    if (model_n_ctx > 0U && ctx > model_n_ctx) {
        ctx = model_n_ctx;
    }
    return ctx;
}

/**
 * Parse a CLI/REPL/API token-limit string.
 * Values <= 0 return 0U (unlimited); empty or malformed inputs fall back to
 * LFM_DEFAULT_MAX_TOKENS.
 */
static inline uint32_t lfm_parse_max_tokens_arg(const char *s) {
    if (!s || !*s) {
        return (uint32_t)LFM_DEFAULT_MAX_TOKENS;
    }
    errno = 0;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (end == s) {
        return (uint32_t)LFM_DEFAULT_MAX_TOKENS;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        end++;
    }
    if (*end != '\0') {
        return (uint32_t)LFM_DEFAULT_MAX_TOKENS;
    }
    if (v <= 0) {
        return 0U;
    }
    if (errno == ERANGE || (unsigned long long)v > (unsigned long long)UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)v;
}

#endif /* LFM_DEFAULTS_H */

