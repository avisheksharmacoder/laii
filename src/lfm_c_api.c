#define _POSIX_C_SOURCE 200809L
#include "lfm_c_api.h"
#include "lfm_defaults.h"
#include "lfm_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

lfm_engine_t* lfm_create(const char *model_path, int num_threads) {
    if (!model_path) return NULL;

    LFMEngine *engine = (LFMEngine*)malloc(sizeof(LFMEngine));
    if (!engine) return NULL;

    uint32_t threads = (num_threads > 0) ? (uint32_t)num_threads : 0;
    if (!lfm_engine_init(engine, model_path, threads)) {
        free(engine);
        return NULL;
    }

    return (lfm_engine_t*)engine;
}

void lfm_destroy(lfm_engine_t *engine) {
    if (!engine) return;
    LFMEngine *eng = (LFMEngine*)engine;
    lfm_engine_free(eng);
    free(eng);
}

typedef struct {
    char *buf;
    size_t capacity;
    size_t len;
} BufferAccumulator;

static void buffer_callback(uint32_t token_id, const char *piece, void *user_data) {
    (void)token_id;
    BufferAccumulator *acc = (BufferAccumulator*)user_data;
    if (!piece || !acc->buf || acc->len >= acc->capacity) return;

    size_t plen = strlen(piece);
    size_t remaining = acc->capacity - acc->len - 1; // leave 1 byte for null terminator
    size_t to_copy = plen < remaining ? plen : remaining;

    if (to_copy > 0) {
        memcpy(acc->buf + acc->len, piece, to_copy);
        acc->len += to_copy;
        acc->buf[acc->len] = '\0';
    }
}

int lfm_generate(
    lfm_engine_t *engine,
    const char *prompt,
    char *out_buf,
    size_t out_buf_size,
    const lfm_sampler_params_t *params
) {
    if (!engine || !prompt || !out_buf || out_buf_size == 0) return -1;

    out_buf[0] = '\0';
    BufferAccumulator acc = {
        .buf = out_buf,
        .capacity = out_buf_size,
        .len = 0
    };

    LFMSamplerConfig cfg = {
        .temperature = params ? params->temperature : 0.8f,
        .top_k = params ? params->top_k : 40,
        .max_new_tokens = params ? params->max_new_tokens : (uint32_t)LFM_DEFAULT_MAX_TOKENS,
        .seed = params ? params->seed : 42,
        .echo_prompt = params ? params->echo_prompt : false
    };

    LFMEngine *eng = (LFMEngine*)engine;
    size_t n_gen = lfm_engine_generate(eng, prompt, &cfg, buffer_callback, &acc);
    return (int)n_gen;
}

int lfm_generate_stream(
    lfm_engine_t *engine,
    const char *prompt,
    const lfm_sampler_params_t *params,
    lfm_token_callback_t callback,
    void *user_data
) {
    if (!engine || !prompt) return -1;

    LFMSamplerConfig cfg = {
        .temperature = params ? params->temperature : 0.8f,
        .top_k = params ? params->top_k : 40,
        .max_new_tokens = params ? params->max_new_tokens : (uint32_t)LFM_DEFAULT_MAX_TOKENS,
        .seed = params ? params->seed : 42,
        .echo_prompt = params ? params->echo_prompt : false
    };

    LFMEngine *eng = (LFMEngine*)engine;
    size_t n_gen = lfm_engine_generate(eng, prompt, &cfg, (LFMTokenCallback)callback, user_data);
    return (int)n_gen;
}

int lfm_tokenize(
    lfm_engine_t *engine,
    const char *text,
    int32_t *tokens_out,
    int max_tokens
) {
    if (!engine || !text || !tokens_out || max_tokens <= 0) return -1;

    LFMEngine *eng = (LFMEngine*)engine;
    uint32_t *tmp = (uint32_t*)malloc(max_tokens * sizeof(uint32_t));
    if (!tmp) return -1;

    size_t count = lfm_tokenizer_encode(&eng->tokenizer, text, tmp, (size_t)max_tokens, true);

    for (size_t i = 0; i < count; ++i) {
        tokens_out[i] = (int32_t)tmp[i];
    }
    free(tmp);

    return (int)count;
}

int lfm_detokenize(
    lfm_engine_t *engine,
    int32_t token,
    char *buf_out,
    size_t buf_size
) {
    if (!engine || !buf_out || buf_size == 0) return -1;

    LFMEngine *eng = (LFMEngine*)engine;
    const char *piece = lfm_tokenizer_decode_id(&eng->tokenizer, (uint32_t)token);
    if (!piece) return -1;

    size_t len = strlen(piece);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(buf_out, piece, len);
    buf_out[len] = '\0';

    return (int)len;
}

uint32_t lfm_get_hidden_dim(const lfm_engine_t *engine) {
    if (!engine) return 0;
    const LFMEngine *eng = (const LFMEngine*)engine;
    return eng->model.config.hidden_dim;
}

uint32_t lfm_get_num_layers(const lfm_engine_t *engine) {
    if (!engine) return 0;
    const LFMEngine *eng = (const LFMEngine*)engine;
    return eng->model.config.n_layers;
}

uint32_t lfm_get_vocab_size(const lfm_engine_t *engine) {
    if (!engine) return 0;
    const LFMEngine *eng = (const LFMEngine*)engine;
    return eng->model.config.vocab_size;
}

const char* lfm_get_architecture(const lfm_engine_t *engine) {
    (void)engine;
    return "lfm2";
}
