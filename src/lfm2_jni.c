/*
 * src/lfm2_jni.c - Native JNI Bridge for Android & Java/Kotlin
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Avishek and Contributors
 *
 * Implements high-performance JNI bindings for `ai.liquid.lfm2.LFM2Model`
 * and `ai.liquid.lfm2.LFM2Context`. Compatible with Android ART (API 26+)
 * and OpenJDK 8+.
 */

#ifndef LFM2_JNI_C
#define LFM2_JNI_C

#include "lfm2.h"
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#define LFM2_JNI_EXPORT JNIEXPORT JNICALL

/* ============================================================================
 * ai.liquid.lfm2.LFM2Model Native Methods
 * ============================================================================ */

LFM2_JNI_EXPORT jlong Java_ai_liquid_lfm2_LFM2Model_nativeLoadFromFile(
    JNIEnv *env, jclass clazz, jstring path, jboolean warmup) {
    (void)clazz;
    if (!path) return 0;
    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return 0;

    lfm2_model_t *model = lfm2_model_load(c_path, warmup ? 1 : 0);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jlong)(uintptr_t)model;
}

LFM2_JNI_EXPORT jlong Java_ai_liquid_lfm2_LFM2Model_nativeLoadFromFd(
    JNIEnv *env, jclass clazz, jint fd, jlong fileSize, jboolean warmup) {
    (void)env; (void)clazz;
    if (fd < 0) return 0;
    lfm2_model_t *model = lfm2_model_load_from_fd(
        (int)fd,
        (fileSize > 0) ? (uint64_t)fileSize : 0ULL,
        warmup ? 1 : 0
    );
    return (jlong)(uintptr_t)model;
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Model_nativeFree(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_model_t *model = (lfm2_model_t *)(uintptr_t)handle;
        lfm2_model_free(model);
    }
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Model_nativeReleasePageCache(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_model_t *model = (lfm2_model_t *)(uintptr_t)handle;
        lfm2_model_release_page_cache(model);
    }
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Model_nativeWarmupPageCache(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_model_t *model = (lfm2_model_t *)(uintptr_t)handle;
        lfm2_model_warmup_page_cache(model);
    }
}

LFM2_JNI_EXPORT jlong Java_ai_liquid_lfm2_LFM2Model_nativeGetProcessRss(
    JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)lfm2_get_process_rss_bytes();
}

LFM2_JNI_EXPORT jstring Java_ai_liquid_lfm2_LFM2Model_nativeGetArch(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)clazz;
    if (!handle) return NULL;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) {
        return NULL;
    }
    return (*env)->NewStringUTF(env, info.arch);
}

LFM2_JNI_EXPORT jstring Java_ai_liquid_lfm2_LFM2Model_nativeGetActiveIsa(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)clazz;
    if (!handle) return NULL;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) {
        return NULL;
    }
    return (*env)->NewStringUTF(env, info.active_isa);
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetNumLayers(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_layers;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetNumShortConvLayers(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_shortconv_layers;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetNumGqaLayers(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_gqa_layers;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetHiddenDim(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_embd;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetFfnDim(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_ff;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetVocabSize(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_vocab;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetTrainCtx(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_ctx_train;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetNumExperts(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_experts;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Model_nativeGetNumExpertsUsed(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jint)info.n_experts_used;
}

LFM2_JNI_EXPORT jlong Java_ai_liquid_lfm2_LFM2Model_nativeGetFileSize(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    lfm2_model_info_t info;
    if (lfm2_model_get_info((const lfm2_model_t *)(uintptr_t)handle, &info) != LFM2_OK) return 0;
    return (jlong)info.file_size_bytes;
}

LFM2_JNI_EXPORT jintArray Java_ai_liquid_lfm2_LFM2Model_nativeTokenize(
    JNIEnv *env, jclass clazz, jlong handle, jstring text, jint maxTokens) {
    (void)clazz;
    if (!handle || !text || maxTokens <= 0) return NULL;
    const char *c_text = (*env)->GetStringUTFChars(env, text, NULL);
    if (!c_text) return NULL;

    uint32_t *tokens = (uint32_t *)malloc((size_t)maxTokens * sizeof(uint32_t));
    if (!tokens) {
        (*env)->ReleaseStringUTFChars(env, text, c_text);
        return NULL;
    }

    int32_t count = lfm2_tokenize_text(
        (const lfm2_model_t *)(uintptr_t)handle,
        c_text,
        tokens,
        (uint32_t)maxTokens
    );
    (*env)->ReleaseStringUTFChars(env, text, c_text);

    if (count < 0) {
        free(tokens);
        return NULL;
    }

    jintArray result = (*env)->NewIntArray(env, count);
    if (result) {
        (*env)->SetIntArrayRegion(env, result, 0, count, (const jint *)tokens);
    }
    free(tokens);
    return result;
}

LFM2_JNI_EXPORT jlong Java_ai_liquid_lfm2_LFM2Model_nativeCreateContext(
    JNIEnv *env, jclass clazz, jlong handle,
    jint maxSeqLen, jint numThreads, jint sinkTokens,
    jboolean autoClampRam, jboolean warmupMmap) {
    (void)env; (void)clazz;
    if (!handle) return 0;

    lfm2_context_params_t params = lfm2_context_default_params();
    if (maxSeqLen > 0)    params.max_seq_len    = (uint32_t)maxSeqLen;
    if (numThreads >= 0)  params.n_threads      = (uint32_t)numThreads;
    if (sinkTokens > 0)   params.n_sink_tokens  = (uint32_t)sinkTokens;
    params.auto_clamp_ram = autoClampRam ? 1 : 0;
    params.warmup_mmap    = warmupMmap ? 1 : 0;

    lfm2_context_t *ctx = lfm2_context_create(
        (const lfm2_model_t *)(uintptr_t)handle,
        &params
    );
    return (jlong)(uintptr_t)ctx;
}

/* ============================================================================
 * ai.liquid.lfm2.LFM2Context Native Methods
 * ============================================================================ */

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Context_nativeFree(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;
        lfm2_context_free(ctx);
    }
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Context_nativeReset(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;
        lfm2_context_reset(ctx);
    }
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeGetPos(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (!handle) return 0;
    return (jint)lfm2_context_get_pos((const lfm2_context_t *)(uintptr_t)handle);
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeSetSystemPrompt(
    JNIEnv *env, jclass clazz, jlong handle, jstring prompt, jstring toolsJson) {
    (void)clazz;
    if (!handle) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    const char *c_prompt = prompt ? (*env)->GetStringUTFChars(env, prompt, NULL) : NULL;
    const char *c_tools  = toolsJson ? (*env)->GetStringUTFChars(env, toolsJson, NULL) : NULL;

    lfm2_status_t st = lfm2_context_set_system_prompt(ctx, c_prompt, c_tools);

    if (prompt && c_prompt) (*env)->ReleaseStringUTFChars(env, prompt, c_prompt);
    if (toolsJson && c_tools) (*env)->ReleaseStringUTFChars(env, toolsJson, c_tools);

    return (jint)st;
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Context_nativeClearStops(
    JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    if (handle) {
        lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;
        lfm2_context_clear_stops(ctx);
    }
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeAddStop(
    JNIEnv *env, jclass clazz, jlong handle, jstring stopStr) {
    (void)clazz;
    if (!handle || !stopStr) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    const char *c_stop = (*env)->GetStringUTFChars(env, stopStr, NULL);
    if (!c_stop) return LFM2_ERR_INVALID_ARG;

    lfm2_status_t st = lfm2_context_add_stop(ctx, c_stop);
    (*env)->ReleaseStringUTFChars(env, stopStr, c_stop);
    return (jint)st;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeBeginTurn(
    JNIEnv *env, jclass clazz, jlong handle, jstring message,
    jfloat temperature, jfloat minP, jint topK, jfloat repPenalty,
    jint repWindow, jlong seed, jint grammarType, jint maxThinkTokens) {
    (void)clazz;
    if (!handle || !message) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    const char *c_msg = (*env)->GetStringUTFChars(env, message, NULL);
    if (!c_msg) return LFM2_ERR_INVALID_ARG;

    lfm2_sample_params_t sp = lfm2_sample_default_params();
    sp.temperature      = (float)temperature;
    sp.min_p            = (float)minP;
    sp.top_k            = (uint32_t)topK;
    sp.rep_penalty      = (float)repPenalty;
    sp.rep_window       = (uint32_t)repWindow;
    sp.seed             = (uint64_t)seed;
    sp.grammar_type     = (int32_t)grammarType;
    sp.max_think_tokens = (uint32_t)maxThinkTokens;

    lfm2_status_t st = lfm2_begin_chat_turn(ctx, c_msg, &sp);
    (*env)->ReleaseStringUTFChars(env, message, c_msg);
    return (jint)st;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeNextUtf8(
    JNIEnv *env, jclass clazz, jlong handle, jbyteArray outBuf, jintArray outMeta) {
    (void)clazz;
    if (!handle || !outBuf || !outMeta) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    jsize buf_len = (*env)->GetArrayLength(env, outBuf);
    if (buf_len < 8) return LFM2_ERR_INVALID_ARG;

    /* Use thread-local or static stack buffer capped at buf_len */
    char stack_buf[1024];
    char *target_buf = stack_buf;
    size_t cap = sizeof(stack_buf);
    if ((size_t)buf_len < cap) cap = (size_t)buf_len;

    size_t written = 0;
    int32_t think_phase = 0;
    lfm2_status_t st = lfm2_context_next_utf8(ctx, target_buf, cap, &written, &think_phase);

    if (written > 0) {
        (*env)->SetByteArrayRegion(env, outBuf, 0, (jsize)written, (const jbyte *)target_buf);
    }

    jint meta[2];
    meta[0] = (jint)written;
    meta[1] = (jint)think_phase;
    (*env)->SetIntArrayRegion(env, outMeta, 0, 2, meta);

    return (jint)st;
}

LFM2_JNI_EXPORT jfloatArray Java_ai_liquid_lfm2_LFM2Context_nativeEmbed(
    JNIEnv *env, jclass clazz, jlong handle, jstring text, jint poolingType) {
    (void)clazz;
    if (!handle || !text) return NULL;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    lfm2_model_info_t info;
    if (lfm2_model_get_info(ctx->model, &info) != LFM2_OK) return NULL;

    const char *c_text = (*env)->GetStringUTFChars(env, text, NULL);
    if (!c_text) return NULL;

    float *embd = (float *)malloc((size_t)info.n_embd * sizeof(float));
    if (!embd) {
        (*env)->ReleaseStringUTFChars(env, text, c_text);
        return NULL;
    }

    lfm2_status_t st = lfm2_embed_text(ctx, c_text, (int32_t)poolingType, embd);
    (*env)->ReleaseStringUTFChars(env, text, c_text);

    if (st != LFM2_OK) {
        free(embd);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, (jsize)info.n_embd);
    if (result) {
        (*env)->SetFloatArrayRegion(env, result, 0, (jsize)info.n_embd, embd);
    }
    free(embd);
    return result;
}

LFM2_JNI_EXPORT void Java_ai_liquid_lfm2_LFM2Context_nativeGetStats(
    JNIEnv *env, jclass clazz, jlong handle, jdoubleArray outStats) {
    (void)clazz;
    if (!handle || !outStats) return;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    lfm2_perf_stats_t stats;
    lfm2_context_get_stats(ctx, &stats);

    jdouble d[7];
    d[0] = (jdouble)stats.prompt_tokens;
    d[1] = (jdouble)stats.generated_tokens;
    d[2] = (jdouble)stats.current_pos;
    d[3] = (jdouble)stats.prefill_ms;
    d[4] = (jdouble)stats.decode_ms;
    d[5] = (jdouble)stats.prefill_tok_per_sec;
    d[6] = (jdouble)stats.decode_tok_per_sec;

    (*env)->SetDoubleArrayRegion(env, outStats, 0, 7, d);
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeSavePromptCache(
    JNIEnv *env, jclass clazz, jlong handle, jstring path) {
    (void)clazz;
    if (!handle || !path) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return LFM2_ERR_INVALID_ARG;

    lfm2_status_t st = lfm2_context_save_prompt_cache(ctx, c_path);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jint)st;
}

LFM2_JNI_EXPORT jint Java_ai_liquid_lfm2_LFM2Context_nativeLoadPromptCache(
    JNIEnv *env, jclass clazz, jlong handle, jstring path, jintArray tokens, jintArray outMatched) {
    (void)clazz;
    if (!handle || !path || !tokens) return LFM2_ERR_INVALID_ARG;
    lfm2_context_t *ctx = (lfm2_context_t *)(uintptr_t)handle;

    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return LFM2_ERR_INVALID_ARG;

    jsize n_prompt = (*env)->GetArrayLength(env, tokens);
    jint *raw_toks = (*env)->GetIntArrayElements(env, tokens, NULL);
    if (!raw_toks) {
        (*env)->ReleaseStringUTFChars(env, path, c_path);
        return LFM2_ERR_INVALID_ARG;
    }

    uint32_t matched = 0;
    lfm2_status_t st = lfm2_context_load_prompt_cache(
        ctx, c_path, (const uint32_t *)raw_toks, (uint32_t)n_prompt, &matched);

    (*env)->ReleaseIntArrayElements(env, tokens, raw_toks, JNI_ABORT);
    (*env)->ReleaseStringUTFChars(env, path, c_path);

    if (outMatched && (*env)->GetArrayLength(env, outMatched) >= 1) {
        jint jmatched = (jint)matched;
        (*env)->SetIntArrayRegion(env, outMatched, 0, 1, &jmatched);
    }
    return (jint)st;
}

/* ============================================================================
 * Dynamic Registration via JNI_OnLoad
 * ============================================================================ */

static const JNINativeMethod s_model_methods[] = {
    {"nativeLoadFromFile",        "(Ljava/lang/String;Z)J",       (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeLoadFromFile},
    {"nativeLoadFromFd",          "(IJZ)J",                       (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeLoadFromFd},
    {"nativeFree",                "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeFree},
    {"nativeReleasePageCache",    "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeReleasePageCache},
    {"nativeWarmupPageCache",     "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeWarmupPageCache},
    {"nativeGetProcessRss",       "()J",                          (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetProcessRss},
    {"nativeGetArch",             "(J)Ljava/lang/String;",        (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetArch},
    {"nativeGetActiveIsa",        "(J)Ljava/lang/String;",        (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetActiveIsa},
    {"nativeGetNumLayers",        "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetNumLayers},
    {"nativeGetNumShortConvLayers","(J)I",                        (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetNumShortConvLayers},
    {"nativeGetNumGqaLayers",     "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetNumGqaLayers},
    {"nativeGetHiddenDim",        "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetHiddenDim},
    {"nativeGetFfnDim",           "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetFfnDim},
    {"nativeGetVocabSize",        "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetVocabSize},
    {"nativeGetTrainCtx",         "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetTrainCtx},
    {"nativeGetNumExperts",       "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetNumExperts},
    {"nativeGetNumExpertsUsed",   "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetNumExpertsUsed},
    {"nativeGetFileSize",         "(J)J",                         (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeGetFileSize},
    {"nativeTokenize",            "(JLjava/lang/String;I)[I",     (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeTokenize},
    {"nativeCreateContext",       "(JIIIZZ)J",                    (void *)&Java_ai_liquid_lfm2_LFM2Model_nativeCreateContext}
};

static const JNINativeMethod s_ctx_methods[] = {
    {"nativeFree",                "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeFree},
    {"nativeReset",               "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeReset},
    {"nativeGetPos",              "(J)I",                         (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeGetPos},
    {"nativeSetSystemPrompt",     "(JLjava/lang/String;Ljava/lang/String;)I", (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeSetSystemPrompt},
    {"nativeClearStops",          "(J)V",                         (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeClearStops},
    {"nativeAddStop",             "(JLjava/lang/String;)I",       (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeAddStop},
    {"nativeBeginTurn",           "(JLjava/lang/String;FFIFIJII)I", (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeBeginTurn},
    {"nativeNextUtf8",            "(J[B[I)I",                     (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeNextUtf8},
    {"nativeEmbed",               "(JLjava/lang/String;I)[F",     (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeEmbed},
    {"nativeGetStats",            "(J[D)V",                       (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeGetStats},
    {"nativeSavePromptCache",     "(JLjava/lang/String;)I",       (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeSavePromptCache},
    {"nativeLoadPromptCache",     "(JLjava/lang/String;[I[I)I",   (void *)&Java_ai_liquid_lfm2_LFM2Context_nativeLoadPromptCache}
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    JNIEnv *env = NULL;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    jclass model_cls = (*env)->FindClass(env, "ai/liquid/lfm2/LFM2Model");
    if (model_cls) {
        (*env)->RegisterNatives(env, model_cls, s_model_methods,
                                sizeof(s_model_methods) / sizeof(s_model_methods[0]));
        (*env)->DeleteLocalRef(env, model_cls);
    }

    jclass ctx_cls = (*env)->FindClass(env, "ai/liquid/lfm2/LFM2Context");
    if (ctx_cls) {
        (*env)->RegisterNatives(env, ctx_cls, s_ctx_methods,
                                sizeof(s_ctx_methods) / sizeof(s_ctx_methods[0]));
        (*env)->DeleteLocalRef(env, ctx_cls);
    }

    return JNI_VERSION_1_6;
}

#endif /* LFM2_JNI_C */
