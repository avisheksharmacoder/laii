#ifndef LFM2_GUI_WORKER_H
#define LFM2_GUI_WORKER_H

/*
 * lfm2_gui_worker.h — Headless engine, model browser, SSE client, and pthread worker
 * for the LFM2/LFM3 native desktop GUI.
 *
 * Uses include/lfm2.h (liblfm2.so) for zero-copy local GGUF inference and pure POSIX
 * sockets for optional OpenAI-compatible remote server streaming (bin/lfm2-server).
 */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <netdb.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "lfm2.h"
#include "lfm_defaults.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LFM_BACKEND_LOCAL = 0,
    LFM_BACKEND_SERVER = 1
} LfmGuiBackendMode;

typedef enum {
    LFM_GUI_IDLE = 0,
    LFM_GUI_LOADING_MODEL = 1,
    LFM_GUI_GENERATING = 2,
    LFM_GUI_ERROR = 3
} LfmGuiStatus;

typedef enum {
    LFM_MSG_USER = 0,
    LFM_MSG_ASSISTANT = 1
} LfmGuiRole;

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} LfmGuiString;

typedef struct {
    LfmGuiRole   role;
    LfmGuiString think;
    LfmGuiString answer;
    bool         think_expanded;
    bool         streaming;
    char         stop_reason[64];
    uint32_t     gen_tokens;
    double       decode_tok_s;
    double       prefill_ms;
} LfmGuiMessage;

typedef struct {
    char              model_path[512];
    char              server_host[128];
    int32_t           server_port;
    char              system_prompt[512];
    int32_t           ctx_len;
    int32_t           max_tokens;   /* 0 = unlimited, default 2000 */
    int32_t           n_threads;    /* 0 = auto */
    float             temperature;
    float             min_p;
    int32_t           top_k;
    float             rep_penalty;
    bool              warmup_mmap;
    LfmGuiBackendMode backend_mode;
} LfmGuiConfig;

typedef struct {
    char     name[256];
    char     full_path[512];
    bool     is_dir;
    uint64_t size_bytes;
} LfmGuiFileEntry;

typedef struct {
    int  phase;       /* 0 = pre/answer, 1 = inside <think>, 2 = after </think> */
    char pending[16];
    size_t pending_len;
} LfmGuiThinkParser;

typedef enum {
    LFM_JOB_NONE = 0,
    LFM_JOB_LOAD_MODEL = 1,
    LFM_JOB_GENERATE = 2
} LfmGuiJobType;

typedef void (*LfmGuiWakeFn)(void *userdata);

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    pthread_t       thread;
    bool            thread_started;
    bool            quit_requested;
    atomic_bool     cancel_requested;

    LfmGuiWakeFn    wake_fn;
    void           *wake_ud;

    LfmGuiConfig    cfg;
    LfmGuiStatus    status;
    char            status_text[256];
    char            error_text[256];

    /* Local engine handles owned exclusively by worker thread when active */
    lfm2_model_t     *model;
    lfm2_context_t   *ctx;
    lfm2_model_info_t model_info;
    bool              model_loaded;
    char              loaded_path[512];
    char              loaded_sys_prompt[512];
    int32_t           loaded_ctx_len;
    int32_t           loaded_threads;

    /* Live telemetry updated safely under mutex */
    uint32_t        ctx_pos;
    uint32_t        ctx_max;
    double          rss_mb;
    double          last_prefill_ms;
    double          last_decode_tps;

    /* Conversation messages guarded by mutex */
    LfmGuiMessage  *msgs;
    size_t          n_msgs;
    size_t          cap_msgs;

    /* Pending job queue */
    LfmGuiJobType   pending_job;
    char            pending_model_path[512];
    bool            pending_reset_ctx;
} LfmGuiEngine;

/* -------- Dynamic String Helpers -------- */

static inline void lfm_gui_str_init(LfmGuiString *s) {
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

static inline void lfm_gui_str_free(LfmGuiString *s) {
    free(s->data);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

static inline void lfm_gui_str_clear(LfmGuiString *s) {
    s->len = 0;
    if (s->data) s->data[0] = '\0';
}

static inline bool lfm_gui_str_append(LfmGuiString *s, const char *chunk, size_t n) {
    if (!chunk || n == 0) return true;
    if (s->len + n + 1 > s->cap) {
        size_t next = s->cap ? s->cap * 2 : 256;
        while (next < s->len + n + 1) next *= 2;
        char *nb = (char *)realloc(s->data, next);
        if (!nb) return false;
        s->data = nb;
        s->cap = next;
    }
    memcpy(s->data + s->len, chunk, n);
    s->len += n;
    s->data[s->len] = '\0';
    return true;
}

static inline const char *lfm_gui_basename(const char *path) {
    if (!path || !path[0]) return "";
    const char *slash = strrchr(path, '/');
    return slash ? (slash + 1) : path;
}

static inline uint64_t lfm_gui_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000ULL + (uint64_t)(tv.tv_usec / 1000);
}

/* -------- GGUF File Discovery -------- */

static inline int lfm_gui_ends_with_ci(const char *s, const char *suffix) {
    size_t sl = strlen(s);
    size_t tl = strlen(suffix);
    if (sl < tl) return 0;
    return strcasecmp(s + sl - tl, suffix) == 0;
}

static int lfm_gui_file_cmp(const void *pa, const void *pb) {
    const LfmGuiFileEntry *a = (const LfmGuiFileEntry *)pa;
    const LfmGuiFileEntry *b = (const LfmGuiFileEntry *)pb;
    if (a->is_dir != b->is_dir) return a->is_dir ? -1 : 1;
    if (strcmp(a->name, "..") == 0) return -1;
    if (strcmp(b->name, "..") == 0) return 1;
    return strcasecmp(a->name, b->name);
}

static size_t lfm_gui_scan_dir(const char *dir_path, LfmGuiFileEntry *out, size_t cap) {
    if (!dir_path || !dir_path[0] || !out || cap == 0) return 0;
    DIR *dp = opendir(dir_path);
    if (!dp) return 0;

    char resolved_dir[512];
    char *rp = realpath(dir_path, NULL);
    if (rp) {
        snprintf(resolved_dir, sizeof(resolved_dir), "%s", rp);
        free(rp);
    } else {
        snprintf(resolved_dir, sizeof(resolved_dir), "%s", dir_path);
    }

    size_t count = 0;
    if (strcmp(resolved_dir, "/") != 0 && count < cap) {
        LfmGuiFileEntry *e = &out[count++];
        memset(e, 0, sizeof(*e));
        snprintf(e->name, sizeof(e->name), "..");
        snprintf(e->full_path, sizeof(e->full_path), "%.*s", (int)sizeof(e->full_path) - 4, resolved_dir);
        char *last = strrchr(e->full_path, '/');
        if (last && last != e->full_path) *last = '\0';
        else snprintf(e->full_path, sizeof(e->full_path), "/");
        e->is_dir = true;
    }

    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL && count < cap) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        if (ent->d_name[0] == '.') continue;

        char full[512];
        if (strcmp(resolved_dir, "/") == 0) {
            snprintf(full, sizeof(full), "/%.*s", (int)sizeof(full) - 2, ent->d_name);
        } else {
            size_t dl = strlen(resolved_dir);
            if (dl + 1 + strlen(ent->d_name) >= sizeof(full)) continue;
            memcpy(full, resolved_dir, dl);
            full[dl] = '/';
            snprintf(full + dl + 1, sizeof(full) - dl - 1, "%s", ent->d_name);
        }

        struct stat st;
        if (stat(full, &st) != 0) continue;
        bool is_dir = S_ISDIR(st.st_mode);
        if (!is_dir && !lfm_gui_ends_with_ci(ent->d_name, ".gguf")) continue;

        LfmGuiFileEntry *e = &out[count++];
        memset(e, 0, sizeof(*e));
        snprintf(e->name, sizeof(e->name), "%s", ent->d_name);
        snprintf(e->full_path, sizeof(e->full_path), "%s", full);
        e->is_dir = is_dir;
        e->size_bytes = is_dir ? 0ULL : (uint64_t)st.st_size;
    }

    closedir(dp);
    if (count > 1) {
        qsort(out, count, sizeof(LfmGuiFileEntry), lfm_gui_file_cmp);
    }
    return count;
}

static bool lfm_gui_pick_default_model(char *out, size_t cap) {
    if (!out || cap == 0) return false;
    out[0] = '\0';

    const char *preferred = "models/LFM3-1.7B-Q4_K_M.gguf";
    struct stat st;
    if (stat(preferred, &st) == 0 && S_ISREG(st.st_mode)) {
        snprintf(out, cap, "%s", preferred);
        return true;
    }

    LfmGuiFileEntry entries[64];
    size_t n = lfm_gui_scan_dir("models", entries, 64);
    for (size_t i = 0; i < n; ++i) {
        if (!entries[i].is_dir && lfm_gui_ends_with_ci(entries[i].name, ".gguf")) {
            snprintf(out, cap, "%s", entries[i].full_path);
            return true;
        }
    }
    return false;
}

/* -------- Streaming <think> Splitter + SSE JSON Parser -------- */

static void lfm_gui_think_parser_init(LfmGuiThinkParser *tp) {
    memset(tp, 0, sizeof(*tp));
}

static void lfm_gui_route_piece(LfmGuiString *think, LfmGuiString *answer, bool is_think,
                                const char *s, size_t n) {
    if (!s || n == 0) return;
    lfm_gui_str_append(is_think ? think : answer, s, n);
}

static void lfm_gui_think_parser_feed(LfmGuiThinkParser *tp, const char *chunk, size_t n,
                                      LfmGuiString *think, LfmGuiString *answer) {
    static const char OPEN_TAG[] = "<think>";
    static const char CLOSE_TAG[] = "</think>";

    for (size_t i = 0; i < n; ++i) {
        char c = chunk[i];
        if (tp->pending_len < sizeof(tp->pending) - 1) {
            tp->pending[tp->pending_len++] = c;
            tp->pending[tp->pending_len] = '\0';
        }

        const char *target = (tp->phase == 1) ? CLOSE_TAG : OPEN_TAG;
        size_t tlen = strlen(target);
        if (tp->pending_len <= tlen && strncmp(tp->pending, target, tp->pending_len) == 0) {
            if (tp->pending_len == tlen) {
                tp->phase = (tp->phase == 1) ? 2 : 1;
                tp->pending_len = 0;
            }
            continue;
        }

        /* Mismatch: flush pending prefix cleanly */
        lfm_gui_route_piece(think, answer, tp->phase == 1, tp->pending, tp->pending_len);
        tp->pending_len = 0;
    }
}

static void lfm_gui_think_parser_flush(LfmGuiThinkParser *tp, LfmGuiString *think, LfmGuiString *answer) {
    if (tp->pending_len > 0) {
        lfm_gui_route_piece(think, answer, tp->phase == 1, tp->pending, tp->pending_len);
        tp->pending_len = 0;
    }
}

static bool lfm_gui_json_unescape(const char *src, const char *limit, char *out, size_t cap,
                                  size_t *out_len, const char **out_end) {
    size_t w = 0;
    const char *p = src;
    while (p < limit && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c == '\\' && p < limit) {
            char esc = *p++;
            switch (esc) {
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case '\\': c = '\\'; break;
                case '"':  c = '"'; break;
                case '/':  c = '/'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'u': {
                    if (p + 4 <= limit) {
                        unsigned int cp = 0;
                        if (sscanf(p, "%4x", &cp) == 1) {
                            p += 4;
                            if (cp < 0x80 && w + 1 < cap) {
                                out[w++] = (char)cp;
                            } else if (cp < 0x800 && w + 2 < cap) {
                                out[w++] = (char)(0xC0 | (cp >> 6));
                                out[w++] = (char)(0x80 | (cp & 0x3F));
                            } else if (w + 3 < cap) {
                                out[w++] = (char)(0xE0 | (cp >> 12));
                                out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                out[w++] = (char)(0x80 | (cp & 0x3F));
                            }
                        }
                    }
                    continue;
                }
                default: c = (unsigned char)esc; break;
            }
        }
        if (w + 1 < cap) out[w++] = (char)c;
    }
    if (w < cap) out[w] = '\0';
    if (out_len) *out_len = w;
    if (out_end) *out_end = p;
    return (p < limit && *p == '"');
}

static const char *lfm_gui_find_json_str_field(const char *json, const char *key,
                                               char *out, size_t cap, size_t *out_len) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *hit = strstr(json, pat);
    if (!hit) return NULL;
    const char *p = hit + strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == ':' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return NULL;
    p++;
    const char *limit = json + strlen(json);
    if (!lfm_gui_json_unescape(p, limit, out, cap, out_len, NULL)) return NULL;
    return out;
}

static bool lfm_gui_extract_sse_line(const char *line, char *out_delta, size_t delta_cap,
                                     size_t *out_delta_len, char *out_finish, size_t finish_cap,
                                     bool *out_done) {
    if (out_delta_len) *out_delta_len = 0;
    if (out_delta && delta_cap > 0) out_delta[0] = '\0';
    if (out_finish && finish_cap > 0) out_finish[0] = '\0';
    if (out_done) *out_done = false;
    if (!line) return false;

    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (strncmp(p, "data:", 5) != 0) return false;
    p += 5;
    while (*p == ' ' || *p == '\t') p++;

    if (strncmp(p, "[DONE]", 6) == 0) {
        if (out_done) *out_done = true;
        return true;
    }

    bool any = false;
    const char *delta = strstr(p, "\"delta\"");
    if (delta) {
        size_t dl = 0;
        if (lfm_gui_find_json_str_field(delta, "content", out_delta, delta_cap, &dl)) {
            if (out_delta_len) *out_delta_len = dl;
            any = true;
        }
    }
    char fin[32];
    size_t fl = 0;
    if (out_finish && finish_cap > 0 &&
        lfm_gui_find_json_str_field(p, "finish_reason", fin, sizeof(fin), &fl) && fl > 0) {
        snprintf(out_finish, finish_cap, "%s", fin);
        any = true;
    }
    return any;
}

static void lfm_gui_json_append_escaped(LfmGuiString *dst, const char *s) {
    lfm_gui_str_append(dst, "\"", 1);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
            unsigned char c = *p;
            if (c == '"') lfm_gui_str_append(dst, "\\\"", 2);
            else if (c == '\\') lfm_gui_str_append(dst, "\\\\", 2);
            else if (c == '\n') lfm_gui_str_append(dst, "\\n", 2);
            else if (c == '\r') lfm_gui_str_append(dst, "\\r", 2);
            else if (c == '\t') lfm_gui_str_append(dst, "\\t", 2);
            else if (c < 0x20) {
                char u[8];
                snprintf(u, sizeof(u), "\\u%04x", (unsigned int)c);
                lfm_gui_str_append(dst, u, 6);
            } else {
                lfm_gui_str_append(dst, (const char *)&c, 1);
            }
        }
    }
    lfm_gui_str_append(dst, "\"", 1);
}

/* -------- Config & Message Array Management -------- */

static void lfm_gui_config_defaults(LfmGuiConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    lfm_gui_pick_default_model(cfg->model_path, sizeof(cfg->model_path));
    snprintf(cfg->server_host, sizeof(cfg->server_host), "127.0.0.1");
    cfg->server_port = 8080;
    snprintf(cfg->system_prompt, sizeof(cfg->system_prompt), "You are a helpful AI assistant.");
    cfg->ctx_len = (int32_t)LFM_DEFAULT_CTX_LEN;
    cfg->max_tokens = (int32_t)LFM_DEFAULT_MAX_TOKENS;
    cfg->n_threads = 0;
    cfg->temperature = 0.30f;
    cfg->min_p = 0.15f;
    cfg->top_k = 50;
    cfg->rep_penalty = 1.05f;
    cfg->warmup_mmap = false;
    cfg->backend_mode = LFM_BACKEND_LOCAL;
}

static size_t lfm_gui_add_msg_locked(LfmGuiEngine *eng, LfmGuiRole role, const char *text) {
    if (eng->n_msgs + 1 > eng->cap_msgs) {
        size_t next = eng->cap_msgs ? eng->cap_msgs * 2 : 16;
        LfmGuiMessage *nm = (LfmGuiMessage *)realloc(eng->msgs, next * sizeof(LfmGuiMessage));
        if (!nm) return (size_t)-1;
        eng->msgs = nm;
        eng->cap_msgs = next;
    }
    LfmGuiMessage *m = &eng->msgs[eng->n_msgs];
    memset(m, 0, sizeof(*m));
    m->role = role;
    lfm_gui_str_init(&m->think);
    lfm_gui_str_init(&m->answer);
    if (text && text[0]) {
        lfm_gui_str_append(&m->answer, text, strlen(text));
    }
    return eng->n_msgs++;
}

static void lfm_gui_clear_msgs_locked(LfmGuiEngine *eng) {
    for (size_t i = 0; i < eng->n_msgs; ++i) {
        lfm_gui_str_free(&eng->msgs[i].think);
        lfm_gui_str_free(&eng->msgs[i].answer);
    }
    eng->n_msgs = 0;
}

static void lfm_gui_wake_ui(LfmGuiEngine *eng) {
    if (eng->wake_fn) eng->wake_fn(eng->wake_ud);
}

/* -------- Worker Job Executors -------- */

static void lfm_gui_worker_unload_model(LfmGuiEngine *eng) {
    if (eng->ctx) {
        lfm2_context_free(eng->ctx);
        eng->ctx = NULL;
    }
    if (eng->model) {
        lfm2_model_free(eng->model);
        eng->model = NULL;
    }
    pthread_mutex_lock(&eng->mu);
    eng->model_loaded = false;
    eng->loaded_path[0] = '\0';
    memset(&eng->model_info, 0, sizeof(eng->model_info));
    eng->ctx_pos = 0;
    eng->ctx_max = (uint32_t)(eng->cfg.ctx_len > 0 ? eng->cfg.ctx_len : (int32_t)LFM_DEFAULT_CTX_LEN);
    pthread_mutex_unlock(&eng->mu);
}

static bool lfm_gui_worker_ensure_model(LfmGuiEngine *eng, const char *path,
                                        int32_t req_ctx, int32_t req_threads,
                                        bool warmup_mmap, const char *sys_prompt) {
    bool need_model = (!eng->model || strcmp(eng->loaded_path, path) != 0);
    bool need_ctx = need_model || (!eng->ctx) || (eng->loaded_ctx_len != req_ctx) ||
                    (eng->loaded_threads != req_threads);

    if (need_model) {
        pthread_mutex_lock(&eng->mu);
        eng->status = LFM_GUI_LOADING_MODEL;
        snprintf(eng->status_text, sizeof(eng->status_text), "Loading %s...", lfm_gui_basename(path));
        eng->error_text[0] = '\0';
        pthread_mutex_unlock(&eng->mu);
        lfm_gui_wake_ui(eng);

        lfm_gui_worker_unload_model(eng);

        lfm2_model_t *m = lfm2_model_load(path, warmup_mmap ? 1 : 0);
        if (!m) {
            pthread_mutex_lock(&eng->mu);
            eng->status = LFM_GUI_ERROR;
            snprintf(eng->status_text, sizeof(eng->status_text), "Model load failed");
            snprintf(eng->error_text, sizeof(eng->error_text), "Failed to open/parse GGUF: %s", path);
            pthread_mutex_unlock(&eng->mu);
            lfm_gui_wake_ui(eng);
            return false;
        }
        eng->model = m;
        lfm2_model_get_info(m, &eng->model_info);
        snprintf(eng->loaded_path, sizeof(eng->loaded_path), "%s", path);
        need_ctx = true;
    }

    if (need_ctx) {
        if (eng->ctx) {
            lfm2_context_free(eng->ctx);
            eng->ctx = NULL;
        }
        lfm2_context_params_t cp = lfm2_context_default_params();
        cp.max_seq_len = (uint32_t)lfm_resolve_ctx_len(
            req_ctx > 0 ? (uint32_t)req_ctx : LFM_DEFAULT_CTX_LEN,
            eng->model_info.n_ctx_train);
        cp.n_threads = req_threads > 0 ? (uint32_t)req_threads : 0U;
        cp.warmup_mmap = warmup_mmap ? 1 : 0;

        lfm2_context_t *c = lfm2_context_create(eng->model, &cp);
        if (!c) {
            pthread_mutex_lock(&eng->mu);
            eng->status = LFM_GUI_ERROR;
            snprintf(eng->status_text, sizeof(eng->status_text), "Context allocation failed");
            snprintf(eng->error_text, sizeof(eng->error_text), "Failed to create KV context (len=%u)", cp.max_seq_len);
            pthread_mutex_unlock(&eng->mu);
            lfm_gui_wake_ui(eng);
            return false;
        }
        eng->ctx = c;
        eng->loaded_ctx_len = req_ctx;
        eng->loaded_threads = req_threads;
        eng->loaded_sys_prompt[0] = '\0';

        pthread_mutex_lock(&eng->mu);
        eng->model_loaded = true;
        eng->ctx_max = cp.max_seq_len;
        eng->ctx_pos = 0;
        eng->rss_mb = (double)lfm2_get_process_rss_bytes() / (1024.0 * 1024.0);
        pthread_mutex_unlock(&eng->mu);
    }

    if (eng->ctx) {
        if (eng->pending_reset_ctx) {
            lfm2_context_reset(eng->ctx);
            eng->pending_reset_ctx = false;
            eng->loaded_sys_prompt[0] = '\0';
        }
        if (strcmp(eng->loaded_sys_prompt, sys_prompt) != 0) {
            lfm2_context_set_system_prompt(eng->ctx, sys_prompt[0] ? sys_prompt : NULL, NULL);
            snprintf(eng->loaded_sys_prompt, sizeof(eng->loaded_sys_prompt), "%s", sys_prompt);
        }
    }

    pthread_mutex_lock(&eng->mu);
    if (eng->status == LFM_GUI_LOADING_MODEL) {
        eng->status = LFM_GUI_IDLE;
        snprintf(eng->status_text, sizeof(eng->status_text), "Ready (%s)", lfm_gui_basename(path));
    }
    eng->rss_mb = (double)lfm2_get_process_rss_bytes() / (1024.0 * 1024.0);
    pthread_mutex_unlock(&eng->mu);
    lfm_gui_wake_ui(eng);
    return true;
}

static void lfm_gui_worker_run_local(LfmGuiEngine *eng, const LfmGuiConfig *cfg,
                                     size_t msg_idx, const char *prompt) {
    if (!lfm_gui_worker_ensure_model(eng, cfg->model_path, cfg->ctx_len, cfg->n_threads,
                                     cfg->warmup_mmap, cfg->system_prompt)) {
        pthread_mutex_lock(&eng->mu);
        if (msg_idx < eng->n_msgs) {
            eng->msgs[msg_idx].streaming = false;
            snprintf(eng->msgs[msg_idx].stop_reason, sizeof(eng->msgs[msg_idx].stop_reason), "error");
        }
        pthread_mutex_unlock(&eng->mu);
        lfm_gui_wake_ui(eng);
        return;
    }

    lfm2_sample_params_t sp = lfm2_sample_default_params();
    sp.temperature = cfg->temperature;
    sp.min_p = cfg->min_p;
    sp.top_k = cfg->top_k > 0 ? (uint32_t)cfg->top_k : 0U;
    sp.rep_penalty = cfg->rep_penalty;

    pthread_mutex_lock(&eng->mu);
    eng->status = LFM_GUI_GENERATING;
    snprintf(eng->status_text, sizeof(eng->status_text), "Generating...");
    pthread_mutex_unlock(&eng->mu);
    lfm_gui_wake_ui(eng);

    lfm2_status_t st = lfm2_begin_chat_turn(eng->ctx, prompt, &sp);
    if (st != LFM2_OK) {
        pthread_mutex_lock(&eng->mu);
        if (msg_idx < eng->n_msgs) {
            eng->msgs[msg_idx].streaming = false;
            snprintf(eng->msgs[msg_idx].stop_reason, sizeof(eng->msgs[msg_idx].stop_reason),
                     st == LFM2_ERR_CTX_FULL ? "context_full" : "error");
        }
        eng->status = LFM_GUI_ERROR;
        snprintf(eng->error_text, sizeof(eng->error_text), "Chat turn failed (code %d)", (int)st);
        snprintf(eng->status_text, sizeof(eng->status_text), "Turn failed");
        pthread_mutex_unlock(&eng->mu);
        lfm_gui_wake_ui(eng);
        return;
    }

    uint32_t gen_tokens = 0;
    uint32_t max_toks = cfg->max_tokens > 0 ? (uint32_t)cfg->max_tokens : 0U;
    const char *stop_reason = "stop";
    uint64_t last_wake = lfm_gui_now_ms();

    while (true) {
        if (atomic_load(&eng->cancel_requested)) {
            stop_reason = "cancelled";
            break;
        }
        if (max_toks > 0 && gen_tokens >= max_toks) {
            stop_reason = "length";
            break;
        }

        char utf8[512];
        size_t written = 0;
        int32_t think_phase = 0;
        st = lfm2_context_next_utf8(eng->ctx, utf8, sizeof(utf8), &written, &think_phase);
        if (st == LFM2_DONE) {
            if (written > 0) {
                pthread_mutex_lock(&eng->mu);
                if (msg_idx < eng->n_msgs) {
                    LfmGuiMessage *m = &eng->msgs[msg_idx];
                    lfm_gui_str_append(think_phase ? &m->think : &m->answer, utf8, written);
                }
                pthread_mutex_unlock(&eng->mu);
            }
            stop_reason = "stop";
            break;
        }
        if (st == LFM2_ERR_CTX_FULL) {
            stop_reason = "context_full";
            break;
        }
        if (st != LFM2_OK) {
            stop_reason = "error";
            break;
        }

        gen_tokens++;
        lfm2_perf_stats_t ps;
        lfm2_context_get_stats(eng->ctx, &ps);

        pthread_mutex_lock(&eng->mu);
        if (msg_idx < eng->n_msgs) {
            LfmGuiMessage *m = &eng->msgs[msg_idx];
            if (written > 0) {
                lfm_gui_str_append(think_phase ? &m->think : &m->answer, utf8, written);
            }
            m->gen_tokens = gen_tokens;
            m->prefill_ms = ps.prefill_ms;
            m->decode_tok_s = ps.decode_tok_per_sec;
        }
        eng->ctx_pos = ps.current_pos;
        eng->last_prefill_ms = ps.prefill_ms;
        eng->last_decode_tps = ps.decode_tok_per_sec;
        pthread_mutex_unlock(&eng->mu);

        uint64_t now = lfm_gui_now_ms();
        if (now - last_wake >= 16) {
            last_wake = now;
            lfm_gui_wake_ui(eng);
        }
    }

    lfm2_perf_stats_t final_ps;
    lfm2_context_get_stats(eng->ctx, &final_ps);

    pthread_mutex_lock(&eng->mu);
    if (msg_idx < eng->n_msgs) {
        LfmGuiMessage *m = &eng->msgs[msg_idx];
        m->streaming = false;
        m->gen_tokens = gen_tokens;
        m->prefill_ms = final_ps.prefill_ms;
        m->decode_tok_s = final_ps.decode_tok_per_sec;
        snprintf(m->stop_reason, sizeof(m->stop_reason), "%s", stop_reason);
    }
    eng->ctx_pos = final_ps.current_pos;
    eng->last_prefill_ms = final_ps.prefill_ms;
    eng->last_decode_tps = final_ps.decode_tok_per_sec;
    eng->rss_mb = (double)lfm2_get_process_rss_bytes() / (1024.0 * 1024.0);
    eng->status = LFM_GUI_IDLE;
    snprintf(eng->status_text, sizeof(eng->status_text), "Ready");
    pthread_mutex_unlock(&eng->mu);
    lfm_gui_wake_ui(eng);
}

static int lfm_gui_tcp_connect(const char *host, int32_t port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_s[16];
    snprintf(port_s, sizeof(port_s), "%d", port);
    if (getaddrinfo(host, port_s, &hints, &res) != 0) return -1;

    int fd = -1;
    for (struct addrinfo *it = res; it; it = it->ai_next) {
        fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = {5, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static void lfm_gui_build_chat_payload_locked(const LfmGuiEngine *eng, const LfmGuiConfig *cfg,
                                              size_t up_to_msg_idx, LfmGuiString *out_json) {
    char buf[256];
    lfm_gui_str_append(out_json, "{\"model\":\"lfm-local\",\"stream\":true,", 33);
    if (cfg->max_tokens > 0) {
        snprintf(buf, sizeof(buf), "\"max_tokens\":%d,", cfg->max_tokens);
        lfm_gui_str_append(out_json, buf, strlen(buf));
    }
    snprintf(buf, sizeof(buf), "\"temperature\":%.3f,\"min_p\":%.3f,\"top_k\":%d,\"messages\":[",
             cfg->temperature, cfg->min_p, cfg->top_k);
    lfm_gui_str_append(out_json, buf, strlen(buf));

    bool need_comma = false;
    if (cfg->system_prompt[0]) {
        lfm_gui_str_append(out_json, "{\"role\":\"system\",\"content\":", 27);
        lfm_gui_json_append_escaped(out_json, cfg->system_prompt);
        lfm_gui_str_append(out_json, "}", 1);
        need_comma = true;
    }
    for (size_t i = 0; i < up_to_msg_idx && i < eng->n_msgs; ++i) {
        const LfmGuiMessage *m = &eng->msgs[i];
        const char *role_str = (m->role == LFM_MSG_USER) ? "user" : "assistant";
        if (need_comma) lfm_gui_str_append(out_json, ",", 1);
        lfm_gui_str_append(out_json, "{\"role\":\"", 9);
        lfm_gui_str_append(out_json, role_str, strlen(role_str));
        lfm_gui_str_append(out_json, "\",\"content\":", 12);
        lfm_gui_json_append_escaped(out_json, m->answer.data ? m->answer.data : "");
        lfm_gui_str_append(out_json, "}", 1);
        need_comma = true;
    }
    lfm_gui_str_append(out_json, "]}", 2);
}

static void lfm_gui_worker_run_server(LfmGuiEngine *eng, const LfmGuiConfig *cfg, size_t msg_idx) {
    LfmGuiString body;
    lfm_gui_str_init(&body);

    pthread_mutex_lock(&eng->mu);
    eng->status = LFM_GUI_GENERATING;
    snprintf(eng->status_text, sizeof(eng->status_text), "Streaming from %s:%d...", cfg->server_host, cfg->server_port);
    lfm_gui_build_chat_payload_locked(eng, cfg, msg_idx, &body);
    pthread_mutex_unlock(&eng->mu);
    lfm_gui_wake_ui(eng);

    int fd = lfm_gui_tcp_connect(cfg->server_host, cfg->server_port);
    if (fd < 0) {
        lfm_gui_str_free(&body);
        pthread_mutex_lock(&eng->mu);
        if (msg_idx < eng->n_msgs) {
            eng->msgs[msg_idx].streaming = false;
            snprintf(eng->msgs[msg_idx].stop_reason, sizeof(eng->msgs[msg_idx].stop_reason), "error");
        }
        eng->status = LFM_GUI_ERROR;
        snprintf(eng->error_text, sizeof(eng->error_text), "Cannot connect to %s:%d (start bin/lfm2-server first)",
                 cfg->server_host, cfg->server_port);
        snprintf(eng->status_text, sizeof(eng->status_text), "Server unreachable");
        pthread_mutex_unlock(&eng->mu);
        lfm_gui_wake_ui(eng);
        return;
    }

    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
                        "POST /v1/chat/completions HTTP/1.1\r\n"
                        "Host: %s:%d\r\n"
                        "Content-Type: application/json\r\n"
                        "Content-Length: %zu\r\n"
                        "Connection: close\r\n\r\n",
                        cfg->server_host, cfg->server_port, body.len);
    send(fd, hdr, (size_t)hlen, 0);
    if (body.data && body.len > 0) send(fd, body.data, body.len, 0);
    lfm_gui_str_free(&body);

    LfmGuiString rx;
    lfm_gui_str_init(&rx);
    LfmGuiThinkParser tp;
    lfm_gui_think_parser_init(&tp);

    bool headers_done = false;
    bool stream_done = false;
    const char *stop_reason = "stop";
    uint64_t t_start = lfm_gui_now_ms();
    uint32_t chunks = 0;

    while (!stream_done) {
        if (atomic_load(&eng->cancel_requested)) {
            stop_reason = "cancelled";
            break;
        }
        char buf[4096];
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            stop_reason = "error";
            break;
        }
        if (n == 0) break;
        lfm_gui_str_append(&rx, buf, (size_t)n);

        if (!headers_done) {
            const char *sep = strstr(rx.data, "\r\n\r\n");
            if (!sep) continue;
            headers_done = true;
            size_t body_off = (size_t)(sep - rx.data) + 4;
            memmove(rx.data, rx.data + body_off, rx.len - body_off + 1);
            rx.len -= body_off;
        }

        char *nl;
        while (rx.data && (nl = strchr(rx.data, '\n')) != NULL) {
            size_t line_len = (size_t)(nl - rx.data);
            if (line_len > 0 && rx.data[line_len - 1] == '\r') rx.data[line_len - 1] = '\0';
            rx.data[line_len] = '\0';

            char delta[2048];
            char fin[32];
            size_t dlen = 0;
            bool done = false;
            if (lfm_gui_extract_sse_line(rx.data, delta, sizeof(delta), &dlen, fin, sizeof(fin), &done)) {
                if (done) {
                    stream_done = true;
                } else {
                    if (fin[0]) {
                        stop_reason = (strcmp(fin, "length") == 0) ? "length" : "stop";
                    }
                    if (dlen > 0) {
                        chunks++;
                        pthread_mutex_lock(&eng->mu);
                        if (msg_idx < eng->n_msgs) {
                            LfmGuiMessage *m = &eng->msgs[msg_idx];
                            lfm_gui_think_parser_feed(&tp, delta, dlen, &m->think, &m->answer);
                            double sec = (double)(lfm_gui_now_ms() - t_start) / 1000.0;
                            m->gen_tokens = chunks;
                            m->decode_tok_s = sec > 0.05 ? ((double)chunks / sec) : 0.0;
                            eng->last_decode_tps = m->decode_tok_s;
                        }
                        pthread_mutex_unlock(&eng->mu);
                        lfm_gui_wake_ui(eng);
                    }
                }
            }

            size_t adv = line_len + 1;
            memmove(rx.data, rx.data + adv, rx.len - adv + 1);
            rx.len -= adv;
        }
    }

    close(fd);
    lfm_gui_str_free(&rx);

    pthread_mutex_lock(&eng->mu);
    if (msg_idx < eng->n_msgs) {
        LfmGuiMessage *m = &eng->msgs[msg_idx];
        lfm_gui_think_parser_flush(&tp, &m->think, &m->answer);
        m->streaming = false;
        snprintf(m->stop_reason, sizeof(m->stop_reason), "%s", stop_reason);
    }
    eng->status = LFM_GUI_IDLE;
    snprintf(eng->status_text, sizeof(eng->status_text), "Ready (Server %s:%d)", cfg->server_host, cfg->server_port);
    pthread_mutex_unlock(&eng->mu);
    lfm_gui_wake_ui(eng);
}

static void *lfm_gui_worker_main(void *arg) {
    LfmGuiEngine *eng = (LfmGuiEngine *)arg;
    while (true) {
        pthread_mutex_lock(&eng->mu);
        while (!eng->quit_requested && eng->pending_job == LFM_JOB_NONE) {
            pthread_cond_wait(&eng->cv, &eng->mu);
        }
        if (eng->quit_requested) {
            pthread_mutex_unlock(&eng->mu);
            break;
        }

        LfmGuiJobType job = eng->pending_job;
        eng->pending_job = LFM_JOB_NONE;
        LfmGuiConfig cfg = eng->cfg;
        char model_req[512];
        snprintf(model_req, sizeof(model_req), "%s", eng->pending_model_path);

        size_t assistant_idx = (size_t)-1;
        LfmGuiString prompt_copy;
        lfm_gui_str_init(&prompt_copy);
        if (job == LFM_JOB_GENERATE && eng->n_msgs >= 2) {
            assistant_idx = eng->n_msgs - 1;
            const LfmGuiMessage *u = &eng->msgs[eng->n_msgs - 2];
            if (u->answer.data && u->answer.len > 0) {
                lfm_gui_str_append(&prompt_copy, u->answer.data, u->answer.len);
            }
        }
        atomic_store(&eng->cancel_requested, false);
        pthread_mutex_unlock(&eng->mu);

        if (job == LFM_JOB_LOAD_MODEL && model_req[0]) {
            lfm_gui_worker_ensure_model(eng, model_req, cfg.ctx_len, cfg.n_threads,
                                        cfg.warmup_mmap, cfg.system_prompt);
        } else if (job == LFM_JOB_GENERATE && assistant_idx != (size_t)-1) {
            if (cfg.backend_mode == LFM_BACKEND_SERVER) {
                lfm_gui_worker_run_server(eng, &cfg, assistant_idx);
            } else {
                lfm_gui_worker_run_local(eng, &cfg, assistant_idx, prompt_copy.data ? prompt_copy.data : "");
            }
        }
        lfm_gui_str_free(&prompt_copy);
    }

    lfm_gui_worker_unload_model(eng);
    return NULL;
}

/* -------- Public Control API -------- */

static void lfm_gui_engine_init(LfmGuiEngine *eng, const LfmGuiConfig *cfg,
                                LfmGuiWakeFn wake_fn, void *wake_ud) {
    memset(eng, 0, sizeof(*eng));
    pthread_mutex_init(&eng->mu, NULL);
    pthread_cond_init(&eng->cv, NULL);
    atomic_init(&eng->cancel_requested, false);
    eng->wake_fn = wake_fn;
    eng->wake_ud = wake_ud;
    if (cfg) eng->cfg = *cfg;
    else lfm_gui_config_defaults(&eng->cfg);

    eng->status = LFM_GUI_IDLE;
    eng->ctx_max = (uint32_t)(eng->cfg.ctx_len > 0 ? eng->cfg.ctx_len : (int32_t)LFM_DEFAULT_CTX_LEN);
    eng->rss_mb = (double)lfm2_get_process_rss_bytes() / (1024.0 * 1024.0);
    if (eng->cfg.backend_mode == LFM_BACKEND_SERVER) {
        snprintf(eng->status_text, sizeof(eng->status_text), "Server %.120s:%d",
                 eng->cfg.server_host, eng->cfg.server_port);
    } else if (eng->cfg.model_path[0]) {
        snprintf(eng->status_text, sizeof(eng->status_text), "Ready to load %.200s",
                 lfm_gui_basename(eng->cfg.model_path));
    } else {
        snprintf(eng->status_text, sizeof(eng->status_text), "Select a .gguf model to begin");
    }

    if (pthread_create(&eng->thread, NULL, lfm_gui_worker_main, eng) == 0) {
        eng->thread_started = true;
    }
}

static void lfm_gui_engine_destroy(LfmGuiEngine *eng) {
    if (eng->thread_started) {
        pthread_mutex_lock(&eng->mu);
        eng->quit_requested = true;
        atomic_store(&eng->cancel_requested, true);
        pthread_cond_broadcast(&eng->cv);
        pthread_mutex_unlock(&eng->mu);
        pthread_join(eng->thread, NULL);
        eng->thread_started = false;
    }
    lfm_gui_clear_msgs_locked(eng);
    free(eng->msgs);
    eng->msgs = NULL;
    pthread_cond_destroy(&eng->cv);
    pthread_mutex_destroy(&eng->mu);
}

static void lfm_gui_engine_load_model(LfmGuiEngine *eng, const char *path) {
    if (!path || !path[0]) return;
    pthread_mutex_lock(&eng->mu);
    atomic_store(&eng->cancel_requested, true);
    snprintf(eng->cfg.model_path, sizeof(eng->cfg.model_path), "%s", path);
    snprintf(eng->pending_model_path, sizeof(eng->pending_model_path), "%s", path);
    eng->cfg.backend_mode = LFM_BACKEND_LOCAL;
    eng->pending_job = LFM_JOB_LOAD_MODEL;
    pthread_cond_broadcast(&eng->cv);
    pthread_mutex_unlock(&eng->mu);
}

static bool lfm_gui_engine_send_prompt(LfmGuiEngine *eng, const char *prompt) {
    if (!prompt || !prompt[0]) return false;
    pthread_mutex_lock(&eng->mu);
    if (eng->status == LFM_GUI_GENERATING || eng->status == LFM_GUI_LOADING_MODEL) {
        pthread_mutex_unlock(&eng->mu);
        return false;
    }
    if (eng->cfg.backend_mode == LFM_BACKEND_LOCAL && !eng->cfg.model_path[0]) {
        eng->status = LFM_GUI_ERROR;
        snprintf(eng->error_text, sizeof(eng->error_text), "No GGUF model selected");
        pthread_mutex_unlock(&eng->mu);
        return false;
    }
    eng->error_text[0] = '\0';
    lfm_gui_add_msg_locked(eng, LFM_MSG_USER, prompt);
    size_t aidx = lfm_gui_add_msg_locked(eng, LFM_MSG_ASSISTANT, "");
    if (aidx != (size_t)-1) {
        eng->msgs[aidx].streaming = true;
    }
    eng->pending_job = LFM_JOB_GENERATE;
    pthread_cond_broadcast(&eng->cv);
    pthread_mutex_unlock(&eng->mu);
    return true;
}

static void lfm_gui_engine_cancel(LfmGuiEngine *eng) {
    atomic_store(&eng->cancel_requested, true);
}

static void lfm_gui_engine_reset_chat(LfmGuiEngine *eng) {
    pthread_mutex_lock(&eng->mu);
    atomic_store(&eng->cancel_requested, true);
    lfm_gui_clear_msgs_locked(eng);
    eng->pending_reset_ctx = true;
    eng->ctx_pos = 0;
    eng->error_text[0] = '\0';
    if (eng->status == LFM_GUI_ERROR) {
        eng->status = LFM_GUI_IDLE;
        snprintf(eng->status_text, sizeof(eng->status_text), "Ready");
    }
    pthread_mutex_unlock(&eng->mu);
}

#ifdef __cplusplus
}
#endif

#endif /* LFM2_GUI_WORKER_H */

