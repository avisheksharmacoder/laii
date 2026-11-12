#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef LFM2_H
#if __has_include("include/lfm2.h")
#include "include/lfm2.h"
#elif __has_include("lfm2.h")
#include "lfm2.h"
#elif __has_include("../include/lfm2.h")
#include "../include/lfm2.h"
#endif
#endif

#ifndef LFM_DEFAULTS_H
#if defined(__has_include)
#  if __has_include("lfm_defaults.h")
#    include "lfm_defaults.h"
#  elif __has_include("include/lfm_defaults.h")
#    include "include/lfm_defaults.h"
#  elif __has_include("../include/lfm_defaults.h")
#    include "../include/lfm_defaults.h"
#  endif
#else
#  include "lfm_defaults.h"
#endif
#endif

static uint32_t g_default_max_tokens = LFM_DEFAULT_MAX_TOKENS;

#define LFM2_MAX_CONNS        32
#define LFM2_REQ_BUF_CAP      (64 * 1024)  /* 64 KB max request payload */
#define LFM2_RESP_BUF_CAP     (128 * 1024) /* 128 KB max buffered response (embeddings) */
#define LFM2_SLOWLORIS_SEC    10.0         /* Max seconds to finish sending HTTP headers */

static volatile sig_atomic_t g_server_stop = 0;

static void server_sigint_handler(int sig) {
    (void)sig;
    g_server_stop = 1;
}

static double monotonic_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================================
 * 1. Embedded Zero-Dependency HTML/JS Chat UI (Served at GET /)
 * ============================================================================ */
static const char LFM2_INDEX_HTML[] =
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>LFM2 Local Engine</title><style>"
    ":root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--txt:#e6edf3;--muted:#8b949e;--acc:#58a6ff}"
    "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);"
    "font:14px/1.5 'JetBrains Mono',monospace;display:flex;flex-direction:column;height:100vh}"
    "header{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;"
    "justify-content:space-between;align-items:center;background:var(--panel)}"
    "#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}"
    ".msg{padding:12px 16px;border-radius:6px;border:1px solid var(--border);max-width:85ch;"
    "white-space:pre-wrap;word-break:break-word}"
    ".usr{background:#1f2937;align-self:flex-end}.ast{background:var(--panel);align-self:flex-start}"
    "form{padding:14px 20px;border-top:1px solid var(--border);display:flex;gap:10px;background:var(--panel)}"
    "input{flex:1;background:var(--bg);color:var(--txt);border:1px solid var(--border);"
    "padding:10px 12px;border-radius:4px;font:inherit}"
    "button{background:var(--acc);color:#000;border:0;padding:10px 18px;border-radius:4px;"
    "font:inherit;font-weight:700;cursor:pointer}</style></head><body>"
    "<header><strong>LFM2 C11 Inference Server</strong><span id=\"stat\" style=\"color:var(--muted)\">Ready</span></header>"
    "<div id=\"log\"></div>"
    "<form id=\"f\"><input id=\"q\" placeholder=\"Ask LFM2 anything...\" autocomplete=\"off\" autofocus>"
    "<button type=\"submit\">Send</button></form><script>"
    "const log=document.getElementById('log'),f=document.getElementById('f'),"
    "q=document.getElementById('q'),stat=document.getElementById('stat');"
    "let msgs=[];"
    "f.onsubmit=async(e)=>{e.preventDefault();const txt=q.value.trim();if(!txt)return;q.value='';"
    "msgs.push({role:'user',content:txt});"
    "let u=document.createElement('div');u.className='msg usr';u.textContent=txt;log.appendChild(u);"
    "let a=document.createElement('div');a.className='msg ast';log.appendChild(a);"
    "let t0=0,tok=0,full='';stat.textContent='Thinking...';"
    "const res=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({messages:msgs,stream:true,temperature:0.3})});"
    "const rdr=res.body.getReader(),dec=new TextDecoder();let buf='';"
    "while(true){const{value,done}=await rdr.read();if(done)break;"
    "buf+=dec.decode(value,{stream:true});const lines=buf.split('\\n');buf=lines.pop();"
    "for(const ln of lines){if(!ln.startsWith('data: '))continue;const d=ln.slice(6).trim();"
    "if(d==='[DONE]'){stat.textContent+=` (${tok} tokens)`;break;}try{const j=JSON.parse(d),c=j.choices[0].delta.content||j.choices[0].delta.reasoning_content||'';"
    "if(c){full+=c;if(!t0){t0=performance.now();}else{tok++;const dt=(performance.now()-t0)/1000;if(dt>0.05)stat.textContent=`${(tok/dt).toFixed(1)} tok/s`;}"
    "a.textContent=full;log.scrollTop=log.scrollHeight;}}catch(_){}}}"
    "msgs.push({role:'assistant',content:full});};</script></body></html>";

/* ============================================================================
 * 2. Zero-Allocation HTTP/1.1 & JSON Request Scanner / Escaper
 * ============================================================================ */

/* Escapes raw UTF-8 bytes into a valid JSON string literal (without outer quotes) */
static size_t json_escape_utf8(const char *src, size_t src_len, char *dst, size_t dst_cap) {
    size_t out = 0;
    for (size_t i = 0; i < src_len && out + 7 < dst_cap; ++i) {
        const uint8_t c = (uint8_t)src[i];
        switch (c) {
            case '"':  dst[out++] = '\\'; dst[out++] = '"';  break;
            case '\\': dst[out++] = '\\'; dst[out++] = '\\'; break;
            case '\b': dst[out++] = '\\'; dst[out++] = 'b';  break;
            case '\f': dst[out++] = '\\'; dst[out++] = 'f';  break;
            case '\n': dst[out++] = '\\'; dst[out++] = 'n';  break;
            case '\r': dst[out++] = '\\'; dst[out++] = 'r';  break;
            case '\t': dst[out++] = '\\'; dst[out++] = 't';  break;
            default:
                if (c < 0x20) {
                    out += (size_t)snprintf(dst + out, dst_cap - out, "\\u%04x", c);
                } else {
                    dst[out++] = (char)c;
                }
                break;
        }
    }
    dst[out] = '\0';
    return out;
}

/* Finds a top-level or nested JSON key `"key_name"` and returns pointer after ':' */
static const char *json_find_key(const char *json, const char *key_name) {
    char pattern[128];
    int plen = snprintf(pattern, sizeof(pattern), "\"%s\"", key_name);
    if (plen <= 0 || (size_t)plen >= sizeof(pattern)) return NULL;

    const char *p = json;
    while ((p = strstr(p, pattern)) != NULL) {
        p += plen;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (*p == ':') {
            p++;
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
            return p;
        }
    }
    return NULL;
}

/* Unescapes a JSON string value starting at `"..."` into `out_buf` and returns
 * pointer immediately past the closing quote. */
static const char *json_parse_string(const char *p, char *out_buf, size_t out_cap) {
    if (!p || *p != '"' || out_cap == 0) return NULL;
    p++; /* Skip opening quote */
    size_t pos = 0;

    while (*p && *p != '"') {
        if (*p == '\\' && p[1] != '\0') {
            p++;
            char esc = *p;
            switch (esc) {
                case 'n':  esc = '\n'; break;
                case 'r':  esc = '\r'; break;
                case 't':  esc = '\t'; break;
                case 'b':  esc = '\b'; break;
                case 'f':  esc = '\f'; break;
                case '"':  esc = '"';  break;
                case '\\': esc = '\\'; break;
                case '/':  esc = '/';  break;
                case 'u': {
                    if (isxdigit((uint8_t)p[1]) && isxdigit((uint8_t)p[2]) &&
                        isxdigit((uint8_t)p[3]) && isxdigit((uint8_t)p[4])) {
                        char hex[5] = { p[1], p[2], p[3], p[4], 0 };
                        unsigned long uval = strtoul(hex, NULL, 16);
                        p += 4;
                        if (uval > 0 && uval < 128) esc = (char)uval;
                        else esc = ' ';
                    }
                    break;
                }
                default: break;
            }
            if (pos + 1 < out_cap) out_buf[pos++] = esc;
        } else {
            if (pos + 1 < out_cap) out_buf[pos++] = *p;
        }
        p++;
    }
    out_buf[pos] = '\0';
    return (*p == '"') ? (p + 1) : NULL;
}

static float json_get_float(const char *json, const char *key, float fallback) {
    const char *v = json_find_key(json, key);
    if (!v || (*v != '-' && (*v < '0' || *v > '9'))) return fallback;
    return strtof(v, NULL);
}

static uint32_t json_get_u32(const char *json, const char *key, uint32_t fallback) {
    const char *v = json_find_key(json, key);
    if (!v || *v < '0' || *v > '9') return fallback;
    return (uint32_t)strtoul(v, NULL, 10);
}

static bool json_get_bool(const char *json, const char *key, bool fallback) {
    const char *v = json_find_key(json, key);
    if (!v) return fallback;
    if (strncmp(v, "true", 4) == 0)  return true;
    if (strncmp(v, "false", 5) == 0) return false;
    return fallback;
}

/* Finds matching closing brace while safely ignoring strings with escapes */
static const char *json_find_matching_brace(const char *p) {
    if (!p || *p != '{') return NULL;
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (; *p; ++p) {
        if (esc) { esc = false; continue; }
        if (*p == '\\' && in_str) { esc = true; continue; }
        if (*p == '"') { in_str = !in_str; continue; }
        if (!in_str) {
            if (*p == '{') depth++;
            else if (*p == '}') {
                depth--;
                if (depth == 0) return p;
            }
        }
    }
    return NULL;
}

/* Parses OpenAI `"messages": [{"role":"system","content":"..."}, ...]`
 * Extracts `system_out` and the latest `user_out` message (or builds full multi-turn). */
static bool json_extract_chat_messages(const char *json,
                                       char *system_out, size_t sys_cap,
                                       char *user_out, size_t user_cap) {
    system_out[0] = '\0';
    user_out[0]   = '\0';

    const char *msgs = json_find_key(json, "messages");
    if (!msgs || *msgs != '[') {
        /* Fallback: allow simple `{"prompt": "..."}` payloads */
        const char *pr = json_find_key(json, "prompt");
        if (pr && *pr == '"') {
            return json_parse_string(pr, user_out, user_cap) != NULL;
        }
        return false;
    }

    const char *cur = msgs + 1;
    while (*cur && *cur != ']') {
        const char *obj = strchr(cur, '{');
        if (!obj) break;
        const char *obj_end = json_find_matching_brace(obj);
        if (!obj_end) break;

        /* Temporary bounded slice for this message object */
        size_t obj_len = (size_t)(obj_end - obj) + 1;
        char slice[4096];
        if (obj_len >= sizeof(slice)) obj_len = sizeof(slice) - 1;
        memcpy(slice, obj, obj_len);
        slice[obj_len] = '\0';

        char role[32] = {0};
        const char *r_ptr = json_find_key(slice, "role");
        if (r_ptr && *r_ptr == '"') {
            json_parse_string(r_ptr, role, sizeof(role));
        }

        const char *c_ptr = json_find_key(slice, "content");
        if (c_ptr && *c_ptr == '"') {
            if (strcmp(role, "system") == 0) {
                json_parse_string(c_ptr, system_out, sys_cap);
            } else if (strcmp(role, "user") == 0) {
                json_parse_string(c_ptr, user_out, user_cap);
            }
        }
        cur = obj_end + 1;
    }
    return (user_out[0] != '\0');
}

/* ============================================================================
 * 3. Pre-Allocated Non-Blocking Connection State Machine
 * ============================================================================ */
typedef enum {
    CONN_FREE          = 0,
    CONN_READING_REQ   = 1, /* Reading HTTP headers & body up to Content-Length */
    CONN_WRITING_RESP  = 2, /* Writing static/buffered HTTP response            */
    CONN_STREAMING_SSE = 3  /* Stepping 1 token per tick via epoll cooperative loop */
} conn_state_t;

typedef struct {
    int          fd;
    conn_state_t state;
    double       accepted_at;

    /* Request buffer & parsed HTTP framing */
    char         req_buf[LFM2_REQ_BUF_CAP];
    size_t       req_len;
    size_t       header_end_pos;
    size_t       content_length;
    char         method[16];
    char         path[128];

    /* Outbound response buffer & write cursor */
    char         resp_buf[LFM2_RESP_BUF_CAP];
    size_t       resp_len;
    size_t       resp_sent;

    /* Per-request SSE / Non-stream generation tracking */
    uint32_t     max_tokens;
    uint32_t     generated_tokens;
    uint64_t     completion_id;
} LFM2HttpConn;

typedef struct {
    int               listen_fd;
    int               epoll_fd;
    uint16_t          port;
    lfm2_model_t     *model;
    lfm2_context_t   *ctx;
    lfm2_model_info_t model_info;
    float            *embd_scratch;
    char              json_scratch[LFM2_RESP_BUF_CAP];
    int               active_sse_slot; /* Index of active streaming conn (-1 if idle) */
    LFM2HttpConn      conns[LFM2_MAX_CONNS];
} LFM2HttpServer;

static int set_fd_nonblocking_nodelay(int fd, bool is_tcp_client) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -1;

    if (is_tcp_client) {
        int one = 1;
        /* Critical for SSE: Disable Nagle's algorithm so every token flushes in <50us */
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    return 0;
}

static void server_close_conn(LFM2HttpServer *srv, int slot) {
    LFM2HttpConn *c = &srv->conns[slot];
    if (c->fd >= 0) {
        epoll_ctl(srv->epoll_fd, EPOLL_CTL_DEL, c->fd, NULL);
        close(c->fd);
    }
    if (srv->active_sse_slot == slot) {
        /* If an SSE client disconnected mid-generation, close the turn cleanly */
        srv->active_sse_slot = -1;
    }
    memset(c, 0, sizeof(*c));
    c->fd    = -1;
    c->state = CONN_FREE;
}

/* Non-blocking socket writer using MSG_NOSIGNAL (never raises SIGPIPE) */
static bool conn_try_flush_resp(LFM2HttpServer *srv, int slot) {
    LFM2HttpConn *c = &srv->conns[slot];
    while (c->resp_sent < c->resp_len) {
        ssize_t n = send(c->fd, c->resp_buf + c->resp_sent,
                         c->resp_len - c->resp_sent, MSG_NOSIGNAL);
        if (n > 0) {
            c->resp_sent += (size_t)n;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return true; /* Socket send buffer full; wait for next EPOLLOUT */
        } else {
            /* Client closed connection (EPIPE / ECONNRESET) */
            server_close_conn(srv, slot);
            return false;
        }
    }
    c->resp_len  = 0;
    c->resp_sent = 0;
    return true;
}

static void conn_queue_http_json(LFM2HttpServer *srv, int slot,
                                 int status_code, const char *status_text,
                                 const char *json_body) {
    LFM2HttpConn *c = &srv->conns[slot];
    size_t body_len = strlen(json_body);
    c->resp_len = (size_t)snprintf(c->resp_buf, sizeof(c->resp_buf),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n%s",
        status_code, status_text, body_len, json_body);
    c->resp_sent = 0;
    c->state     = CONN_WRITING_RESP;
    if (conn_try_flush_resp(srv, slot) && c->resp_len == 0) {
        server_close_conn(srv, slot);
    }
}

/* ============================================================================
 * 4. Route Handlers: /v1/models, /v1/embeddings, /v1/chat/completions
 * ============================================================================ */
static void handle_embeddings_endpoint(LFM2HttpServer *srv, int slot, const char *body) {
    char input_text[8192];
    const char *in_ptr = json_find_key(body, "input");
    if (!in_ptr) {
        conn_queue_http_json(srv, slot, 400, "Bad Request",
                             "{\"error\":{\"message\":\"Missing 'input' field in request.\"}}");
        return;
    }
    /* Support both `"input": "text"` and `"input": ["text"]` */
    if (*in_ptr == '[') {
        in_ptr++;
        while (*in_ptr && isspace((uint8_t)*in_ptr)) in_ptr++;
    }
    if (*in_ptr != '"' || !json_parse_string(in_ptr, input_text, sizeof(input_text))) {
        conn_queue_http_json(srv, slot, 400, "Bad Request",
                             "{\"error\":{\"message\":\"Invalid 'input' string.\"}}");
        return;
    }

    int32_t pool_mode = LFM2_POOL_LAST_TOKEN;
    const char *pm = json_find_key(body, "encoding_pooling");
    if (pm && strncmp(pm, "\"mean\"", 6) == 0) pool_mode = LFM2_POOL_MEAN;

    lfm2_status_t st = lfm2_embed_text(srv->ctx, input_text, pool_mode, srv->embd_scratch);
    if (st != LFM2_OK) {
        conn_queue_http_json(srv, slot, 500, "Internal Server Error",
                             "{\"error\":{\"message\":\"lfm2_embed_text failed.\"}}");
        return;
    }

    /* Zero heap allocation: use srv->json_scratch (pre-allocated 128KB) */
    char *json_buf = srv->json_scratch;
    size_t pos = (size_t)snprintf(json_buf, sizeof(srv->json_scratch),
        "{\"object\":\"list\",\"model\":\"%s\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[",
        srv->model_info.arch);

    const uint32_t D = srv->model_info.n_embd;
    for (uint32_t d = 0; d < D && pos + 24 < sizeof(srv->json_scratch); ++d) {
        pos += (size_t)snprintf(json_buf + pos, sizeof(srv->json_scratch) - pos,
                                (d + 1 < D) ? "%.6f," : "%.6f", srv->embd_scratch[d]);
    }

    lfm2_perf_stats_t stats;
    lfm2_context_get_stats(srv->ctx, &stats);
    pos += (size_t)snprintf(json_buf + pos, sizeof(srv->json_scratch) - pos,
        "]}],\"usage\":{\"prompt_tokens\":%u,\"total_tokens\":%u}}",
        stats.prompt_tokens, stats.prompt_tokens);

    LFM2HttpConn *c = &srv->conns[slot];
    c->resp_len = (size_t)snprintf(c->resp_buf, sizeof(c->resp_buf),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n%s",
        pos, json_buf);

    c->resp_sent = 0;
    c->state     = CONN_WRITING_RESP;
    if (conn_try_flush_resp(srv, slot) && c->resp_len == 0) {
        server_close_conn(srv, slot);
    }
}

static void handle_chat_completions_endpoint(LFM2HttpServer *srv, int slot, const char *body) {
    if (srv->active_sse_slot >= 0 && srv->active_sse_slot != slot) {
        conn_queue_http_json(srv, slot, 429, "Too Many Requests",
                             "{\"error\":{\"message\":\"Engine busy streaming another request.\"}}");
        return;
    }

    char sys_msg[2048];
    char usr_msg[8192];
    if (!json_extract_chat_messages(body, sys_msg, sizeof(sys_msg), usr_msg, sizeof(usr_msg))) {
        conn_queue_http_json(srv, slot, 400, "Bad Request",
                             "{\"error\":{\"message\":\"Missing 'messages' array with user content.\"}}");
        return;
    }

    /* Reset context if requested or if a new system prompt is provided */
    bool reset_ctx = json_get_bool(body, "reset_context", true);
    if (reset_ctx) {
        lfm2_context_reset(srv->ctx);
        if (sys_msg[0] != '\0') {
            lfm2_context_set_system_prompt(srv->ctx, sys_msg, NULL);
        }
    }

    /* Configure Sampling, Grammar & <think> Budget */
    lfm2_sample_params_t sp = lfm2_sample_default_params();
    sp.temperature      = json_get_float(body, "temperature", 0.30f);
    sp.min_p            = json_get_float(body, "min_p", 0.15f);
    sp.top_k            = json_get_u32(body, "top_k", 50);
    sp.rep_penalty      = json_get_float(body, "repetition_penalty", 1.05f);
    sp.max_think_tokens = json_get_u32(body, "max_think_tokens", 0);

    const char *rf = json_find_key(body, "response_format");
    if (rf && strstr(rf, "\"json_object\"") != NULL) {
        sp.grammar_type = LFM2_GRAMMAR_JSON_OBJECT;
    } else if (json_get_bool(body, "tool_choice", false)) {
        sp.grammar_type = LFM2_GRAMMAR_TOOL_CALL;
    }

    lfm2_context_clear_stops(srv->ctx);
    const char *stop_p = json_find_key(body, "stop");
    if (stop_p && *stop_p == '"') {
        char stop_str[64];
        if (json_parse_string(stop_p, stop_str, sizeof(stop_str))) {
            lfm2_context_add_stop(srv->ctx, stop_str);
        }
    }

    if (lfm2_begin_chat_turn(srv->ctx, usr_msg, &sp) != LFM2_OK) {
        conn_queue_http_json(srv, slot, 500, "Internal Server Error",
                             "{\"error\":{\"message\":\"lfm2_begin_chat_turn prefill failed.\"}}");
        return;
    }

    LFM2HttpConn *c      = &srv->conns[slot];
    c->max_tokens        = json_get_u32(body, "max_tokens", g_default_max_tokens);
    c->generated_tokens  = 0;
    c->completion_id     = (uint64_t)time(NULL) ^ (uint64_t)slot;
    const bool is_stream = json_get_bool(body, "stream", false);

    if (is_stream) {
        /* Send HTTP 200 SSE Headers immediately and transition to cooperative stepping */
        c->resp_len = (size_t)snprintf(c->resp_buf, sizeof(c->resp_buf),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "X-Accel-Buffering: no\r\n\r\n");
        c->resp_sent         = 0;
        c->state             = CONN_STREAMING_SSE;
        srv->active_sse_slot = slot;
        conn_try_flush_resp(srv, slot);
    } else {
        /* Non-streaming: collect complete response and return single OpenAI JSON */
        char full_reply[32768];
        size_t full_len = 0;
        full_reply[0] = '\0';

        bool hit_done = false;
        while (c->max_tokens == 0 || c->generated_tokens < c->max_tokens) {
            char utf8_chunk[512];
            size_t chunk_len = 0;
            int32_t think_phase = 0;
            lfm2_status_t st = lfm2_context_next_utf8(
                srv->ctx, utf8_chunk, sizeof(utf8_chunk), &chunk_len, &think_phase);
            c->generated_tokens++;

            if (chunk_len > 0 && full_len + chunk_len + 1 < sizeof(full_reply)) {
                memcpy(full_reply + full_len, utf8_chunk, chunk_len);
                full_len += chunk_len;
                full_reply[full_len] = '\0';
            }
            if (st == LFM2_DONE || st < 0) { hit_done = true; break; }
        }
        const char *reason = (!hit_done && c->max_tokens > 0 && c->generated_tokens >= c->max_tokens)
                             ? "length" : "stop";

        char escaped[65536];
        json_escape_utf8(full_reply, full_len, escaped, sizeof(escaped));

        lfm2_perf_stats_t stats;
        lfm2_context_get_stats(srv->ctx, &stats);

        char json_resp[68000];
        snprintf(json_resp, sizeof(json_resp),
            "{\"id\":\"chatcmpl-%" PRIx64 "\",\"object\":\"chat.completion\","
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":\"%s\"},\"finish_reason\":\"%s\"}],"
            "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u},"
            "\"timings\":{\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"decode_tok_per_sec\":%.2f}}",
            c->completion_id, srv->model_info.arch, escaped, reason,
            stats.prompt_tokens, stats.generated_tokens,
            stats.prompt_tokens + stats.generated_tokens,
            stats.prefill_ms, stats.decode_ms, stats.decode_tok_per_sec);

        conn_queue_http_json(srv, slot, 200, "OK", json_resp);
    }
}

/* Cooperative Single-Token Stepper for Active SSE Stream:
 * Generates ONE token chunk, formats it as an OpenAI SSE `data: {...}\n\n` event,
 * and flushes immediately over the non-blocking TCP_NODELAY socket. */
static void server_step_active_sse(LFM2HttpServer *srv) {
    const int slot = srv->active_sse_slot;
    if (slot < 0) return;

    LFM2HttpConn *c = &srv->conns[slot];
    if (c->state != CONN_STREAMING_SSE) {
        srv->active_sse_slot = -1;
        return;
    }

    /* Wait until previous SSE frame has completely flushed to the kernel socket */
    if (!conn_try_flush_resp(srv, slot)) return;
    if (c->resp_len > 0) return;

    char utf8_chunk[512];
    size_t chunk_len    = 0;
    int32_t think_phase = 0;

    lfm2_status_t st = lfm2_context_next_utf8(
        srv->ctx, utf8_chunk, sizeof(utf8_chunk), &chunk_len, &think_phase);
    c->generated_tokens++;

    if (chunk_len > 0) {
        char escaped[1024];
        json_escape_utf8(utf8_chunk, chunk_len, escaped, sizeof(escaped));

        /* Emit both standard content and reasoning_content flag when inside <think> */
        const char *field = (think_phase == 1) ? "reasoning_content" : "content";
        c->resp_len = (size_t)snprintf(c->resp_buf, sizeof(c->resp_buf),
            "data: {\"id\":\"chatcmpl-%" PRIx64 "\",\"object\":\"chat.completion.chunk\","
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"%s\":\"%s\"},"
            "\"finish_reason\":null}]}\n\n",
            c->completion_id, srv->model_info.arch, field, escaped);
        c->resp_sent = 0;
        if (!conn_try_flush_resp(srv, slot)) return;
    }

    bool hit_max_tokens = (c->max_tokens > 0 && c->generated_tokens >= c->max_tokens);
    if (st == LFM2_DONE || st < 0 || hit_max_tokens) {
        const char *reason = (st != LFM2_DONE && hit_max_tokens) ? "length" : "stop";
        lfm2_perf_stats_t stats;
        lfm2_context_get_stats(srv->ctx, &stats);

        c->resp_len += (size_t)snprintf(c->resp_buf + c->resp_len,
            sizeof(c->resp_buf) - c->resp_len,
            "data: {\"id\":\"chatcmpl-%" PRIx64 "\",\"object\":\"chat.completion.chunk\","
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
            "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}\n\n"
            "data: [DONE]\n\n",
            c->completion_id, srv->model_info.arch, reason,
            stats.prompt_tokens, stats.generated_tokens,
            stats.prompt_tokens + stats.generated_tokens);

        c->state             = CONN_WRITING_RESP;
        srv->active_sse_slot = -1;
        if (conn_try_flush_resp(srv, slot) && c->resp_len == 0) {
            server_close_conn(srv, slot);
        }
    }
}

/* ============================================================================
 * 5. HTTP Request Framing & Dispatcher
 * ============================================================================ */
static void server_dispatch_request(LFM2HttpServer *srv, int slot) {
    LFM2HttpConn *c = &srv->conns[slot];
    const char *body = c->req_buf + c->header_end_pos;

    /* 1. Handle CORS Preflight */
    if (strcmp(c->method, "OPTIONS") == 0) {
        conn_queue_http_json(srv, slot, 204, "No Content", "");
        return;
    }

    /* 2. Serve Embedded Zero-Dependency Web Chat UI at GET / */
    if (strcmp(c->method, "GET") == 0 &&
        (strcmp(c->path, "/") == 0 || strcmp(c->path, "/index.html") == 0)) {
        const size_t html_len = sizeof(LFM2_INDEX_HTML) - 1;
        c->resp_len = (size_t)snprintf(c->resp_buf, sizeof(c->resp_buf),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n\r\n%s",
            html_len, LFM2_INDEX_HTML);
        c->resp_sent = 0;
        c->state     = CONN_WRITING_RESP;
        if (conn_try_flush_resp(srv, slot) && c->resp_len == 0) {
            server_close_conn(srv, slot);
        }
        return;
    }

    /* 3. Serve /health and /v1/models */
    if (strcmp(c->method, "GET") == 0 && strcmp(c->path, "/health") == 0) {
        conn_queue_http_json(srv, slot, 200, "OK", "{\"status\":\"ok\"}");
        return;
    }
    if (strcmp(c->method, "GET") == 0 && strcmp(c->path, "/v1/models") == 0) {
        char models_json[512];
        snprintf(models_json, sizeof(models_json),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"liquid-ai\",\"meta\":{\"isa\":\"%s\",\"n_layers\":%u,"
            "\"shortconv_layers\":%u,\"gqa_layers\":%u,\"n_embd\":%u}}]}",
            srv->model_info.arch, srv->model_info.active_isa,
            srv->model_info.n_layers, srv->model_info.n_shortconv_layers,
            srv->model_info.n_gqa_layers, srv->model_info.n_embd);
        conn_queue_http_json(srv, slot, 200, "OK", models_json);
        return;
    }

    /* 4. Serve POST /v1/chat/completions */
    if (strcmp(c->method, "POST") == 0 && strcmp(c->path, "/v1/chat/completions") == 0) {
        handle_chat_completions_endpoint(srv, slot, body);
        return;
    }

    /* 5. Serve POST /v1/embeddings */
    if (strcmp(c->method, "POST") == 0 && strcmp(c->path, "/v1/embeddings") == 0) {
        handle_embeddings_endpoint(srv, slot, body);
        return;
    }

    conn_queue_http_json(srv, slot, 404, "Not Found",
                         "{\"error\":{\"message\":\"Endpoint not found.\"}}");
}

static void server_handle_client_read(LFM2HttpServer *srv, int slot) {
    LFM2HttpConn *c = &srv->conns[slot];

    for (;;) {
        if (c->req_len + 1 >= sizeof(c->req_buf)) {
            conn_queue_http_json(srv, slot, 413, "Payload Too Large",
                                 "{\"error\":{\"message\":\"Request exceeds 64KB buffer.\"}}");
            return;
        }
        ssize_t n = recv(c->fd, c->req_buf + c->req_len,
                         sizeof(c->req_buf) - 1 - c->req_len, 0);
        if (n > 0) {
            c->req_len += (size_t)n;
            c->req_buf[c->req_len] = '\0';
        } else if (n == 0) {
            server_close_conn(srv, slot);
            return;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            server_close_conn(srv, slot);
            return;
        }
    }

    /* Parse HTTP headers once `\r\n\r\n` arrives */
    if (c->header_end_pos == 0) {
        const char *hdr_end = strstr(c->req_buf, "\r\n\r\n");
        if (!hdr_end) return; /* Wait for more header bytes */

        c->header_end_pos = (size_t)(hdr_end - c->req_buf) + 4;
        if (sscanf(c->req_buf, "%15s %127s", c->method, c->path) != 2) {
            server_close_conn(srv, slot);
            return;
        }

        const char *cl = strcasestr(c->req_buf, "Content-Length:");
        if (cl && cl < hdr_end) {
            c->content_length = (size_t)strtoul(cl + 15, NULL, 10);
        }
    }

    /* Check if full HTTP body has arrived */
    if (c->req_len >= c->header_end_pos + c->content_length) {
        c->req_buf[c->header_end_pos + c->content_length] = '\0';
        server_dispatch_request(srv, slot);
    }
}

/* ============================================================================
 * 6. Main Linux epoll Event Loop
 * ============================================================================ */
LFM2_API int lfm2_run_http_server(lfm2_model_t *model, lfm2_context_t *ctx,
                                  const char *bind_ip, uint16_t port) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = server_sigint_handler;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    LFM2HttpServer *srv = (LFM2HttpServer *)calloc(1, sizeof(LFM2HttpServer));
    if (!srv) return -1;

    srv->model           = model;
    srv->ctx             = ctx;
    srv->port            = port;
    srv->active_sse_slot = -1;
    lfm2_model_get_info(model, &srv->model_info);

    srv->embd_scratch = (float *)calloc(srv->model_info.n_embd, sizeof(float));
    for (int i = 0; i < LFM2_MAX_CONNS; ++i) srv->conns[i].fd = -1;

    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    set_fd_nonblocking_nodelay(srv->listen_fd, false);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, bind_ip ? bind_ip : "127.0.0.1", &addr.sin_addr);

    if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(srv->listen_fd, 64) < 0) {
        perror("bind/listen");
        close(srv->listen_fd);
        free(srv->embd_scratch);
        free(srv);
        return -1;
    }

    srv->epoll_fd = epoll_create1(0);
    struct epoll_event ev = { .events = EPOLLIN, .data.u32 = UINT32_MAX };
    epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, srv->listen_fd, &ev);

    printf("================================================================\n");
    printf(" LFM2 C11 Zero-Dependency epoll HTTP/SSE Server Listening\n");
    printf(" Web UI           : http://%s:%u/\n", bind_ip ? bind_ip : "127.0.0.1", port);
    printf(" Chat Completions : POST http://%s:%u/v1/chat/completions\n", bind_ip ? bind_ip : "127.0.0.1", port);
    printf(" Embeddings       : POST http://%s:%u/v1/embeddings\n", bind_ip ? bind_ip : "127.0.0.1", port);
    printf(" Active Engine    : %s (%s, %u layers, %u embd)\n",
           srv->model_info.arch, srv->model_info.active_isa,
           srv->model_info.n_layers, srv->model_info.n_embd);
    printf("================================================================\n");
    fflush(stdout);

    struct epoll_event events[32];

    while (!g_server_stop) {
        /* When streaming SSE, poll with timeout = 0 so token generation runs
         * at full CPU speed while still checking for EPOLLRDHUP disconnects! */
        const int timeout_ms = (srv->active_sse_slot >= 0) ? 0 : 200;
        const int nfds = epoll_wait(srv->epoll_fd, events, 32, timeout_ms);

        const double now = monotonic_sec();

        for (int i = 0; i < nfds; ++i) {
            if (events[i].data.u32 == UINT32_MAX) {
                /* Accept incoming client connections */
                for (;;) {
                    int cfd = accept(srv->listen_fd, NULL, NULL);
                    if (cfd < 0) break;
                    set_fd_nonblocking_nodelay(cfd, true);

                    int slot = -1;
                    for (int s = 0; s < LFM2_MAX_CONNS; ++s) {
                        if (srv->conns[s].state == CONN_FREE) { slot = s; break; }
                    }
                    if (slot < 0) { close(cfd); continue; }

                    LFM2HttpConn *c = &srv->conns[slot];
                    memset(c, 0, sizeof(*c));
                    c->fd          = cfd;
                    c->state       = CONN_READING_REQ;
                    c->accepted_at = now;

                    struct epoll_event cev = {
                        .events   = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET,
                        .data.u32 = (uint32_t)slot
                    };
                    epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, cfd, &cev);
                }
            } else {
                const int slot = (int)events[i].data.u32;
                const uint32_t mask = events[i].events;
                LFM2HttpConn *c = &srv->conns[slot];
                if (c->state == CONN_FREE) continue;

                /* Immediately abort SSE generation if client disconnected! */
                if (mask & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
                    server_close_conn(srv, slot);
                    continue;
                }
                if ((mask & EPOLLIN) && c->state == CONN_READING_REQ) {
                    server_handle_client_read(srv, slot);
                }
                if ((mask & EPOLLOUT) && c->state != CONN_FREE && c->resp_len > 0) {
                    if (conn_try_flush_resp(srv, slot) &&
                        c->state == CONN_WRITING_RESP && c->resp_len == 0) {
                        server_close_conn(srv, slot);
                    }
                }
            }
        }

        /* Step 1 token on the active SSE stream if one is in flight */
        if (srv->active_sse_slot >= 0) {
            server_step_active_sse(srv);
        }

        /* Anti-Slowloris sweep: drop stalled connections that haven't sent headers */
        for (int s = 0; s < LFM2_MAX_CONNS; ++s) {
            if (srv->conns[s].state == CONN_READING_REQ &&
                now - srv->conns[s].accepted_at > LFM2_SLOWLORIS_SEC) {
                server_close_conn(srv, s);
            }
        }
    }

    printf("\nShutting down LFM2 HTTP server...\n");
    for (int s = 0; s < LFM2_MAX_CONNS; ++s) server_close_conn(srv, s);
    close(srv->epoll_fd);
    close(srv->listen_fd);
    free(srv->embd_scratch);
    free(srv);
    return 0;
}

/* ============================================================================
 * 7. Standalone Mock / CLI Entry Point for Immediate Testing
 * ============================================================================ */
#ifdef LFM2_SERVER_STANDALONE_DEMO
/* Minimal mock implementation of include/lfm2.h so you can compile and test
 * the epoll HTTP/SSE server & embedded browser UI standalone right now. */
struct lfm2_model   { int dummy; };
struct lfm2_context { uint32_t step; uint32_t pos; };

lfm2_sample_params_t lfm2_sample_default_params(void) {
    lfm2_sample_params_t p = { .temperature = 0.3f, .min_p = 0.15f, .top_k = 50, .rep_penalty = 1.05f };
    return p;
}
lfm2_status_t lfm2_model_get_info(const lfm2_model_t *m, lfm2_model_info_t *info) {
    (void)m; memset(info, 0, sizeof(*info));
    snprintf(info->arch, sizeof(info->arch), "lfm2.5-1.2b");
    snprintf(info->active_isa, sizeof(info->active_isa), "AVX2 + FMA + F16C");
    info->n_layers = 16; info->n_shortconv_layers = 10; info->n_gqa_layers = 6; info->n_embd = 16;
    return LFM2_OK;
}
void lfm2_context_reset(lfm2_context_t *c) { c->step = 0; c->pos = 0; }
lfm2_status_t lfm2_context_set_system_prompt(lfm2_context_t *c, const char *s, const char *t) {
    (void)c; (void)s; (void)t; return LFM2_OK;
}
void lfm2_context_clear_stops(lfm2_context_t *c) { (void)c; }
lfm2_status_t lfm2_context_add_stop(lfm2_context_t *c, const char *s) { (void)c; (void)s; return LFM2_OK; }
lfm2_status_t lfm2_begin_chat_turn(lfm2_context_t *c, const char *u, const lfm2_sample_params_t *p) {
    (void)u; (void)p; c->step = 0; c->pos += 12; return LFM2_OK;
}
lfm2_status_t lfm2_context_next_utf8(lfm2_context_t *c, char *buf, size_t cap, size_t *written, int32_t *phase) {
    static const char *demo_toks[] = {
        "Liquid ", "AI ", "LFM2 ", "is ", "streaming ", "via ", "Linux ", "epoll ", "SSE! \xF0\x9F\x94\xA5"
    };
    *phase = 0;
    if (c->step >= 9) { *written = 0; return LFM2_DONE; }
    const char *t = demo_toks[c->step++];
    *written = strlen(t);
    snprintf(buf, cap, "%s", t);
    c->pos++;
    return LFM2_OK;
}
void lfm2_context_get_stats(const lfm2_context_t *c, lfm2_perf_stats_t *st) {
    memset(st, 0, sizeof(*st));
    st->prompt_tokens = 12; st->generated_tokens = c->step; st->current_pos = c->pos;
    st->prefill_ms = 4.2; st->decode_ms = 28.5; st->decode_tok_per_sec = 85.4;
}
lfm2_status_t lfm2_embed_text(lfm2_context_t *c, const char *txt, int32_t pool, float *out) {
    (void)c; (void)txt; (void)pool;
    for (int i = 0; i < 16; ++i) out[i] = 0.25f * sinf((float)i);
    return LFM2_OK;
}
#endif /* LFM2_SERVER_STANDALONE_DEMO */

#ifndef LFM2_SERVER_NO_MAIN
int main(int argc, char **argv) {
#ifdef LFM2_SERVER_STANDALONE_DEMO
    uint16_t port = (argc >= 2) ? (uint16_t)atoi(argv[1]) : 8080;
    const char *bind_ip = (argc >= 3) ? argv[2] : "127.0.0.1";
    struct lfm2_model   mock_m = {0};
    struct lfm2_context mock_c = {0};
    return lfm2_run_http_server(&mock_m, &mock_c, bind_ip, port);
#else
    const char *gguf_path = NULL;
    uint16_t port = 8080;
    const char *bind_ip = "127.0.0.1";
    uint32_t threads = 0;
    uint32_t max_seq_len = LFM_DEFAULT_CTX_LEN;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc) {
            port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_ip = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            max_seq_len = (uint32_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            g_default_max_tokens = lfm_parse_max_tokens_arg(argv[++i]);
        } else if (argv[i][0] != '-') {
            if (!gguf_path) {
                gguf_path = argv[i];
            } else if (port == 8080 && atoi(argv[i]) > 0) {
                port = (uint16_t)atoi(argv[i]);
            }
        }
    }

    if (!gguf_path) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [--port 8080] [--bind 127.0.0.1] [--threads 0] [--ctx %u] [-n|--max-tokens %u]\n", argv[0], (unsigned)LFM_DEFAULT_CTX_LEN, (unsigned)LFM_DEFAULT_MAX_TOKENS);
        return 1;
    }

    printf("[lfm2_server] Loading model: %s...\n", gguf_path);
    lfm2_model_t *model = lfm2_model_load(gguf_path, 1);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", gguf_path);
        return 1;
    }

    lfm2_context_params_t cp = lfm2_context_default_params();
    if (threads > 0) cp.n_threads = threads;
    if (max_seq_len > 0) cp.max_seq_len = max_seq_len;

    lfm2_context_t *ctx = lfm2_context_create(model, &cp);
    if (!ctx) {
        fprintf(stderr, "Failed to create inference context\n");
        lfm2_model_free(model);
        return 1;
    }

    int rc = lfm2_run_http_server(model, ctx, bind_ip, port);

    lfm2_context_free(ctx);
    lfm2_model_free(model);
    return rc;
#endif
}
#endif /* LFM2_SERVER_NO_MAIN */
