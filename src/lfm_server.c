#define _POSIX_C_SOURCE 200809L
#include "lfm_server.h"
#include "lfm_c_api.h"
#include "lfm_webui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

static volatile sig_atomic_t g_server_running = 1;

static void handle_signal(int sig) {
    (void)sig;
    g_server_running = 0;
}

static void escape_json_string(const char *src, char *dst, size_t dst_size) {
    size_t d = 0;
    for (size_t s = 0; src[s] != '\0' && d + 4 < dst_size; ++s) {
        unsigned char c = (unsigned char)src[s];
        if (c == '"') {
            dst[d++] = '\\'; dst[d++] = '"';
        } else if (c == '\\') {
            dst[d++] = '\\'; dst[d++] = '\\';
        } else if (c == '\n') {
            dst[d++] = '\\'; dst[d++] = 'n';
        } else if (c == '\r') {
            dst[d++] = '\\'; dst[d++] = 'r';
        } else if (c == '\t') {
            dst[d++] = '\\'; dst[d++] = 't';
        } else if (c < 32) {
            // ignore or substitute other control characters
            dst[d++] = ' ';
        } else {
            dst[d++] = (char)c;
        }
    }
    dst[d] = '\0';
}

static bool parse_json_string(const char *json, const char *key, char *out, size_t out_size) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return false;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == ':' || *pos == '\t' || *pos == '\r' || *pos == '\n') pos++;
    if (*pos != '"') return false;
    pos++; // skip opening quote

    size_t i = 0;
    while (*pos && *pos != '"' && i + 1 < out_size) {
        if (*pos == '\\' && *(pos + 1)) {
            pos++;
            if (*pos == 'n') out[i++] = '\n';
            else if (*pos == 't') out[i++] = '\t';
            else if (*pos == 'r') out[i++] = '\r';
            else out[i++] = *pos;
        } else {
            out[i++] = *pos;
        }
        pos++;
    }
    out[i] = '\0';
    return true;
}

static float parse_json_float(const char *json, const char *key, float default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return default_val;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == ':' || *pos == '\t') pos++;
    return (float)atof(pos);
}

static int parse_json_int(const char *json, const char *key, int default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return default_val;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == ':' || *pos == '\t') pos++;
    return atoi(pos);
}

static void send_http_response(int fd, int status, const char *content_type, const char *body, size_t body_len) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Server: LiquidAI-LFM2-C-Engine/1.0\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, (status == 200 ? "OK" : status == 204 ? "No Content" : "Not Found"),
        content_type, body_len);

    ssize_t w1 = write(fd, header, hlen);
    (void)w1;
    if (body_len > 0 && body) {
        ssize_t w2 = write(fd, body, body_len);
        (void)w2;
    }
}

int lfm_server_start(lfm_engine_t *engine, const LFMServerConfig *config) {
    if (!engine) return -1;

    int port = config && config->port > 0 ? config->port : 8080;
    const char *host = config && config->host ? config->host : "0.0.0.0";

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGPIPE, SIG_IGN);

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("[Server] socket creation failed");
        return -1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Set receive timeout so accept() checks g_server_running periodically
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, host, &serv_addr.sin_addr);

    if (bind(listen_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        fprintf(stderr, "[Server] bind failed on %s:%d: %s\n", host, port, strerror(errno));
        close(listen_fd);
        return -1;
    }

    if (listen(listen_fd, 64) < 0) {
        perror("[Server] listen failed");
        close(listen_fd);
        return -1;
    }

    printf("=================================================================\n");
    printf("  Liquid AI (LFM2) Single-Binary HTTP Server & Web Studio        \n");
    printf("=================================================================\n");
    printf("  Server Listening on: http://%s:%d\n", host, port);
    printf("  Web Studio URL:      http://localhost:%d\n", port);
    printf("  Chat Endpoint:       POST http://localhost:%d/v1/chat/completions\n", port);
    printf("  Health & Model Info: GET  http://localhost:%d/health\n", port);
    printf("=================================================================\n");
    printf("Press Ctrl+C to stop server.\n\n");

    char *req_buf = (char*)malloc(65536);
    char *gen_buf = (char*)malloc(65536);
    char *escaped_buf = (char*)malloc(65536);
    char *resp_buf = (char*)malloc(131072);

    while (g_server_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue; // normal timeout to check g_server_running
            }
            break;
        }

        // Read HTTP request
        ssize_t bytes_read = read(client_fd, req_buf, 65535);
        if (bytes_read <= 0) {
            close(client_fd);
            continue;
        }
        req_buf[bytes_read] = '\0';

        // Parse HTTP request line
        char method[16] = {0};
        char path[256] = {0};
        sscanf(req_buf, "%15s %255s", method, path);

        if (strcmp(method, "OPTIONS") == 0) {
            send_http_response(client_fd, 204, "text/plain", "", 0);
        } else if (strcmp(method, "GET") == 0 && (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0)) {
            // Serve Embedded Web UI
            size_t html_len = strlen(LFM_EMBEDDED_INDEX_HTML);
            send_http_response(client_fd, 200, "text/html; charset=utf-8", LFM_EMBEDDED_INDEX_HTML, html_len);
        } else if (strcmp(method, "GET") == 0 && (strcmp(path, "/health") == 0 || strcmp(path, "/v1/models") == 0)) {
            // Serve Health / Model Metadata
            uint32_t layers = lfm_get_num_layers(engine);
            uint32_t hdim = lfm_get_hidden_dim(engine);
            uint32_t vocab = lfm_get_vocab_size(engine);
            const char *arch = lfm_get_architecture(engine);

            int rlen = snprintf(resp_buf, 131072,
                "{\"status\":\"ok\",\"model\":\"lfm2\",\"architecture\":\"%s\","
                "\"num_layers\":%u,\"hidden_dim\":%u,\"vocab_size\":%u}\n",
                arch, layers, hdim, vocab);
            send_http_response(client_fd, 200, "application/json", resp_buf, rlen);
        } else if (strcmp(method, "POST") == 0 && (strcmp(path, "/v1/chat/completions") == 0 || strcmp(path, "/v1/completions") == 0)) {
            // Parse POST body
            char prompt[4096] = {0};
            const char *body = strstr(req_buf, "\r\n\r\n");
            if (body) body += 4;
            else body = req_buf;

            // Attempt to extract "content" from chat messages or "prompt"
            if (!parse_json_string(body, "content", prompt, sizeof(prompt))) {
                parse_json_string(body, "prompt", prompt, sizeof(prompt));
            }
            if (strlen(prompt) == 0) {
                strncpy(prompt, "Hello Liquid AI", sizeof(prompt) - 1);
            }

            float temp = parse_json_float(body, "temperature", 0.7f);
            int top_k = parse_json_int(body, "top_k", 40);
            int max_tokens = parse_json_int(body, "max_tokens", (int)LFM_DEFAULT_MAX_TOKENS);
            uint32_t max_new_tokens = (max_tokens <= 0) ? UINT32_MAX : (uint32_t)max_tokens;

            lfm_sampler_params_t params = {
                .temperature = temp,
                .top_k = (uint32_t)top_k,
                .max_new_tokens = max_new_tokens,
                .seed = 42,
                .echo_prompt = false
            };

            // Run generation
            int n_gen = lfm_generate(engine, prompt, gen_buf, 65536, &params);
            if (n_gen < 0) gen_buf[0] = '\0';

            escape_json_string(gen_buf, escaped_buf, 65536);

            int rlen = snprintf(resp_buf, 131072,
                "{\n"
                "  \"id\": \"chatcmpl-lfm2\",\n"
                "  \"object\": \"chat.completion\",\n"
                "  \"created\": %ld,\n"
                "  \"model\": \"lfm2\",\n"
                "  \"choices\": [\n"
                "    {\n"
                "      \"index\": 0,\n"
                "      \"message\": {\n"
                "        \"role\": \"assistant\",\n"
                "        \"content\": \"%s\"\n"
                "      },\n"
                "      \"finish_reason\": \"%s\"\n"
                "    }\n"
                "  ]\n"
                "}\n",
                (long)time(NULL), escaped_buf,
                (params.max_new_tokens != UINT32_MAX && n_gen >= 0 && (uint32_t)n_gen >= params.max_new_tokens) ? "length" : "stop");

            send_http_response(client_fd, 200, "application/json", resp_buf, rlen);
        } else {
            const char not_found[] = "{\"error\":\"Not Found\"}\n";
            send_http_response(client_fd, 404, "application/json", not_found, strlen(not_found));
        }

        close(client_fd);
    }

    printf("\n[Server] Shutting down server gracefully.\n");
    close(listen_fd);
    free(req_buf);
    free(gen_buf);
    free(escaped_buf);
    free(resp_buf);
    return 0;
}
