#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "lfm2.h"
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ANSI_GREEN "\033[1;32m"
#define ANSI_RED   "\033[1;31m"
#define ANSI_RESET "\033[0m"

static void test_assert_internal(bool condition, const char *expr, const char *file, int line) {
    if (!condition) {
        fprintf(stderr, "%s[FAIL] Assertion failed: %s (%s:%d)%s\n",
                ANSI_RED, expr, file, line, ANSI_RESET);
        exit(EXIT_FAILURE);
    }
}
#define TEST_ASSERT(cond) test_assert_internal((cond), #cond, __FILE__, __LINE__)

static const char *TOY_MODEL_PATH = "tests/lfm2_toy.gguf";
static const uint16_t TEST_PORT   = 19472;

static int connect_to_server(const char *ip, uint16_t port, int max_retries) {
    for (int r = 0; r < max_retries; ++r) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;

        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(port);
        inet_pton(AF_INET, ip, &addr.sin_addr);

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 }; /* 50ms */
        nanosleep(&ts, NULL);
    }
    return -1;
}

static size_t http_roundtrip(const char *ip, uint16_t port, const char *request,
                             char *resp_buf, size_t buf_cap) {
    int fd = connect_to_server(ip, port, 10);
    TEST_ASSERT(fd >= 0);

    size_t req_len = strlen(request);
    size_t sent = 0;
    while (sent < req_len) {
        ssize_t n = send(fd, request + sent, req_len - sent, 0);
        TEST_ASSERT(n > 0);
        sent += (size_t)n;
    }

    size_t total_read = 0;
    for (;;) {
        if (total_read + 1 >= buf_cap) break;
        ssize_t n = recv(fd, resp_buf + total_read, buf_cap - 1 - total_read, 0);
        if (n <= 0) break;
        total_read += (size_t)n;
        resp_buf[total_read] = '\0';

        /* If Content-Length is present, check if we received header + body */
        const char *hdr_end = strstr(resp_buf, "\r\n\r\n");
        if (hdr_end) {
            const char *cl = strcasestr(resp_buf, "Content-Length:");
            if (cl && cl < hdr_end) {
                size_t expected_len = (size_t)strtoul(cl + 15, NULL, 10);
                size_t body_start = (size_t)(hdr_end - resp_buf) + 4;
                if (total_read >= body_start + expected_len) break;
            } else if (strstr(resp_buf, "data: [DONE]\n\n") != NULL) {
                break;
            }
        }
    }
    close(fd);
    resp_buf[total_read] = '\0';
    return total_read;
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI LFM2 epoll HTTP/SSE Server Test Suite\n");
    printf("=======================================================\n");

    /* 1. Fork background server process */
    printf("[Setup] Initializing LFM2 Model & Spawning epoll server on port %u...\n", TEST_PORT);
    pid_t pid = fork();
    TEST_ASSERT(pid >= 0);

    if (pid == 0) {
        /* Child process: load model and run server */
        lfm2_model_t *model = lfm2_model_load(TOY_MODEL_PATH, 0);
        if (!model) {
            fprintf(stderr, "Child: failed to load %s\n", TOY_MODEL_PATH);
            exit(1);
        }
        lfm2_context_params_t cp = lfm2_context_default_params();
        cp.max_seq_len = 1024;
        cp.n_threads = 2;
        lfm2_context_t *ctx = lfm2_context_create(model, &cp);
        if (!ctx) {
            fprintf(stderr, "Child: failed to create context\n");
            exit(1);
        }
        /* Redirect child stdout to /dev/null so test output is clean */
        FILE *devnull = fopen("/dev/null", "w");
        if (devnull) {
            dup2(fileno(devnull), STDOUT_FILENO);
            fclose(devnull);
        }
        int rc = lfm2_run_http_server(model, ctx, "127.0.0.1", TEST_PORT);
        lfm2_context_free(ctx);
        lfm2_model_free(model);
        exit(rc);
    }

    /* Wait for server to become ready */
    int probe_fd = connect_to_server("127.0.0.1", TEST_PORT, 40);
    if (probe_fd < 0) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        fprintf(stderr, "%s[FAIL] Server failed to start on port %u within 2 seconds%s\n",
                ANSI_RED, TEST_PORT, ANSI_RESET);
        exit(EXIT_FAILURE);
    }
    close(probe_fd);
    printf("  Server is listening on 127.0.0.1:%u (PID: %d)\n", TEST_PORT, (int)pid);

    char resp[65536];

    /* ------------------------------------------------------------------------
     * Test 1: GET /health
     * ------------------------------------------------------------------------ */
    printf("[Test 1/8] Verifying GET /health...\n");
    http_roundtrip("127.0.0.1", TEST_PORT,
                   "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                   resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "\"status\":\"ok\"") != NULL);
    printf("  GET /health returned 200 OK: %s\n", strstr(resp, "{\"status\""));

    /* ------------------------------------------------------------------------
     * Test 2: GET / (Embedded Browser UI)
     * ------------------------------------------------------------------------ */
    printf("[Test 2/8] Verifying Embedded HTML Chat Studio at GET /...\n");
    http_roundtrip("127.0.0.1", TEST_PORT,
                   "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                   resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "Content-Type: text/html") != NULL);
    TEST_ASSERT(strstr(resp, "<!DOCTYPE html>") != NULL);
    TEST_ASSERT(strstr(resp, "LFM2 C11 Inference Server") != NULL);
    printf("  GET / served complete embedded studio (%zu bytes).\n", strlen(resp));

    /* ------------------------------------------------------------------------
     * Test 3: GET /v1/models
     * ------------------------------------------------------------------------ */
    printf("[Test 3/8] Verifying GET /v1/models OpenAI endpoint...\n");
    http_roundtrip("127.0.0.1", TEST_PORT,
                   "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                   resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "\"object\":\"list\"") != NULL);
    TEST_ASSERT(strstr(resp, "liquid-ai") != NULL);
    printf("  GET /v1/models reported model list.\n");

    /* ------------------------------------------------------------------------
     * Test 4: OPTIONS /v1/chat/completions (CORS Preflight)
     * ------------------------------------------------------------------------ */
    printf("[Test 4/8] Verifying CORS preflight (OPTIONS)...\n");
    http_roundtrip("127.0.0.1", TEST_PORT,
                   "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                   resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 204 No Content") != NULL);
    printf("  OPTIONS preflight succeeded.\n");

    /* ------------------------------------------------------------------------
     * Test 5: POST /v1/chat/completions (Non-Streaming JSON)
     * ------------------------------------------------------------------------ */
    printf("[Test 5/8] Verifying Non-Streaming POST /v1/chat/completions...\n");
    const char *chat_json =
        "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a fast C engine.\"},"
        "{\"role\":\"user\",\"content\":\"Hello LFM2!\"}],\"stream\":false,\"max_tokens\":6}";
    char post_req[1024];
    snprintf(post_req, sizeof(post_req),
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: 127.0.0.1\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(chat_json), chat_json);
    http_roundtrip("127.0.0.1", TEST_PORT, post_req, resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "\"object\":\"chat.completion\"") != NULL);
    TEST_ASSERT(strstr(resp, "\"choices\"") != NULL);
    TEST_ASSERT(strstr(resp, "\"usage\"") != NULL);
    printf("  Non-streaming completion received successfully.\n");

    /* ------------------------------------------------------------------------
     * Test 6: POST /v1/chat/completions (Streaming SSE with Single-Token Stepping)
     * ------------------------------------------------------------------------ */
    printf("[Test 6/8] Verifying Cooperative Streaming SSE POST /v1/chat/completions...\n");
    const char *stream_json =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Stream tokens please\"}],"
        "\"stream\":true,\"max_tokens\":5}";
    snprintf(post_req, sizeof(post_req),
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: 127.0.0.1\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(stream_json), stream_json);
    http_roundtrip("127.0.0.1", TEST_PORT, post_req, resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "text/event-stream") != NULL);
    TEST_ASSERT(strstr(resp, "data: {\"id\":\"chatcmpl-") != NULL);
    TEST_ASSERT(strstr(resp, "data: [DONE]") != NULL);
    printf("  Streaming SSE events and [DONE] verified.\n");

    /* ------------------------------------------------------------------------
     * Test 7: POST /v1/embeddings (L2-Normalized Embedding Scratch)
     * ------------------------------------------------------------------------ */
    printf("[Test 7/8] Verifying POST /v1/embeddings...\n");
    const char *embd_json = "{\"input\":\"Liquid AI foundation models\",\"encoding_pooling\":\"mean\"}";
    snprintf(post_req, sizeof(post_req),
             "POST /v1/embeddings HTTP/1.1\r\n"
             "Host: 127.0.0.1\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(embd_json), embd_json);
    http_roundtrip("127.0.0.1", TEST_PORT, post_req, resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(resp, "\"object\":\"embedding\"") != NULL);
    TEST_ASSERT(strstr(resp, "\"embedding\":[") != NULL);
    printf("  Embedding vector response verified.\n");

    /* ------------------------------------------------------------------------
     * Test 8: Instant Disconnect Cancellation (EPOLLRDHUP)
     * ------------------------------------------------------------------------ */
    printf("[Test 8/8] Verifying Instant Client Disconnect Cancellation (EPOLLRDHUP)...\n");
    int stream_fd = connect_to_server("127.0.0.1", TEST_PORT, 10);
    TEST_ASSERT(stream_fd >= 0);
    const char *long_stream_json =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Long generation\"}],\"stream\":true,\"max_tokens\":1000}";
    snprintf(post_req, sizeof(post_req),
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: 127.0.0.1\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(long_stream_json), long_stream_json);
    send(stream_fd, post_req, strlen(post_req), 0);

    /* Read first chunk of response */
    char partial[256];
    ssize_t pn = recv(stream_fd, partial, sizeof(partial) - 1, 0);
    TEST_ASSERT(pn > 0);

    /* Abruptly close socket (simulate user clicking 'Stop' or curl aborting) */
    close(stream_fd);

    /* Give epoll loop ~50ms to process EPOLLRDHUP */
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    /* Immediately send a new request - must succeed without 429 Too Many Requests */
    http_roundtrip("127.0.0.1", TEST_PORT,
                   "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                   resp, sizeof(resp));
    TEST_ASSERT(strstr(resp, "HTTP/1.1 200 OK") != NULL);
    printf("  Client disconnect handled cleanly without engine starvation.\n");

    /* ------------------------------------------------------------------------
     * Teardown: Clean SIGTERM Shutdown
     * ------------------------------------------------------------------------ */
    printf("[Teardown] Sending SIGTERM to server process (PID %d)...\n", (int)pid);
    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
    TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    printf("  Server process exited cleanly with code 0.\n");

    printf("=======================================================\n");
    printf("%sALL 8 EPOLL HTTP/SSE SERVER TESTS PASSED (100%% SUCCESS)%s\n",
           ANSI_GREEN, ANSI_RESET);
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}

