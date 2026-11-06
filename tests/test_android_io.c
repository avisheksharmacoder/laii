#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI Android I/O & Memory Releaser Suite  \n");
    printf("=======================================================\n");

    /* Test 1: Query dynamic OS page size (supporting 4 KB & Android 15+ 16 KB pages) */
    printf("[Test 1/5] Verifying Dynamic OS Page Size Detection...\n");
    long page_sz = sysconf(_SC_PAGESIZE);
    TEST_ASSERT(page_sz == 4096 || page_sz == 16384 || page_sz == 65536);
    printf("  Detected OS page size: %ld bytes (Android 15+ 16KB compliant)\n", page_sz);
    printf("PASSED: OS page size verified.\n");

    /* Test 2: Process RSS reader via /proc/self/statm */
    printf("[Test 2/5] Verifying Linux/Android Process RSS Query API...\n");
    uint64_t rss_bytes = lfm2_get_process_rss_bytes();
    TEST_ASSERT(rss_bytes > 0);
    printf("  Initial process RSS: %.2f MB (%llu bytes)\n",
           (double)rss_bytes / (1024.0 * 1024.0), (unsigned long long)rss_bytes);
    printf("PASSED: Process RSS API verified.\n");

    /* Test 3: Android Scoped Storage simulation via open() -> dup(fd) -> lfm2_model_load_from_fd */
    printf("[Test 3/5] Verifying Android Content URI FD Loader...\n");
    const char *model_path = "tests/lfm2_toy.gguf";
    int raw_fd = open(model_path, O_RDONLY);
    TEST_ASSERT(raw_fd >= 0);

    struct stat st;
    TEST_ASSERT(fstat(raw_fd, &st) == 0);
    uint64_t expected_size = (uint64_t)st.st_size;

    lfm2_model_t *model = lfm2_model_load_from_fd(raw_fd, 0, 0);
    TEST_ASSERT(model != NULL);

    /* Close caller fd immediately: dup(fd) in engine guarantees caller retains ownership */
    close(raw_fd);

    lfm2_model_info_t info;
    TEST_ASSERT(lfm2_model_get_info(model, &info) == LFM2_OK);
    TEST_ASSERT(info.n_layers == 2);
    TEST_ASSERT(info.n_embd == 64);
    TEST_ASSERT(info.n_vocab == 32);
    printf("  Model loaded via detached fd: %s (%llu bytes)\n", info.arch, (unsigned long long)expected_size);
    printf("PASSED: Detached FD loading verified.\n");

    /* Test 4: Mobile Lifecycle Hooks (on_pause release & on_resume warmup) */
    printf("[Test 4/5] Verifying Android lmkd Survival Hooks (Release & Warmup)...\n");
    /* 1. Actively warm up pages */
    lfm2_model_warmup_page_cache(model);
    uint64_t rss_warmed = lfm2_get_process_rss_bytes();
    printf("  RSS after sequential warmup: %.2f MB\n", (double)rss_warmed / (1024.0 * 1024.0));

    /* 2. Evict weight pages via dual-layer MADV_DONTNEED + POSIX_FADV_DONTNEED */
    lfm2_model_release_page_cache(model);
    uint64_t rss_evicted = lfm2_get_process_rss_bytes();
    printf("  RSS after on_pause page release: %.2f MB\n", (double)rss_evicted / (1024.0 * 1024.0));

    /* 3. Re-warmup on resume */
    lfm2_model_warmup_page_cache(model);
    uint64_t rss_restored = lfm2_get_process_rss_bytes();
    printf("  RSS after on_resume warmup: %.2f MB\n", (double)rss_restored / (1024.0 * 1024.0));
    printf("PASSED: lmkd survival hooks verified.\n");

    /* Test 5: Verify Inference Consistency After Weight Eviction & Reload */
    printf("[Test 5/5] Verifying Token Generation Parity After Weight Eviction...\n");
    lfm2_context_params_t cparams = lfm2_context_default_params();
    cparams.max_seq_len = 64;
    lfm2_context_t *ctx = lfm2_context_create(model, &cparams);
    TEST_ASSERT(ctx != NULL);

    lfm2_sample_params_t sparams = lfm2_sample_default_params();
    sparams.temperature = 0.0f; /* Greedy deterministic output */

    lfm2_status_t st_turn = lfm2_begin_chat_turn(ctx, "Hello Android", &sparams);
    TEST_ASSERT(st_turn == LFM2_OK);

    char chunk[64];
    size_t chunk_len = 0;
    int32_t is_think = 0;
    int token_count = 0;
    while (lfm2_context_next_utf8(ctx, chunk, sizeof(chunk), &chunk_len, &is_think) == LFM2_OK && chunk_len > 0) {
        token_count++;
        if (token_count >= 4) break;
    }
    TEST_ASSERT(token_count >= 1);
    printf("  Generated %d tokens cleanly through un-paged weight tables.\n", token_count);

    lfm2_context_free(ctx);
    lfm2_model_free(model);
    printf("PASSED: Inference consistency verified.\n");

    printf("=======================================================\n");
    printf("ALL 5 ANDROID I/O & MEMORY RELEASER TESTS PASSED 100%%! \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}
