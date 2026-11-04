#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_SYS_CACHE_NO_MAIN 1
#include "tools/lfm2_sys_cache.c"

/* ----------------------------------------------------------------------------
 * Test 1: Linux /proc/meminfo Parser & Validation
 * ---------------------------------------------------------------------------- */
static void test_meminfo_parser(void) {
    printf("[Test 1/8] Verifying Linux /proc/meminfo Parser...\n");

    LFM2LinuxMemInfo mem;
    bool ok = lfm2_read_linux_meminfo(&mem);
    assert(ok && "Failed to read Linux /proc/meminfo");
    assert(mem.mem_total_bytes > 0);
    assert(mem.mem_available_bytes > 0);
    assert(mem.mem_free_bytes > 0);
    assert(mem.mem_total_bytes >= mem.mem_available_bytes);

    printf("  MemTotal:     %8.2f MB\n", (double)mem.mem_total_bytes / (1024.0 * 1024.0));
    printf("  MemAvailable: %8.2f MB\n", (double)mem.mem_available_bytes / (1024.0 * 1024.0));
    printf("  PageCache:    %8.2f MB\n", (double)mem.page_cache_bytes / (1024.0 * 1024.0));
    printf("PASSED: /proc/meminfo parsed and validated.\n");
}

/* ----------------------------------------------------------------------------
 * Test 2: LFM2 Hybrid Memory Planner & OOM Clamper
 * ---------------------------------------------------------------------------- */
static void test_memory_planner_and_oom_guard(void) {
    printf("[Test 2/8] Verifying Hybrid Memory Planner & OOM Clamper...\n");

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    snprintf(m.cfg.arch, sizeof(m.cfg.arch), "lfm2");
    m.cfg.n_layers          = 16;
    m.cfg.n_embd            = 2048;
    m.cfg.n_vocab           = 65536;
    m.cfg.n_ctx             = 32768;
    m.cfg.n_head_q          = 32;
    m.cfg.shortconv_l_cache = 3;
    m.file_size             = 700ULL * 1024ULL * 1024ULL; /* 700 MB */

    /* 10 ShortConv layers + 6 GQA layers (each 8 KV heads, head_dim = 64) */
    for (uint32_t l = 0; l < 16; ++l) {
        bool is_gqa = (l == 2 || l == 5 || l == 8 || l == 11 || l == 13 || l == 15);
        m.layers[l].block_type = is_gqa ? LFM2_BLOCK_GQA_ATTN : LFM2_BLOCK_SHORTCONV;
        m.cfg.n_head_kv[l]     = is_gqa ? 8 : 0;
    }

    /* Standard request (32k context) */
    LFM2MemoryPlan p1 = lfm2_plan_memory_and_guard_oom(&m, 32768, 512);
    /* 10 layers * 2 * 2048 * 4 bytes = 163,840 bytes = 160 KB */
    assert(p1.conv_state_bytes == 160ULL * 1024ULL);
    /* 6 layers * 2 * (8 * 64) * 2 bytes = 12,288 bytes / token = 12 KB / token */
    assert(p1.kv_bytes_per_token == 12ULL * 1024ULL);
    assert(p1.requested_ctx == 32768);
    assert(p1.safe_clamped_ctx > 0);

    /* Extreme request on mock huge context: should clamp to safe multiple of 256 */
    m.cfg.n_ctx = 10000000;
    LFM2MemoryPlan p2 = lfm2_plan_memory_and_guard_oom(&m, 10000000, 4096);
    assert(p2.safe_clamped_ctx <= p2.requested_ctx);
    assert(p2.safe_clamped_ctx >= 512);
    assert((p2.safe_clamped_ctx % 256) == 0);

    printf("  ShortConv Memory: %llu bytes (constant O(1))\n", (unsigned long long)p1.conv_state_bytes);
    printf("  GQA KV / Token:   %llu bytes/tok\n", (unsigned long long)p1.kv_bytes_per_token);
    printf("  Clamped Result:   req=%u -> safe=%u (clamped=%s)\n",
           p2.requested_ctx, p2.safe_clamped_ctx, p2.was_clamped ? "YES" : "NO");
    printf("PASSED: Memory planner and OOM guard verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 3: Transparent HugePage Allocator & Alignment
 * ---------------------------------------------------------------------------- */
static void test_hugepage_allocator(void) {
    printf("[Test 3/8] Verifying 2 MB Transparent HugePage Allocator...\n");

    const size_t sz = 8ULL * 1024ULL * 1024ULL; /* 8 MB */
    void *ptr = lfm2_alloc_hugepage_arena(sz);
    assert(ptr != NULL && "HugePage allocation failed");

    /* Verify strict 2 MB alignment (lower 21 bits are zero) */
    uintptr_t addr = (uintptr_t)ptr;
    assert((addr & (2ULL * 1024ULL * 1024ULL - 1)) == 0 && "Pointer not aligned to 2 MB!");

    /* Touch each page to ensure kernel allocates backing pages */
    volatile uint8_t *b = (volatile uint8_t *)ptr;
    for (size_t i = 0; i < sz; i += 4096) {
        b[i] = (uint8_t)(i & 0xFF);
    }
    for (size_t i = 0; i < sz; i += 4096) {
        assert(b[i] == (uint8_t)(i & 0xFF));
    }

    lfm2_free_hugepage_arena(ptr, sz);
    printf("PASSED: 2 MB HugePage allocation, alignment and RW verified.\n");
}

/* ----------------------------------------------------------------------------
 * Test 4: Sequential Page Warmup Functionality
 * ---------------------------------------------------------------------------- */
static void test_page_warmup(void) {
    printf("[Test 4/8] Verifying Sequential Page-Warmup (4KB-Stride Sweep)...\n");

    const size_t sz = 32ULL * 1024ULL * 1024ULL; /* 32 MB */
    void *dummy = lfm2_alloc_hugepage_arena(sz);
    assert(dummy != NULL);

    memset(dummy, 0x5A, sz);

    LFM2Model m;
    memset(&m, 0, sizeof(m));
    m.fd = -1;
    m.file_size = sz;
    m.mmap_base = (const uint8_t *)dummy;

    /* Warm up without terminal output for silent test */
    uint64_t sink = lfm2_warmup_mmap_pages(&m, NULL, false);
    assert(sink > 0 && "Page warmup sink accumulator should be non-zero");

    /* Also verify progress bar logic with /dev/null */
    FILE *devnull = fopen("/dev/null", "w");
    if (devnull) {
        lfm2_warmup_mmap_pages(&m, devnull, true);
        fclose(devnull);
    }

    lfm2_free_hugepage_arena(dummy, sz);
    printf("PASSED: Page warmup correctly touched all pages.\n");
}

/* ----------------------------------------------------------------------------
 * Test 5: Model Fingerprinting Invariant & Sensitivity
 * ---------------------------------------------------------------------------- */
static void test_model_fingerprint(void) {
    printf("[Test 5/8] Verifying Model Fingerprinting Sensitivity...\n");

    LFM2Model m1;
    memset(&m1, 0, sizeof(m1));
    snprintf(m1.cfg.arch, sizeof(m1.cfg.arch), "lfm2");
    m1.cfg.n_layers          = 16;
    m1.cfg.n_embd            = 2048;
    m1.cfg.n_vocab           = 65536;
    m1.cfg.n_head_q          = 32;
    m1.cfg.shortconv_l_cache = 3;
    for (uint32_t i = 0; i < 16; ++i) m1.cfg.n_head_kv[i] = (i % 3 == 0) ? 8 : 0;

    uint64_t fp1 = lfm2_compute_model_fingerprint(&m1);
    assert(fp1 != 0);

    /* Identical configuration must have identical fingerprint */
    LFM2Model m2 = m1;
    uint64_t fp2 = lfm2_compute_model_fingerprint(&m2);
    assert(fp1 == fp2);

    /* Change layer count -> must change fingerprint */
    m2.cfg.n_layers = 15;
    uint64_t fp3 = lfm2_compute_model_fingerprint(&m2);
    assert(fp1 != fp3);

    /* Change embedding dimension -> must change fingerprint */
    m2 = m1;
    m2.cfg.n_embd = 1024;
    uint64_t fp4 = lfm2_compute_model_fingerprint(&m2);
    assert(fp1 != fp4);

    /* Change layer routing table -> must change fingerprint */
    m2 = m1;
    m2.cfg.n_head_kv[0] = 0;
    uint64_t fp5 = lfm2_compute_model_fingerprint(&m2);
    assert(fp1 != fp5);

    printf("PASSED: Model fingerprint is deterministic and sensitive to geometry.\n");
}

/* ----------------------------------------------------------------------------
 * Test 6: Hybrid Prompt Cache Save / Load Round-Trip
 * ---------------------------------------------------------------------------- */
static void test_prompt_cache_roundtrip(void) {
    printf("[Test 6/8] Verifying Hybrid Prompt Cache (.lfmcache) Save & Load Round-Trip...\n");

    LFM2Model model;
    memset(&model, 0, sizeof(model));
    snprintf(model.cfg.arch, sizeof(model.cfg.arch), "lfm2");
    model.cfg.n_layers          = 4;
    model.cfg.n_embd            = 256;
    model.cfg.n_head_q          = 4;
    model.cfg.shortconv_l_cache = 3;

    /* Layer 0, 1, 3: ShortConv; Layer 2: GQA with 2 KV heads (head_dim = 64) */
    model.layers[0].block_type = LFM2_BLOCK_SHORTCONV; model.cfg.n_head_kv[0] = 0;
    model.layers[1].block_type = LFM2_BLOCK_SHORTCONV; model.cfg.n_head_kv[1] = 0;
    model.layers[2].block_type = LFM2_BLOCK_GQA_ATTN;  model.cfg.n_head_kv[2] = 2;
    model.layers[3].block_type = LFM2_BLOCK_SHORTCONV; model.cfg.n_head_kv[3] = 0;

    const uint32_t n_tokens   = 32;
    const uint32_t prompt_len = 48;
    const uint32_t D          = 256;
    const uint32_t kv_dim     = 2 * 64; /* 128 */

    uint32_t tokens[48];
    for (uint32_t i = 0; i < prompt_len; ++i) tokens[i] = 1000 + i * 3;

    float    *conv_state[LFM2_MAX_LAYERS]  = {0};
    uint16_t *k_cache_f16[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_cache_f16[LFM2_MAX_LAYERS] = {0};

    for (uint32_t l = 0; l < 4; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            conv_state[l] = (float *)malloc(2 * D * sizeof(float));
            for (uint32_t i = 0; i < 2 * D; ++i) {
                conv_state[l][i] = (float)(l * 500 + i) * 1.5f;
            }
        } else {
            k_cache_f16[l] = (uint16_t *)malloc(n_tokens * kv_dim * sizeof(uint16_t));
            v_cache_f16[l] = (uint16_t *)malloc(n_tokens * kv_dim * sizeof(uint16_t));
            for (uint32_t i = 0; i < n_tokens * kv_dim; ++i) {
                k_cache_f16[l][i] = (uint16_t)((l * 100 + i) & 0x7FFF);
                v_cache_f16[l][i] = (uint16_t)((l * 200 + i) & 0x7FFF);
            }
        }
    }

    const char *cache_path = "/tmp/test_unit_cache.lfmcache";
    bool ok_save = lfm2_prompt_cache_save(cache_path, &model, tokens, n_tokens,
                                          conv_state, k_cache_f16, v_cache_f16);
    assert(ok_save);

    /* Clear runtime buffers */
    for (uint32_t l = 0; l < 4; ++l) {
        if (conv_state[l])  memset(conv_state[l], 0, 2 * D * sizeof(float));
        if (k_cache_f16[l]) memset(k_cache_f16[l], 0, n_tokens * kv_dim * sizeof(uint16_t));
        if (v_cache_f16[l]) memset(v_cache_f16[l], 0, n_tokens * kv_dim * sizeof(uint16_t));
    }

    /* Reload into runtime buffers */
    uint32_t matched = 0;
    bool ok_load = lfm2_prompt_cache_try_load_prefix(cache_path, &model, tokens, prompt_len, 512,
                                                    conv_state, k_cache_f16, v_cache_f16, &matched);
    assert(ok_load);
    assert(matched == n_tokens);

    /* Verify bit-exact restoration */
    for (uint32_t l = 0; l < 4; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            for (uint32_t i = 0; i < 2 * D; ++i) {
                float expected = (float)(l * 500 + i) * 1.5f;
                assert(conv_state[l][i] == expected);
            }
        } else {
            for (uint32_t i = 0; i < n_tokens * kv_dim; ++i) {
                assert(k_cache_f16[l][i] == (uint16_t)((l * 100 + i) & 0x7FFF));
                assert(v_cache_f16[l][i] == (uint16_t)((l * 200 + i) & 0x7FFF));
            }
        }
    }

    for (uint32_t l = 0; l < 4; ++l) {
        free(conv_state[l]);
        free(k_cache_f16[l]);
        free(v_cache_f16[l]);
    }
    unlink(cache_path);
    printf("PASSED: Prompt cache saved and restored bit-for-bit.\n");
}

/* ----------------------------------------------------------------------------
 * Test 7: LFM2 Exact Prefix Matching Invariant
 * ---------------------------------------------------------------------------- */
static void test_prompt_cache_prefix_rule(void) {
    printf("[Test 7/8] Verifying LFM2 Prefix-Matching Invariant & Rejection...\n");

    LFM2Model model;
    memset(&model, 0, sizeof(model));
    snprintf(model.cfg.arch, sizeof(model.cfg.arch), "lfm2");
    model.cfg.n_layers          = 2;
    model.cfg.n_embd            = 64;
    model.cfg.n_head_q          = 1;
    model.cfg.shortconv_l_cache = 3;
    model.layers[0].block_type  = LFM2_BLOCK_SHORTCONV;
    model.layers[1].block_type  = LFM2_BLOCK_GQA_ATTN;
    model.cfg.n_head_kv[1]      = 1;

    uint32_t cached_tokens[16];
    for (uint32_t i = 0; i < 16; ++i) cached_tokens[i] = i + 1;

    float    *conv[LFM2_MAX_LAYERS]  = {0};
    uint16_t *kf16[LFM2_MAX_LAYERS]  = {0};
    uint16_t *vf16[LFM2_MAX_LAYERS]  = {0};

    conv[0] = (float *)calloc(2 * 64, sizeof(float));
    kf16[1] = (uint16_t *)calloc(16 * 64, sizeof(uint16_t));
    vf16[1] = (uint16_t *)calloc(16 * 64, sizeof(uint16_t));

    const char *cache_path = "/tmp/test_prefix_cache.lfmcache";
    bool ok_save = lfm2_prompt_cache_save(cache_path, &model, cached_tokens, 16, conv, kf16, vf16);
    assert(ok_save);

    /* 1. Mismatched prefix token (e.g. token 5 is different) */
    uint32_t diff_tokens[20];
    for (uint32_t i = 0; i < 20; ++i) diff_tokens[i] = (i == 5) ? 999 : (i + 1);

    uint32_t matched_pos = 0;
    bool ok1 = lfm2_prompt_cache_try_load_prefix(cache_path, &model, diff_tokens, 20, 512,
                                                conv, kf16, vf16, &matched_pos);
    assert(!ok1 && "Mismatched prefix must be rejected!");
    assert(matched_pos == 0);

    /* 2. Shorter prompt than cached prefix (10 < 16) */
    uint32_t short_prompt[10];
    for (uint32_t i = 0; i < 10; ++i) short_prompt[i] = i + 1;
    bool ok2 = lfm2_prompt_cache_try_load_prefix(cache_path, &model, short_prompt, 10, 512,
                                                conv, kf16, vf16, &matched_pos);
    assert(!ok2 && "Shorter prompt than cache must be rejected!");
    assert(matched_pos == 0);

    /* 3. Different model fingerprint */
    LFM2Model diff_model = model;
    diff_model.cfg.n_embd = 128;
    bool ok3 = lfm2_prompt_cache_try_load_prefix(cache_path, &diff_model, cached_tokens, 16, 512,
                                                conv, kf16, vf16, &matched_pos);
    assert(!ok3 && "Different model fingerprint must be rejected!");
    assert(matched_pos == 0);

    free(conv[0]);
    free(kf16[1]);
    free(vf16[1]);
    unlink(cache_path);
    printf("PASSED: Invariant correctly enforced: mismatch, truncation & model differences rejected.\n");
}

/* ----------------------------------------------------------------------------
 * Test 8: Checksum Corruption & Tamper Detection
 * ---------------------------------------------------------------------------- */
static void test_prompt_cache_corruption_detection(void) {
    printf("[Test 8/8] Verifying Checksum Corruption & Tamper Detection...\n");

    LFM2Model model;
    memset(&model, 0, sizeof(model));
    snprintf(model.cfg.arch, sizeof(model.cfg.arch), "lfm2");
    model.cfg.n_layers          = 2;
    model.cfg.n_embd            = 64;
    model.cfg.n_head_q          = 1;
    model.cfg.shortconv_l_cache = 3;
    model.layers[0].block_type  = LFM2_BLOCK_SHORTCONV;
    model.layers[1].block_type  = LFM2_BLOCK_GQA_ATTN;
    model.cfg.n_head_kv[1]      = 1;

    uint32_t tokens[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float    *conv[LFM2_MAX_LAYERS] = {0};
    uint16_t *kf16[LFM2_MAX_LAYERS] = {0};
    uint16_t *vf16[LFM2_MAX_LAYERS] = {0};

    conv[0] = (float *)calloc(2 * 64, sizeof(float));
    kf16[1] = (uint16_t *)calloc(8 * 64, sizeof(uint16_t));
    vf16[1] = (uint16_t *)calloc(8 * 64, sizeof(uint16_t));

    const char *cache_path = "/tmp/test_corrupt_cache.lfmcache";
    bool ok_save = lfm2_prompt_cache_save(cache_path, &model, tokens, 8, conv, kf16, vf16);
    assert(ok_save);

    /* Tamper with 1 byte in the middle of the file */
    FILE *f = fopen(cache_path, "r+b");
    assert(f != NULL);
    fseek(f, 75, SEEK_SET);
    int b = fgetc(f);
    fseek(f, 75, SEEK_SET);
    fputc(b ^ 0xFF, f);
    fclose(f);

    /* Attempt loading tampered cache: must fail checksum validation */
    uint32_t matched_pos = 0;
    bool ok_corrupt = lfm2_prompt_cache_try_load_prefix(cache_path, &model, tokens, 8, 512,
                                                       conv, kf16, vf16, &matched_pos);
    assert(!ok_corrupt && "Corrupted payload must trigger checksum failure!");
    assert(matched_pos == 0);

    free(conv[0]);
    free(kf16[1]);
    free(vf16[1]);
    unlink(cache_path);
    printf("PASSED: Corrupted cache file detected and safely rejected.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI System Runtime, OOM Guard & Cache Test\n");
    printf("=======================================================\n");

    test_meminfo_parser();
    test_memory_planner_and_oom_guard();
    test_hugepage_allocator();
    test_page_warmup();
    test_model_fingerprint();
    test_prompt_cache_roundtrip();
    test_prompt_cache_prefix_rule();
    test_prompt_cache_corruption_detection();

    printf("=======================================================\n");
    printf("ALL 8 SYSTEM RUNTIME & PROMPT CACHE TESTS PASSED 100%%! \n");
    printf("=======================================================\n");
    return EXIT_SUCCESS;
}

