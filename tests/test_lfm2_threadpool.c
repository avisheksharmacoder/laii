#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#define LFM2_LOADER_NO_MAIN 1
#define LFM2_THREADPOOL_NO_MAIN 1
#define LFM2_FORWARD_NO_MAIN 1

#if __has_include("tools/lfm2_forward.c")
#include "tools/lfm2_forward.c"
#elif __has_include("lfm2_forward.c")
#include "lfm2_forward.c"
#else
#include "../tools/lfm2_forward.c"
#endif

/* Test 1: Physical Core Topology & Affinity */
static void test_topology_detection(void) {
    printf("[Test 1/5] Verifying physical CPU core topology detection...\n");
    int cpu_ids[LFM2_MAX_THREADS];
    uint32_t n_phys = lfm2_detect_physical_cores(cpu_ids);

    assert(n_phys >= 1);
    assert(n_phys <= LFM2_MAX_THREADS);
    printf("  Detected %u unique physical cores: ", n_phys);
    for (uint32_t i = 0; i < n_phys; ++i) {
        printf("core[%u]->cpu%d ", i, cpu_ids[i]);
    }
    printf("\n");

    LFM2ThreadPool pool;
    bool ok = lfm2_threadpool_init(&pool, 0);
    assert(ok);
    uint32_t expected_auto = (n_phys > 1) ? (n_phys - 1) : 1;
    assert(pool.n_threads == expected_auto);
    printf("  ThreadPool initialized with %u workers (N-1 physical core policy, %u physical cores detected).\n",
           pool.n_threads, n_phys);
    lfm2_threadpool_free(&pool);

    /* Explicit thread count must be preserved if within safe N-1 limits */
    if (n_phys >= 3) {
        LFM2ThreadPool pool_exp;
        assert(lfm2_threadpool_init(&pool_exp, 2));
        assert(pool_exp.n_threads == 2);
        lfm2_threadpool_free(&pool_exp);
    }

    /* Hard invariant: Requesting all cores (n_phys) or excessive cores must NEVER use all cores,
     * but always be hard-capped to n_phys - 1 to protect the host machine */
    if (n_phys > 1) {
        LFM2ThreadPool pool_all;
        assert(lfm2_threadpool_init(&pool_all, n_phys));
        assert(pool_all.n_threads == expected_auto);
        lfm2_threadpool_free(&pool_all);

        LFM2ThreadPool pool_huge;
        assert(lfm2_threadpool_init(&pool_huge, 128));
        assert(pool_huge.n_threads == expected_auto);
        lfm2_threadpool_free(&pool_huge);
        printf("  Hard cap verified: requesting %u or 128 cores strictly clamped to %u.\n",
               n_phys, expected_auto);
    }
    printf("PASSED: Topology detection and hard-coded N-1 physical core ceiling verified.\n");
}

/* Test 2: Atomic Chunk Claiming & Dynamic Work-Stealing */
typedef struct {
    LFM2ThreadPool *pool;
    _Atomic uint32_t *visit_counts;
    uint32_t total_items;
} WorkStealTaskCtx;

static void work_steal_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    WorkStealTaskCtx *ctx = (WorkStealTaskCtx *)arg;
    uint32_t start = 0, end = 0;
    while (lfm2_threadpool_claim_chunk(ctx->pool, &start, &end)) {
        for (uint32_t i = start; i < end; ++i) {
            atomic_fetch_add_explicit(&ctx->visit_counts[i], 1, memory_order_relaxed);
        }
    }
}

static void test_work_stealing_barrier(void) {
    printf("[Test 2/5] Verifying spin-barrier dynamic work-stealing correctness...\n");
    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 0);

    const uint32_t N = 16384;
    _Atomic uint32_t *counts = (_Atomic uint32_t *)calloc(N, sizeof(_Atomic uint32_t));
    assert(counts != NULL);

    WorkStealTaskCtx ctx = {
        .pool = &pool,
        .visit_counts = counts,
        .total_items = N
    };

    /* Dispatch with chunk size 32 */
    lfm2_parallel_for(&pool, N, 32, work_steal_task, &ctx);

    /* Verify each item was processed exactly once */
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t c = atomic_load_explicit(&counts[i], memory_order_relaxed);
        assert(c == 1);
    }

    free(counts);
    lfm2_threadpool_free(&pool);
    printf("PASSED: 16,384 work items claimed and processed exactly once across workers.\n");
}

/* Test 3: Multi-Threaded vs Single-Threaded AVX2 MatMul Parity */
static void test_matmul_parity(void) {
    printf("[Test 3/5] Verifying MT vs Single-Thread MatMul numerical equivalence...\n");
    const uint32_t n_cols = 1024;
    const uint32_t n_rows = 512;
    const uint32_t nb     = n_cols / 256;

    size_t x_sz   = (n_cols * sizeof(float) + 63) & ~63ULL;
    size_t out_sz = (n_rows * sizeof(float) + 63) & ~63ULL;
    size_t q8_sz  = (nb * sizeof(block_q8_K_t) + 63) & ~63ULL;
    size_t q4_sz  = ((size_t)n_rows * nb * sizeof(block_q4_K_t) + 63) & ~63ULL;
    size_t q6_sz  = ((size_t)n_rows * nb * sizeof(block_q6_K_t) + 63) & ~63ULL;

    float        *x       = (float *)aligned_alloc(64, x_sz);
    float        *out_seq = (float *)aligned_alloc(64, out_sz);
    float        *out_par = (float *)aligned_alloc(64, out_sz);
    block_q8_K_t *x_q8    = (block_q8_K_t *)aligned_alloc(64, q8_sz);
    block_q4_K_t *w_q4    = (block_q4_K_t *)aligned_alloc(64, q4_sz);
    block_q6_K_t *w_q6    = (block_q6_K_t *)aligned_alloc(64, q6_sz);

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = sinf((float)i * 0.05f);

    /* Setup mock Q4_K */
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3200;
        w_q4[i].dmin = 0x2A00;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i + j * 5) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 13 + j * 29) & 0xFF);
    }

    /* Setup mock Q6_K */
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q6[i].d = 0x3000;
        for (int j = 0; j < 128; ++j) w_q6[i].ql[j] = (uint8_t)((i * 11 + j * 7) & 0xFF);
        for (int j = 0; j < 64; ++j)  w_q6[i].qh[j] = (uint8_t)((i * 3 + j * 19) & 0xFF);
        for (int j = 0; j < 16; ++j)  w_q6[i].scales[j] = (int8_t)((i + j) & 0x1F);
    }

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 0);

    /* 1. Q4_K Sequential vs MT */
    lfm2_matmul_vec_avx2_mt(NULL,  out_seq, w_q4, 12, n_cols, n_rows, x, x_q8);
    lfm2_matmul_vec_avx2_mt(&pool, out_par, w_q4, 12, n_cols, n_rows, x, x_q8);
    for (uint32_t r = 0; r < n_rows; ++r) {
        float diff = fabsf(out_seq[r] - out_par[r]);
        assert(diff == 0.0f);
    }
    printf("  Q4_K MatMul: Bit-exact parity between single-thread and multi-thread.\n");

    /* 2. Q6_K Sequential vs MT */
    lfm2_matmul_vec_avx2_mt(NULL,  out_seq, w_q6, 14, n_cols, n_rows, x, x_q8);
    lfm2_matmul_vec_avx2_mt(&pool, out_par, w_q6, 14, n_cols, n_rows, x, x_q8);
    for (uint32_t r = 0; r < n_rows; ++r) {
        float diff = fabsf(out_seq[r] - out_par[r]);
        assert(diff == 0.0f);
    }
    printf("  Q6_K MatMul: Bit-exact parity between single-thread and multi-thread.\n");

    free(x); free(out_seq); free(out_par); free(x_q8); free(w_q4); free(w_q6);
    lfm2_threadpool_free(&pool);
    printf("PASSED: Numerical parity verified across matrix multiplication formats.\n");
}

/* Test 4: Idle Sleep via Futex and Wake-up Transition */
static void dummy_wake_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid; (void)n_threads;
    _Atomic uint32_t *done = (_Atomic uint32_t *)arg;
    atomic_fetch_add_explicit(done, 1, memory_order_relaxed);
}

static void test_futex_idle_sleep(void) {
    printf("[Test 4/5] Verifying futex idle transition and wake-up recovery...\n");
    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, 0);

    if (pool.n_threads > 1) {
        /* Sleep for 50 ms to guarantee workers exhaust LFM2_SPIN_ITERS and enter kernel futex sleep */
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 };
        nanosleep(&ts, NULL);

        uint32_t sleeping = atomic_load_explicit(&pool.n_sleeping, memory_order_acquire);
        printf("  Workers in kernel futex sleep during idle: %u / %u\n", sleeping, pool.n_threads - 1);
        assert(sleeping == pool.n_threads - 1);

        /* Dispatch new task; futex_wake must restore all workers */
        _Atomic uint32_t done_count = 0;
        lfm2_parallel_for(&pool, pool.n_threads, 1, dummy_wake_task, &done_count);

        uint32_t count = atomic_load_explicit(&done_count, memory_order_relaxed);
        assert(count == pool.n_threads);
        printf("  All %u workers woke from futex and completed task.\n", count);
    }

    lfm2_threadpool_free(&pool);
    printf("PASSED: Futex sleep and wake recovery verified.\n");
}

/* Test 5: End-to-End Forward Pass with ThreadPool */
static void test_forward_with_threadpool(void) {
    printf("[Test 5/5] Verifying hybrid forward pass using threadpool on tests/lfm2_toy.gguf...\n");

    LFM2Model model;
    if (!lfm2_load_gguf("tests/lfm2_toy.gguf", &model)) {
        fprintf(stderr, "FAILED: Could not load tests/lfm2_toy.gguf\n");
        exit(1);
    }

    LFM2State state;
    if (!lfm2_state_init(&state, &model, 512)) {
        fprintf(stderr, "FAILED: lfm2_state_init failed\n");
        exit(1);
    }

    assert(state.pool.n_threads >= 1);
    printf("  LFM2State thread pool active with %u physical threads.\n", state.pool.n_threads);

    /* Run 8 decode steps through ShortConv + GQA attention */
    uint32_t token = 1; /* BOS */
    for (uint32_t step = 0; step < 8; ++step) {
        float *logits = lfm2_forward(&model, &state, token, step);
        assert(logits != NULL);

        /* Argmax */
        uint32_t best_id = 0;
        float best_val = logits[0];
        for (uint32_t i = 1; i < model.cfg.n_vocab; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        token = best_id;
    }
    printf("  8 decode steps executed successfully through thread pool.\n");

    lfm2_state_free(&state);
    lfm2_free_model(&model);
    printf("PASSED: End-to-end forward pass with threadpool verified.\n");
}

int main(void) {
    printf("=======================================================\n");
    printf("Running Liquid AI C11 Spin-Barrier ThreadPool Tests     \n");
    printf("=======================================================\n");

    test_topology_detection();
    test_work_stealing_barrier();
    test_matmul_parity();
    test_futex_idle_sleep();
    test_forward_with_threadpool();

    printf("=======================================================\n");
    printf("ALL 5 THREADPOOL UNIT TESTS PASSED WITH 100%% SUCCESS!   \n");
    printf("=======================================================\n");
    return 0;
}
