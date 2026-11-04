#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "lfm_threadpool.h"
#include "lfm_simd.h"

int main(void) {
    printf("[Test] Running Multi-threaded ThreadPool barrier & concurrency tests...\n");

    // 1. Verify auto-detection uses N-1 physical core policy
    ThreadPool auto_pool;
    if (!threadpool_init(&auto_pool, 0)) {
        fprintf(stderr, "FAILED: Failed to initialize auto-detected threadpool\n");
        return 1;
    }
    printf("  Auto-detected thread pool with %u worker threads (N-1 physical core rule).\n", auto_pool.num_threads);
    assert(auto_pool.num_threads >= 1);
    long n_logical = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_logical > 1) {
        // Auto-detected thread count must be strictly less than total logical threads to leave headroom
        assert(auto_pool.num_threads < (uint32_t)n_logical);
    }
    threadpool_free(&auto_pool);

    // 2. Explicit thread configuration test
    ThreadPool pool;
    uint32_t n_threads = 2;
    if (!threadpool_init(&pool, n_threads)) {
        fprintf(stderr, "FAILED: Failed to initialize threadpool with %u threads\n", n_threads);
        return 1;
    }
    assert(pool.num_threads == n_threads);
    printf("  Initialized explicit thread pool with %u worker threads.\n", pool.num_threads);

    // 3. Hard invariant test: Requesting all cores or 128 cores must be strictly capped to N-1 physical cores
    ThreadPool pool_huge;
    assert(threadpool_init(&pool_huge, 128));
    if (n_logical > 1) {
        assert(pool_huge.num_threads < (uint32_t)n_logical);
    }
    threadpool_free(&pool_huge);

    size_t out_dim = 256;
    size_t in_dim = 256;

    float *W = (float*)malloc(out_dim * in_dim * sizeof(float));
    float *x = (float*)malloc(in_dim * sizeof(float));
    float *out_serial = (float*)malloc(out_dim * sizeof(float));
    float *out_parallel = (float*)malloc(out_dim * sizeof(float));

    for (size_t i = 0; i < out_dim * in_dim; ++i) {
        W[i] = (float)((int)(i % 23) - 11) * 0.05f;
    }
    for (size_t i = 0; i < in_dim; ++i) {
        x[i] = (float)((int)(i % 13) - 6) * 0.1f;
    }

    // 1. Compute serial reference
    lfm_matvec_simd(out_serial, W, GGML_TYPE_F32, x, out_dim, in_dim);

    // 2. Concurrency stress test: run 50 parallel matrix-vector dispatches
    for (int iter = 0; iter < 50; ++iter) {
        memset(out_parallel, 0, out_dim * sizeof(float));
        lfm_matvec_parallel(&pool, out_parallel, W, GGML_TYPE_F32, x, out_dim, in_dim);

        float max_diff = 0.0f;
        for (size_t r = 0; r < out_dim; ++r) {
            float d = fabsf(out_parallel[r] - out_serial[r]);
            if (d > max_diff) max_diff = d;
        }
        assert(max_diff < 1e-5f);
    }
    printf("  Parallel MatVec completed 50 iterations with zero divergence (max diff < 1e-5).\n");

    threadpool_free(&pool);
    assert(pool.num_threads == 0);
    printf("  Thread pool cleanly shut down and joined.\n");

    free(W); free(x); free(out_serial); free(out_parallel);

    printf("PASSED: Multi-threaded ThreadPool tests successful!\n");
    return 0;
}
