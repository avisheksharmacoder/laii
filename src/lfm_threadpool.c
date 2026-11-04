#define _POSIX_C_SOURCE 200809L
#include "lfm_threadpool.h"
#include "lfm_simd.h"
#include "lfm2_simd_avx2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    ThreadPool *pool;
    int thread_idx;
} WorkerContext;

static void* worker_thread_fn(void *arg) {
    lfm2_set_denormals_zero_mode();
    WorkerContext *ctx = (WorkerContext*)arg;
    ThreadPool *pool = ctx->pool;
    int idx = ctx->thread_idx;
    free(ctx);

    uint64_t my_generation = 0;

    while (1) {
        pthread_mutex_lock(&pool->lock);
        while (!pool->stop && pool->generation == my_generation) {
            pthread_cond_wait(&pool->work_cond, &pool->lock);
        }

        if (pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        my_generation = pool->generation;
        ThreadTaskFunc task = pool->current_task;
        void *task_arg = pool->current_arg;
        pthread_mutex_unlock(&pool->lock);

        // Execute task
        task(task_arg, idx);

        pthread_mutex_lock(&pool->lock);
        pool->tasks_remaining--;
        if (pool->tasks_remaining == 0) {
            pthread_cond_signal(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->lock);
    }

    return NULL;
}

static uint32_t lfm_detect_physical_cores(void) {
    long n_logical = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_logical <= 0) return 1;
    if (n_logical > LFM_MAX_THREADS) n_logical = LFM_MAX_THREADS;

    int seen_core_ids[LFM_MAX_THREADS];
    uint32_t n_phys = 0;

    for (int cpu = 0; cpu < (int)n_logical; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE *f = fopen(path, "r");
        int core_id = cpu;
        if (f) {
            if (fscanf(f, "%d", &core_id) != 1) core_id = cpu;
            fclose(f);
        }

        int pkg_id = 0;
        char pkg_path[128];
        snprintf(pkg_path, sizeof(pkg_path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE *fpkg = fopen(pkg_path, "r");
        if (fpkg) {
            if (fscanf(fpkg, "%d", &pkg_id) != 1) pkg_id = 0;
            fclose(fpkg);
        }

        int unique_id = (pkg_id << 16) | (core_id & 0xFFFF);

        bool duplicate = false;
        for (uint32_t i = 0; i < n_phys; ++i) {
            if (seen_core_ids[i] == unique_id) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate && n_phys < LFM_MAX_THREADS) {
            seen_core_ids[n_phys] = unique_id;
            n_phys++;
        }
    }

    if (n_phys == 0) {
        return (n_logical > 1) ? (uint32_t)(n_logical / 2) : 1;
    }
    return n_phys;
}

bool threadpool_init(ThreadPool *pool, uint32_t num_threads) {
    lfm2_set_denormals_zero_mode();
    if (!pool) return false;
    memset(pool, 0, sizeof(ThreadPool));

    /* Check for LFM_NUM_THREADS environment variable override if 0 was passed */
    if (num_threads == 0) {
        const char *env_threads = getenv("LFM_NUM_THREADS");
        if (env_threads) {
            int t = atoi(env_threads);
            if (t > 0) num_threads = (uint32_t)t;
        }
    }

    /* Hardcoded Safety Invariant:
     * Never allow the engine to use all CPU cores under any circumstances.
     * At least 1 physical core must ALWAYS be left free for OS responsiveness,
     * thermal headroom, and preventing hardware shutdowns. */
    uint32_t n_phys = lfm_detect_physical_cores();
    uint32_t max_allowed_threads = (n_phys > 1) ? (n_phys - 1) : 1;

    if (num_threads == 0 || num_threads > max_allowed_threads) {
        num_threads = max_allowed_threads;
    }
    if (num_threads > LFM_MAX_THREADS) num_threads = LFM_MAX_THREADS;
    pool->num_threads = num_threads;

    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->work_cond, NULL);
    pthread_cond_init(&pool->done_cond, NULL);
    pool->stop = false;

    // Spawn num_threads - 1 background workers; thread 0 is the calling thread
    for (uint32_t i = 1; i < num_threads; ++i) {
        WorkerContext *ctx = (WorkerContext*)malloc(sizeof(WorkerContext));
        ctx->pool = pool;
        ctx->thread_idx = (int)i;
        if (pthread_create(&pool->threads[i], NULL, worker_thread_fn, ctx) != 0) {
            free(ctx);
            pool->num_threads = i;
            break;
        }
    }

    return true;
}

void threadpool_dispatch(ThreadPool *pool, ThreadTaskFunc func, void *arg) {
    if (!pool || !func) return;

    if (pool->num_threads <= 1) {
        // Single threaded: execute directly
        func(arg, 0);
        return;
    }

    pthread_mutex_lock(&pool->lock);
    pool->generation++;
    pool->current_task = func;
    pool->current_arg = arg;
    pool->tasks_remaining = (int)pool->num_threads;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->lock);

    // Calling thread performs worker 0 task
    func(arg, 0);

    pthread_mutex_lock(&pool->lock);
    pool->tasks_remaining--;
    while (pool->tasks_remaining > 0) {
        pthread_cond_wait(&pool->done_cond, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
}

void threadpool_free(ThreadPool *pool) {
    if (!pool) return;

    if (pool->num_threads > 1) {
        pthread_mutex_lock(&pool->lock);
        pool->stop = true;
        pthread_cond_broadcast(&pool->work_cond);
        pthread_mutex_unlock(&pool->lock);

        for (uint32_t i = 1; i < pool->num_threads; ++i) {
            pthread_join(pool->threads[i], NULL);
        }
    }

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->work_cond);
    pthread_cond_destroy(&pool->done_cond);
    pool->num_threads = 0;
}

typedef struct {
    float *out;
    const void *weight;
    GGMLType w_type;
    const float *x;
    size_t out_dim;
    size_t in_dim;
    uint32_t num_threads;
} ParallelMatVecArgs;

static void parallel_matvec_worker(void *arg, int thread_idx) {
    ParallelMatVecArgs *args = (ParallelMatVecArgs*)arg;
    size_t out_dim = args->out_dim;
    size_t in_dim = args->in_dim;
    uint32_t n_threads = args->num_threads;

    size_t chunk_size = (out_dim + n_threads - 1) / n_threads;
    size_t r_start = (size_t)thread_idx * chunk_size;
    if (r_start >= out_dim) return;

    size_t r_end = r_start + chunk_size;
    if (r_end > out_dim) r_end = out_dim;
    size_t rows = r_end - r_start;

    const uint8_t *w_base = (const uint8_t*)args->weight;
    if (args->w_type == GGML_TYPE_F32) {
        const float *w_row = (const float*)w_base + r_start * in_dim;
        lfm_matvec_simd(args->out + r_start, w_row, args->w_type, args->x, rows, in_dim);
    } else if (args->w_type == GGML_TYPE_Q8_0) {
        size_t blocks_per_row = in_dim / 32;
        const BlockQ8_0_SIMD *w_blocks = (const BlockQ8_0_SIMD*)w_base + r_start * blocks_per_row;
        lfm_matvec_simd(args->out + r_start, w_blocks, args->w_type, args->x, rows, in_dim);
    } else if (args->w_type == GGML_TYPE_Q4_0) {
        size_t blocks_per_row = in_dim / 32;
        const BlockQ4_0_SIMD *w_blocks = (const BlockQ4_0_SIMD*)w_base + r_start * blocks_per_row;
        lfm_matvec_simd(args->out + r_start, w_blocks, args->w_type, args->x, rows, in_dim);
    } else {
        lfm_matvec_simd(args->out + r_start, w_base, args->w_type, args->x, rows, in_dim);
    }
}

void lfm_matvec_parallel(
    ThreadPool *pool,
    float *out,
    const void *weight,
    GGMLType w_type,
    const float *x,
    size_t out_dim,
    size_t in_dim
) {
    if (!pool || pool->num_threads <= 1 || out_dim < 64) {
        lfm_matvec_simd(out, weight, w_type, x, out_dim, in_dim);
        return;
    }

    ParallelMatVecArgs args = {
        .out = out,
        .weight = weight,
        .w_type = w_type,
        .x = x,
        .out_dim = out_dim,
        .in_dim = in_dim,
        .num_threads = pool->num_threads
    };

    threadpool_dispatch(pool, parallel_matvec_worker, &args);
}
