#ifndef LFM_THREADPOOL_H
#define LFM_THREADPOOL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include "gguf.h"

#define LFM_MAX_THREADS 64

typedef struct ThreadPool ThreadPool;

typedef void (*ThreadTaskFunc)(void *arg, int thread_idx);

struct ThreadPool {
    uint32_t num_threads;
    pthread_t threads[LFM_MAX_THREADS];

    pthread_mutex_t lock;
    pthread_cond_t  work_cond;
    pthread_cond_t  done_cond;

    ThreadTaskFunc  current_task;
    void           *current_arg;
    int             tasks_remaining;
    uint64_t        generation;
    bool            stop;
};

// Initializes thread pool with specified number of worker threads (0 = auto-detect hardware cores)
bool threadpool_init(ThreadPool *pool, uint32_t num_threads);

// Dispatches a function across all worker threads and waits for completion
void threadpool_dispatch(ThreadPool *pool, ThreadTaskFunc func, void *arg);

// Tears down and joins worker threads
void threadpool_free(ThreadPool *pool);

// Parallel Row-Chunked Matrix-Vector Multiplication using SIMD and ThreadPool
void lfm_matvec_parallel(
    ThreadPool *pool,
    float *out,
    const void *weight,
    GGMLType w_type,
    const float *x,
    size_t out_dim,
    size_t in_dim
);

#endif // LFM_THREADPOOL_H
