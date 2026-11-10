#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <errno.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include "lfm2_simd_avx2.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define lfm2_cpu_pause() _mm_pause()
#elif defined(__aarch64__)
#include <arm_neon.h>
#define lfm2_cpu_pause() __asm__ volatile("yield" ::: "memory")
#else
#define lfm2_cpu_pause() do {} while (0)
#endif

#ifndef LFM2_THREADPOOL_INCLUDED
#define LFM2_THREADPOOL_INCLUDED 1

/* Mobile ARM64 uses a shorter user-space spin window before dropping into
 * kernel futex sleep to prevent battery drain and thermal throttling. */
#if defined(__aarch64__) || defined(__arm__)
    #define LFM2_SPIN_ITERS 4000
    #define lfm2_cpu_pause() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(_M_X64)
    #define LFM2_SPIN_ITERS 25000
    #define lfm2_cpu_pause() _mm_pause()
#else
    #define LFM2_SPIN_ITERS 10000
    #define lfm2_cpu_pause() do {} while (0)
#endif

#define LFM2_MAX_THREADS 64

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ============================================================================
 * 1. Linux Futex Wrapper & Cross-Platform CPU Core Topology Detector
 * ============================================================================ */
static inline int futex_wait(_Atomic uint32_t *addr, uint32_t expected) {
    return (int)syscall(SYS_futex, (void *)addr, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
}

static inline int futex_wake(_Atomic uint32_t *addr, int count) {
    return (int)syscall(SYS_futex, (void *)addr, FUTEX_WAKE_PRIVATE, count, NULL, NULL, 0);
}

typedef enum {
    LFM2_CORE_PRIME  = 0, /* Highest-clock Prime core (e.g. Cortex-X4 / X3)    */
    LFM2_CORE_PERF   = 1, /* Performance big core (e.g. Cortex-A720 / A715)    */
    LFM2_CORE_LITTLE = 2  /* Efficiency in-order core (e.g. Cortex-A520 / A510)*/
} lfm2_core_tier_t;

typedef struct {
    int              cpu_id;       /* Logical CPU index in Linux (0..N-1)      */
    int              phys_pkg_id;  /* Socket / cluster package ID              */
    int              core_id;      /* Hardware core ID (for x86 SMT dedup)     */
    uint32_t         max_freq_khz; /* cpufreq/cpuinfo_max_freq in kHz          */
    uint32_t         capacity;     /* Linux EAS cpu_capacity (1..1024)         */
    lfm2_core_tier_t tier;         /* PRIME, PERF, or LITTLE                   */
} LFM2CoreInfo;

typedef struct {
    uint32_t     n_total_physical;                  /* All unique physical cores  */
    uint32_t     n_big_cores;                       /* PRIME + PERF cores count   */
    uint32_t     n_little_cores;                    /* LITTLE cores count         */
    bool         is_heterogeneous;                  /* true if asymmetric clusters*/
    LFM2CoreInfo cores[LFM2_MAX_THREADS];           /* Sorted fastest -> slowest  */
    int          recommended_cpus[LFM2_MAX_THREADS];/* Selected compute core IDs  */
    uint32_t     n_recommended;                     /* Default auto core count    */
    uint32_t     max_safe_threads;                  /* Hard cap preventing freeze */
} LFM2CpuTopology;

static bool read_sysfs_int(const char *path, int *out_val) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    int v = 0;
    bool ok = (fscanf(f, "%d", &v) == 1);
    fclose(f);
    if (ok) *out_val = v;
    return ok;
}

static bool read_sysfs_u32(const char *path, uint32_t *out_val) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    uint32_t v = 0;
    bool ok = (fscanf(f, "%u", &v) == 1);
    fclose(f);
    if (ok) *out_val = v;
    return ok;
}

static int compare_cores_fastest_first(const void *a, const void *b) {
    const LFM2CoreInfo *ca = (const LFM2CoreInfo *)a;
    const LFM2CoreInfo *cb = (const LFM2CoreInfo *)b;
    if (ca->max_freq_khz != cb->max_freq_khz) {
        return (cb->max_freq_khz > ca->max_freq_khz) ? 1 : -1;
    }
    if (ca->capacity != cb->capacity) {
        return (cb->capacity > ca->capacity) ? 1 : -1;
    }
    return (ca->cpu_id - cb->cpu_id);
}

void lfm2_classify_and_select_cores(LFM2CpuTopology *topo) {
    if (topo->n_total_physical == 0) return;

    /* 1. Sort all discovered physical cores from fastest to slowest */
    qsort(topo->cores, topo->n_total_physical, sizeof(LFM2CoreInfo), compare_cores_fastest_first);

    const uint32_t peak_freq = topo->cores[0].max_freq_khz;
    const uint32_t min_freq  = topo->cores[topo->n_total_physical - 1].max_freq_khz;
    const uint32_t peak_cap  = topo->cores[0].capacity;
    const uint32_t min_cap   = topo->cores[topo->n_total_physical - 1].capacity;

    /* Detect if CPU has asymmetric big.LITTLE clusters:
     * True if slowest core is < 78% of peak frequency OR < 65% of peak EAS capacity */
    const bool freq_asymmetric = (peak_freq > 0) && (min_freq * 100U < peak_freq * 78U);
    const bool cap_asymmetric  = (peak_cap  > 0) && (min_cap  * 100U < peak_cap  * 65U);
    topo->is_heterogeneous     = freq_asymmetric || cap_asymmetric;

    topo->n_big_cores    = 0;
    topo->n_little_cores = 0;

    for (uint32_t i = 0; i < topo->n_total_physical; ++i) {
        LFM2CoreInfo *c = &topo->cores[i];
        if (!topo->is_heterogeneous) {
            c->tier = LFM2_CORE_PERF;
            topo->n_big_cores++;
        } else {
            const bool is_little_freq = (peak_freq > 0) && (c->max_freq_khz * 100U < peak_freq * 75U);
            const bool is_little_cap  = (peak_cap  > 0) && (c->capacity     * 100U < peak_cap  * 60U);
            if (is_little_freq || is_little_cap) {
                c->tier = LFM2_CORE_LITTLE;
                topo->n_little_cores++;
            } else {
                const bool is_prime = (c->max_freq_khz * 100U >= peak_freq * 97U) &&
                                      (c->capacity     * 100U >= peak_cap  * 95U) &&
                                      (i == 0);
                c->tier = is_prime ? LFM2_CORE_PRIME : LFM2_CORE_PERF;
                topo->n_big_cores++;
            }
        }
    }

    if (!topo->is_heterogeneous) {
        /* Homogeneous PC / Server (Uniform Cores):
         * Hard-coded safety invariant: Always leave at least 1 core for OS / background daemons.
         * Auto = N_phys - 1; Max safe = N_phys - 1. */
        topo->max_safe_threads = (topo->n_total_physical > 1) ? (topo->n_total_physical - 1) : 1;
        topo->n_recommended   = topo->max_safe_threads;
        for (uint32_t i = 0; i < topo->n_total_physical; ++i) {
            topo->recommended_cpus[i] = topo->cores[i].cpu_id;
        }
    } else {
        /* Heterogeneous Mobile SoC (e.g. 4 Big + 4 Little = 8 cores):
         * 1. Auto-default: Use N_big - 1 cores (e.g., 3 Big cores: 1 Prime + 2 Perf).
         *    Leaves 1 Big core for Kivy/UI AND all 4 Little cores for OS/audio/sensors!
         *    Saturates LPDDR5 bandwidth with 0 straggler delay and cool thermals.
         * 2. Max Safe Ceiling: Leave at least 1 Big core AND 1 Little core free
         *    (e.g., 8 - 2 = 6 cores max), preventing CPU blackout / Vdroop brownouts. */
        topo->n_recommended   = (topo->n_big_cores > 1) ? (topo->n_big_cores - 1) : 1;
        topo->max_safe_threads = (topo->n_total_physical > 2) ? (topo->n_total_physical - 2) : 1;

        /* Order: Big cores first (Prime first, then Perf), then Little cores */
        for (uint32_t i = 0; i < topo->n_total_physical; ++i) {
            topo->recommended_cpus[i] = topo->cores[i].cpu_id;
        }
    }
}

void lfm2_discover_cpu_topology(LFM2CpuTopology *topo) {
    memset(topo, 0, sizeof(*topo));

    long n_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_online <= 0) n_online = 1;
    if (n_online > LFM2_MAX_THREADS) n_online = LFM2_MAX_THREADS;

    for (int cpu = 0; cpu < (int)n_online; ++cpu) {
        char path[160];

        /* Check if core is online */
        if (cpu > 0) {
            int online = 1;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", cpu);
            if (read_sysfs_int(path, &online) && online == 0) continue;
        }

        int pkg_id  = 0;
        int core_id = cpu;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        read_sysfs_int(path, &pkg_id);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        read_sysfs_int(path, &core_id);

#if defined(__x86_64__) || defined(_M_X64)
        /* On x86_64, deduplicate SMT/Hyperthreading siblings sharing (pkg_id, core_id) */
        bool is_smt_sibling = false;
        for (uint32_t i = 0; i < topo->n_total_physical; ++i) {
            if (topo->cores[i].phys_pkg_id == pkg_id && topo->cores[i].core_id == core_id) {
                is_smt_sibling = true;
                break;
            }
        }
        if (is_smt_sibling) continue;
#endif

        uint32_t max_freq = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        if (!read_sysfs_u32(path, &max_freq) || max_freq == 0) {
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu);
            read_sysfs_u32(path, &max_freq);
        }

        uint32_t capacity = 1024;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        read_sysfs_u32(path, &capacity);

        LFM2CoreInfo *ci  = &topo->cores[topo->n_total_physical++];
        ci->cpu_id        = cpu;
        ci->phys_pkg_id   = pkg_id;
        ci->core_id       = core_id;
        ci->max_freq_khz  = max_freq;
        ci->capacity      = capacity;
        ci->tier          = LFM2_CORE_PERF;
    }

    if (topo->n_total_physical == 0) {
        topo->n_total_physical    = 1;
        topo->n_big_cores         = 1;
        topo->n_recommended       = 1;
        topo->max_safe_threads    = 1;
        topo->cores[0].cpu_id     = 0;
        topo->cores[0].capacity   = 1024;
        topo->recommended_cpus[0] = 0;
        return;
    }

    lfm2_classify_and_select_cores(topo);
}

/* Backward compatibility helper for legacy call sites */
static uint32_t lfm2_detect_physical_cores(int out_cpu_ids[LFM2_MAX_THREADS]) {
    LFM2CpuTopology topo;
    lfm2_discover_cpu_topology(&topo);
    for (uint32_t i = 0; i < topo.n_total_physical; ++i) {
        out_cpu_ids[i] = topo.cores[i].cpu_id;
    }
    return topo.n_total_physical;
}

static bool lfm2_pin_current_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0);
}

/* ============================================================================
 * 2. Cache-Line Aligned Spin-Barrier Thread Pool Structures
 * ============================================================================ */

/* Task callback signature:
 *   arg      : pointer to task descriptor struct
 *   tid      : worker index (0 .. n_threads - 1)
 *   n_threads: total active workers */
typedef void (*lfm2_task_fn)(void *arg, uint32_t tid, uint32_t n_threads);

struct LFM2ThreadPool;

typedef struct {
    _Alignas(64) pthread_t thread;
    uint32_t               tid;
    int                    cpu_id;
    struct LFM2ThreadPool *pool;
} LFM2Worker;

typedef struct LFM2ThreadPool {
    uint32_t        n_threads;
    LFM2CpuTopology topology;
    int             cpu_ids[LFM2_MAX_THREADS];

    /* Cache-line 1: Producer dispatch state (written by main thread) */
    _Alignas(64) _Atomic uint32_t generation; /* Incremented per parallel dispatch */
    _Atomic uint32_t              stop;       /* Set to 1 on pool shutdown         */
    lfm2_task_fn                  task_fn;    /* Current kernel function           */
    void                         *task_arg;   /* Current kernel context pointer    */
    uint32_t                      n_items;    /* Total work items (rows or heads)  */
    uint32_t                      chunk_size; /* Dynamic stealing granularity      */

    /* Cache-line 2: Dynamic work-stealing cursor (shared atomic fetch_add) */
    _Alignas(64) _Atomic uint32_t next_item;

    /* Cache-line 3: Completion barrier counter & futex sleep tracker */
    _Alignas(64) _Atomic uint32_t n_pending;
    _Atomic uint32_t              n_sleeping;

    LFM2Worker workers[LFM2_MAX_THREADS];
} LFM2ThreadPool;

/* ============================================================================
 * 3. Worker Thread Loop & Zero-Allocation Parallel Dispatch
 * ============================================================================ */
static void *lfm2_worker_entry(void *ptr) {
    /* Flush denormals to zero (FTZ) & Denormals-are-zero (DAZ) on worker thread (x86 + ARM64 FPCR) */
    lfm2_set_denormals_zero_mode();

    LFM2Worker     *w    = (LFM2Worker *)ptr;
    LFM2ThreadPool *pool = w->pool;

    /* Pin this worker to its dedicated physical CPU core */
    lfm2_pin_current_thread_to_cpu(w->cpu_id);

    uint32_t local_gen = 0;

    for (;;) {
        uint32_t gen = 0;
        uint32_t spin = 0;

        /* Phase 1: Hybrid user-space spin-wait + kernel futex sleep fallback */
        for (;;) {
            gen = atomic_load_explicit(&pool->generation, memory_order_acquire);
            if (gen != local_gen) break;
            if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) return NULL;

            if (spin < LFM2_SPIN_ITERS) {
                lfm2_cpu_pause();
                spin++;
            } else {
                /* Engine is idle (e.g., waiting for user prompt in REPL).
                 * Park thread in OS kernel via futex with 0% CPU usage. */
                atomic_fetch_add_explicit(&pool->n_sleeping, 1, memory_order_relaxed);
                gen = atomic_load_explicit(&pool->generation, memory_order_acquire);
                if (gen == local_gen && !atomic_load_explicit(&pool->stop, memory_order_relaxed)) {
                    futex_wait(&pool->generation, local_gen);
                }
                atomic_fetch_sub_explicit(&pool->n_sleeping, 1, memory_order_relaxed);
                spin = 0;
            }
        }

        if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) return NULL;
        local_gen = gen;

        /* Phase 2: Execute parallel kernel */
        if (pool->task_fn) {
            pool->task_fn(pool->task_arg, w->tid, pool->n_threads);
        }

        /* Phase 3: Decrement barrier counter (release semantics ensures all
         * row writes into `out[]` are visible to main thread before it continues) */
        atomic_fetch_sub_explicit(&pool->n_pending, 1, memory_order_release);
    }
    return NULL;
}

bool lfm2_threadpool_init(LFM2ThreadPool *pool, uint32_t requested_threads) {
    /* Flush denormals to zero (FTZ) & Denormals-are-zero (DAZ) on main thread */
    lfm2_set_denormals_zero_mode();

    memset(pool, 0, sizeof(*pool));

    lfm2_discover_cpu_topology(&pool->topology);

    /* Allow environment override if requested_threads == 0 */
    if (requested_threads == 0) {
        const char *env_threads = getenv("LFM_NUM_THREADS");
        if (env_threads) {
            int t = atoi(env_threads);
            if (t > 0) requested_threads = (uint32_t)t;
        }
    }

    /* Unified Core Selection & Safety Clamping Policy:
     * - If requested_threads == 0: Use auto-recommended thread count:
     *     * PC / Server: N_phys - 1 (leaves 1 physical core for OS/UI)
     *     * Mobile SoC : N_big - 1  (leaves 1 Big core for UI + Little cluster for OS)
     * - If requested_threads > 0: Clamp strictly to max_safe_threads:
     *     * PC / Server: N_phys - 1
     *     * Mobile SoC : N_total - 2 (leaves at least 1 Big core and 1 Little core free)
     * Never allow 100% of all cores to prevent thermal tripping, battery blackouts, or freezing. */
    if (requested_threads == 0) {
        pool->n_threads = pool->topology.n_recommended;
    } else if (requested_threads > pool->topology.max_safe_threads) {
        pool->n_threads = pool->topology.max_safe_threads;
    } else {
        pool->n_threads = requested_threads;
    }

    for (uint32_t i = 0; i < pool->n_threads; ++i) {
        pool->cpu_ids[i] = pool->topology.recommended_cpus[i];
    }

    atomic_init(&pool->generation, 0);
    atomic_init(&pool->stop, 0);
    atomic_init(&pool->next_item, 0);
    atomic_init(&pool->n_pending, 0);
    atomic_init(&pool->n_sleeping, 0);

    /* Pin main thread (Worker 0) to fastest core (Prime core on mobile, or primary physical core on PC) */
    pool->workers[0].tid    = 0;
    pool->workers[0].cpu_id = pool->cpu_ids[0];
    pool->workers[0].pool   = pool;
    lfm2_pin_current_thread_to_cpu(pool->cpu_ids[0]);

    /* Spawn N - 1 background threads for physical cores 1 .. N - 1 */
    for (uint32_t i = 1; i < pool->n_threads; ++i) {
        pool->workers[i].tid    = i;
        pool->workers[i].cpu_id = pool->cpu_ids[i];
        pool->workers[i].pool   = pool;
        if (pthread_create(&pool->workers[i].thread, NULL,
                           lfm2_worker_entry, &pool->workers[i]) != 0) {
            perror("pthread_create");
            pool->n_threads = i;
            break;
        }
    }
    return true;
}

void lfm2_threadpool_free(LFM2ThreadPool *pool) {
    if (pool->n_threads <= 1) {
        memset(pool, 0, sizeof(*pool));
        return;
    }

    atomic_store_explicit(&pool->stop, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    futex_wake(&pool->generation, LFM2_MAX_THREADS);

    for (uint32_t i = 1; i < pool->n_threads; ++i) {
        pthread_join(pool->workers[i].thread, NULL);
    }
    memset(pool, 0, sizeof(*pool));
}

/* Dispatches `fn(arg, tid, n_threads)` across all physical cores and blocks
 * until completion. Main thread executes `tid = 0` directly. */
void lfm2_parallel_for(LFM2ThreadPool *pool, uint32_t n_items, uint32_t chunk_size,
                       lfm2_task_fn fn, void *arg) {
    if (n_items == 0) return;

    /* Single-thread fast-path */
    if (pool == NULL || pool->n_threads <= 1 || n_items == 1) {
        if (pool) {
            pool->n_items    = n_items;
            pool->chunk_size = (chunk_size > 0) ? chunk_size : n_items;
            atomic_store_explicit(&pool->next_item, 0, memory_order_relaxed);
        }
        fn(arg, 0, 1);
        return;
    }

    pool->task_fn    = fn;
    pool->task_arg   = arg;
    pool->n_items    = n_items;
    pool->chunk_size = (chunk_size > 0) ? chunk_size : 16;

    atomic_store_explicit(&pool->next_item, 0, memory_order_relaxed);
    atomic_store_explicit(&pool->n_pending, pool->n_threads - 1, memory_order_relaxed);

    /* Publish task to spinning workers */
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);

    /* Wake any workers that transitioned to futex sleep during REPL idle */
    if (atomic_load_explicit(&pool->n_sleeping, memory_order_acquire) > 0) {
        futex_wake(&pool->generation, (int)(pool->n_threads - 1));
    }

    /* Main thread executes as Worker 0 */
    fn(arg, 0, pool->n_threads);

    /* Spin-wait until all N - 1 background workers complete their chunks */
    while (atomic_load_explicit(&pool->n_pending, memory_order_acquire) > 0) {
        lfm2_cpu_pause();
    }
}

/* Helper called inside `lfm2_task_fn` to claim the next contiguous range [start, end)
 * of rows or attention heads with zero lock contention. */
static inline bool lfm2_threadpool_claim_chunk(LFM2ThreadPool *pool,
                                               uint32_t *start_out,
                                               uint32_t *end_out) {
    if (!pool) return false;
    const uint32_t chunk = pool->chunk_size;
    const uint32_t total = pool->n_items;
    const uint32_t start = atomic_fetch_add_explicit(&pool->next_item, chunk,
                                                     memory_order_relaxed);
    if (start >= total) return false;
    uint32_t end = start + chunk;
    if (end > total) end = total;
    *start_out = start;
    *end_out   = end;
    return true;
}

/* ============================================================================
 * 4. Drop-In Replacement for OpenMP in lfm2_matmul_vec_avx2 & GQA Attention
 * ============================================================================ */

#if defined(__has_include)
#if __has_include("tools/lfm2_simd_avx2.h")
#include "tools/lfm2_simd_avx2.h"
#elif __has_include("lfm2_simd_avx2.h")
#include "lfm2_simd_avx2.h"
#elif __has_include("lfm2_simd_avx2.c")
#include "lfm2_simd_avx2.c"
#endif
#else
#include "tools/lfm2_simd_avx2.h"
#endif

typedef struct {
    LFM2ThreadPool      *pool;
    float               *out;
    const void          *w_data;
    uint32_t             w_type;
    uint32_t             n_cols;
    uint32_t             n_rows;
    const float         *x_f32;
    const block_q8_K_t  *x_q8;
} MatMulTaskCtx;

static void lfm2_matmul_worker_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid;
    (void)n_threads;
    MatMulTaskCtx  *ctx  = (MatMulTaskCtx *)arg;
    LFM2ThreadPool *pool = ctx->pool;

    uint32_t r_start = 0, r_end = 0;

    if (ctx->w_type == 12 /* GGML_TYPE_Q4_K */) {
        const size_t row_blocks = ctx->n_cols / 256;
        const block_q4_K_t *W   = (const block_q4_K_t *)ctx->w_data;
        while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
            uint32_t r = r_start;
#if LFM2_HAS_AVX2
            for (; r + 4 <= r_end; r += 4) {
                lfm2_vec_dot_q4_K_q8_K_4rows_avx2(
                    W + (size_t)(r + 0) * row_blocks,
                    W + (size_t)(r + 1) * row_blocks,
                    W + (size_t)(r + 2) * row_blocks,
                    W + (size_t)(r + 3) * row_blocks,
                    ctx->x_q8, ctx->n_cols,
                    &ctx->out[r + 0], &ctx->out[r + 1], &ctx->out[r + 2], &ctx->out[r + 3]
                );
            }
#endif
            for (; r < r_end; ++r) {
                ctx->out[r] = lfm2_vec_dot_q4_K_q8_K_avx2(
                    W + (size_t)r * row_blocks, ctx->x_q8, ctx->n_cols);
            }
        }
    } else if (ctx->w_type == 14 /* GGML_TYPE_Q6_K */) {
        const size_t row_blocks = ctx->n_cols / 256;
        const block_q6_K_t *W   = (const block_q6_K_t *)ctx->w_data;
        while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
            for (uint32_t r = r_start; r < r_end; ++r) {
                ctx->out[r] = lfm2_vec_dot_q6_K_q8_K_avx2(
                    W + (size_t)r * row_blocks, ctx->x_q8, ctx->n_cols);
            }
        }
    } else if (ctx->w_type == 8 /* GGML_TYPE_Q8_0 */) {
        const size_t row_blocks = ctx->n_cols / 32;
        const block_q8_0_t *W   = (const block_q8_0_t *)ctx->w_data;
        while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
            for (uint32_t r = r_start; r < r_end; ++r) {
                ctx->out[r] = lfm2_vec_dot_q8_0_q8_K_avx2(
                    W + (size_t)r * row_blocks, ctx->x_q8, ctx->n_cols);
            }
        }
    } else if (ctx->w_type == 0 /* GGML_TYPE_F32 */) {
        const float *W = (const float *)ctx->w_data;
        while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
            for (uint32_t r = r_start; r < r_end; ++r) {
                ctx->out[r] = lfm2_vec_dot_f32_avx2(
                    W + (size_t)r * ctx->n_cols, ctx->x_f32, ctx->n_cols);
            }
        }
    }
}

/* Complete OpenMP-free AVX2 Matrix-Vector Multiply:
 * 1. Main thread quantizes `x` to `Q8_K` once per matmul in <1 us.
 * 2. Spin-barrier pool dispatches 16-row chunks dynamically across physical cores. */
void lfm2_matmul_vec_avx2_mt(LFM2ThreadPool *pool,
                             float * restrict out,
                             const void * restrict w_data,
                             uint32_t w_type,
                             uint32_t n_cols,
                             uint32_t n_rows,
                             const float * restrict x,
                             block_q8_K_t * restrict xq_scratch) {
    if (w_type == 12 || w_type == 14 || w_type == 8) {
        if (xq_scratch) {
            lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        }
    }

    if (!pool || pool->n_threads <= 1) {
        lfm2_matmul_vec_avx2(out, w_data, w_type, n_cols, n_rows, x, xq_scratch);
        return;
    }

    MatMulTaskCtx ctx = {
        .pool   = pool,
        .out    = out,
        .w_data = w_data,
        .w_type = w_type,
        .n_cols = n_cols,
        .n_rows = n_rows,
        .x_f32  = x,
        .x_q8   = xq_scratch
    };

    lfm2_parallel_for(pool, n_rows, 16, lfm2_matmul_worker_task, &ctx);
}

/* Fused SwiGLU MLP Kernel (ffn_gate + ffn_up + SiLU in a single threadpool pass) */
typedef struct {
    LFM2ThreadPool     *pool;
    float              *hb_out;
    const void         *gate_data;
    const void         *up_data;
    uint32_t            w_type;
    uint32_t            n_cols;
    uint32_t            n_rows;
    const block_q8_K_t *x_q8;
} SwiGLUTaskCtx;

static void lfm2_swiglu_worker_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid;
    (void)n_threads;
    SwiGLUTaskCtx  *ctx  = (SwiGLUTaskCtx *)arg;
    LFM2ThreadPool *pool = ctx->pool;

    uint32_t r_start = 0, r_end = 0;
    const size_t row_blocks = ctx->n_cols / 256;
    const block_q4_K_t *W_gate = (const block_q4_K_t *)ctx->gate_data;
    const block_q4_K_t *W_up   = (const block_q4_K_t *)ctx->up_data;

    while (lfm2_threadpool_claim_chunk(pool, &r_start, &r_end)) {
        uint32_t r = r_start;
#if LFM2_HAS_AVX2
        for (; r + 4 <= r_end; r += 4) {
            float g[4], u[4];
            lfm2_vec_dot_q4_K_q8_K_4rows_avx2(
                W_gate + (size_t)(r + 0) * row_blocks,
                W_gate + (size_t)(r + 1) * row_blocks,
                W_gate + (size_t)(r + 2) * row_blocks,
                W_gate + (size_t)(r + 3) * row_blocks,
                ctx->x_q8, ctx->n_cols,
                &g[0], &g[1], &g[2], &g[3]
            );
            lfm2_vec_dot_q4_K_q8_K_4rows_avx2(
                W_up + (size_t)(r + 0) * row_blocks,
                W_up + (size_t)(r + 1) * row_blocks,
                W_up + (size_t)(r + 2) * row_blocks,
                W_up + (size_t)(r + 3) * row_blocks,
                ctx->x_q8, ctx->n_cols,
                &u[0], &u[1], &u[2], &u[3]
            );
            for (int k = 0; k < 4; ++k) {
                float silu = g[k] / (1.0f + expf(-g[k]));
                ctx->hb_out[r + k] = silu * u[k];
            }
        }
#endif
        for (; r < r_end; ++r) {
            float g = lfm2_vec_dot_q4_K_q8_K_avx2(W_gate + (size_t)r * row_blocks, ctx->x_q8, ctx->n_cols);
            float u = lfm2_vec_dot_q4_K_q8_K_avx2(W_up   + (size_t)r * row_blocks, ctx->x_q8, ctx->n_cols);
            float silu = g / (1.0f + expf(-g));
            ctx->hb_out[r] = silu * u;
        }
    }
}

void lfm2_matmul_swiglu_avx2_mt(LFM2ThreadPool *pool,
                                float * restrict hb_out,
                                const void * restrict gate_data,
                                const void * restrict up_data,
                                uint32_t w_type,
                                uint32_t n_cols,
                                uint32_t n_rows,
                                const float * restrict x,
                                block_q8_K_t * restrict xq_scratch) {
    if (x && xq_scratch) {
        if (w_type == 12 || w_type == 14 || w_type == 8) {
            lfm2_quantize_row_q8_K_avx2(x, xq_scratch, n_cols);
        }
    }

    if (!pool || pool->n_threads <= 1) {
        const size_t row_blocks = n_cols / 256;
        const block_q4_K_t *W_gate = (const block_q4_K_t *)gate_data;
        const block_q4_K_t *W_up   = (const block_q4_K_t *)up_data;
        for (uint32_t r = 0; r < n_rows; ++r) {
            float g = lfm2_vec_dot_q4_K_q8_K_avx2(W_gate + (size_t)r * row_blocks, xq_scratch, n_cols);
            float u = lfm2_vec_dot_q4_K_q8_K_avx2(W_up   + (size_t)r * row_blocks, xq_scratch, n_cols);
            float silu = g / (1.0f + expf(-g));
            hb_out[r] = silu * u;
        }
        return;
    }

    SwiGLUTaskCtx ctx = {
        .pool      = pool,
        .hb_out    = hb_out,
        .gate_data = gate_data,
        .up_data   = up_data,
        .w_type    = w_type,
        .n_cols    = n_cols,
        .n_rows    = n_rows,
        .x_q8      = xq_scratch
    };

    lfm2_parallel_for(pool, n_rows, 16, lfm2_swiglu_worker_task, &ctx);
}

/* Parallel GQA Multi-Head Attention Task (supports FP16 KV-Cache & Online Softmax) */
typedef struct {
    LFM2ThreadPool  *pool;
    const float     *q;
    const float     *k_cache;
    const float     *v_cache;
    const uint16_t  *k_cache_f16;
    const uint16_t  *v_cache_f16;
    float           *att;
    float           *xb;
    uint32_t         n_head_q;
    uint32_t         kv_mul;
    uint32_t         head_dim;
    uint32_t         kv_dim;
    uint32_t         max_seq_len;
    uint32_t         pos;
    float            scale;
} GQAAttnTaskCtx;

static inline void lfm2_gqa_heads_worker_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)tid;
    (void)n_threads;
    GQAAttnTaskCtx *c = (GQAAttnTaskCtx *)arg;
    uint32_t h_start = 0, h_end = 0;

    /* Claim 1 query head at a time across physical cores */
    while (lfm2_threadpool_claim_chunk(c->pool, &h_start, &h_end)) {
        for (uint32_t h = h_start; h < h_end; ++h) {
            const float *q_head = c->q + (size_t)h * c->head_dim;
            const uint32_t kv_h = h / c->kv_mul;
            float *out_head     = c->xb + (size_t)h * c->head_dim;

#if LFM2_HAS_AVX2
            if (c->k_cache_f16 && c->v_cache_f16) {
                /* Hardware F16C + Single-Pass Online Softmax (FlashAttention-CPU) */
                float m = -1e30f;
                float l = 0.0f;
                __m256 v_acc[8];
                const uint32_t n_vec = (c->head_dim <= 64) ? (c->head_dim / 8) : 8;
                for (uint32_t v = 0; v < n_vec; ++v) {
                    v_acc[v] = _mm256_setzero_ps();
                }

                for (uint32_t t = 0; t <= c->pos; ++t) {
                    const uint16_t *kt_ptr = c->k_cache_f16 + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;
                    const uint16_t *vt_ptr = c->v_cache_f16 + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;

                    __m256 dot_acc = _mm256_setzero_ps();
                    for (uint32_t i = 0; i < c->head_dim; i += 8) {
                        __m256 vq = _mm256_loadu_ps(q_head + i);
                        __m256 vk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kt_ptr + i)));
                        dot_acc = _mm256_fmadd_ps(vq, vk, dot_acc);
                    }
                    float s_t = lfm2_hsum_ps_avx2(dot_acc) * c->scale;

                    if (s_t > m) {
                        float diff = m - s_t;
                        float scale_prev = expf(diff);
                        m = s_t;
                        l = l * scale_prev + 1.0f;
                        __m256 v_scale = _mm256_set1_ps(scale_prev);
                        for (uint32_t v = 0; v < n_vec; ++v) {
                            __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vt_ptr + v * 8)));
                            v_acc[v] = _mm256_fmadd_ps(v_acc[v], v_scale, vt);
                        }
                    } else {
                        float diff = s_t - m;
                        float weight_t = expf(diff);
                        l += weight_t;
                        __m256 v_wt = _mm256_set1_ps(weight_t);
                        for (uint32_t v = 0; v < n_vec; ++v) {
                            __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vt_ptr + v * 8)));
                            v_acc[v] = _mm256_fmadd_ps(vt, v_wt, v_acc[v]);
                        }
                    }
                }

                float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                __m256 vinv = _mm256_set1_ps(inv_l);
                for (uint32_t v = 0; v < n_vec; ++v) {
                    _mm256_storeu_ps(out_head + v * 8, _mm256_mul_ps(v_acc[v], vinv));
                }
                continue;
            }
#elif defined(__aarch64__)
            if (c->k_cache_f16 && c->v_cache_f16 && c->head_dim <= 64 && (c->head_dim % 8) == 0) {
                /* ARM64 NEON Hardware FP16 + Single-Pass Online Softmax */
                float m = -1e30f;
                float l = 0.0f;
                float32x4_t v_acc[16];
                const uint32_t n_vec = c->head_dim / 4;
                for (uint32_t v = 0; v < n_vec; ++v) {
                    v_acc[v] = vdupq_n_f32(0.0f);
                }

                for (uint32_t t = 0; t <= c->pos; ++t) {
                    const uint16_t *kt_ptr = c->k_cache_f16 + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;
                    const uint16_t *vt_ptr = c->v_cache_f16 + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;

                    float32x4_t dot0 = vdupq_n_f32(0.0f);
                    float32x4_t dot1 = vdupq_n_f32(0.0f);
                    for (uint32_t i = 0; i < c->head_dim; i += 8) {
                        uint16x8_t hk = vld1q_u16(kt_ptr + i);
                        float32x4_t vk0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hk)));
                        float32x4_t vk1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hk)));
                        dot0 = vfmaq_f32(dot0, vld1q_f32(q_head + i + 0), vk0);
                        dot1 = vfmaq_f32(dot1, vld1q_f32(q_head + i + 4), vk1);
                    }
                    float s_t = vaddvq_f32(vaddq_f32(dot0, dot1)) * c->scale;

                    if (s_t > m) {
                        float scale_prev = expf(m - s_t);
                        m = s_t;
                        l = l * scale_prev + 1.0f;
                        for (uint32_t v = 0; v < n_vec; v += 2) {
                            uint16x8_t hv = vld1q_u16(vt_ptr + v * 4);
                            float32x4_t vt0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hv)));
                            float32x4_t vt1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hv)));
                            v_acc[v + 0] = vfmaq_n_f32(vt0, v_acc[v + 0], scale_prev);
                            v_acc[v + 1] = vfmaq_n_f32(vt1, v_acc[v + 1], scale_prev);
                        }
                    } else {
                        float weight_t = expf(s_t - m);
                        l += weight_t;
                        for (uint32_t v = 0; v < n_vec; v += 2) {
                            uint16x8_t hv = vld1q_u16(vt_ptr + v * 4);
                            float32x4_t vt0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hv)));
                            float32x4_t vt1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hv)));
                            v_acc[v + 0] = vfmaq_n_f32(v_acc[v + 0], vt0, weight_t);
                            v_acc[v + 1] = vfmaq_n_f32(v_acc[v + 1], vt1, weight_t);
                        }
                    }
                }

                float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                for (uint32_t v = 0; v < n_vec; ++v) {
                    vst1q_f32(out_head + v * 4, vmulq_n_f32(v_acc[v], inv_l));
                }
                continue;
            }
#endif

            /* Fallback FP32 2-Pass Softmax */
            float *scores = c->att + (size_t)h * c->max_seq_len;
            float max_score = -1e30f;
            for (uint32_t t = 0; t <= c->pos; ++t) {
                const float *k_t = c->k_cache + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;
                float dot = 0.0f;
#if LFM2_HAS_AVX2
                dot = lfm2_vec_dot_f32_avx2(q_head, k_t, c->head_dim);
#elif defined(__aarch64__)
                float32x4_t vdot = vdupq_n_f32(0.0f);
                uint32_t i = 0;
                for (; i + 4 <= c->head_dim; i += 4) {
                    vdot = vfmaq_f32(vdot, vld1q_f32(q_head + i), vld1q_f32(k_t + i));
                }
                dot = vaddvq_f32(vdot);
                for (; i < c->head_dim; ++i) dot += q_head[i] * k_t[i];
#else
                for (uint32_t i = 0; i < c->head_dim; ++i) dot += q_head[i] * k_t[i];
#endif
                scores[t] = dot * c->scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            float sum_exp = 0.0f;
            for (uint32_t t = 0; t <= c->pos; ++t) {
                scores[t] = expf(scores[t] - max_score);
                sum_exp  += scores[t];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (uint32_t t = 0; t <= c->pos; ++t) {
                scores[t] *= inv_sum;
            }

            for (uint32_t i = 0; i < c->head_dim; ++i) out_head[i] = 0.0f;

            for (uint32_t t = 0; t <= c->pos; ++t) {
                const float  a_t = scores[t];
                const float *v_t = c->v_cache + (size_t)t * c->kv_dim + (size_t)kv_h * c->head_dim;
                for (uint32_t i = 0; i < c->head_dim; ++i) {
                    out_head[i] += a_t * v_t[i];
                }
            }
        }
    }
}

#endif /* LFM2_THREADPOOL_INCLUDED */

/* ============================================================================
 * 5. Standalone Static-Build Verification & Barrier Latency Benchmark
 * ============================================================================ */
#ifndef LFM2_THREADPOOL_NO_MAIN
static void noop_barrier_task(void *arg, uint32_t tid, uint32_t n_threads) {
    (void)arg; (void)tid; (void)n_threads;
}

int main(int argc, char **argv) {
    uint32_t req_threads = (argc >= 2) ? (uint32_t)atoi(argv[1]) : 0;

    LFM2ThreadPool pool;
    lfm2_threadpool_init(&pool, req_threads);

    printf("=== LFM2 C11 Lock-Free Spin-Barrier Thread Pool (No OpenMP / libgomp) ===\n");
    printf("Active Physical Cores : %u\n", pool.n_threads);
    printf("Core Affinity Pinning : ");
    for (uint32_t i = 0; i < pool.n_threads; ++i) {
        printf("tid[%u]->cpu%d ", i, pool.cpu_ids[i]);
    }
    printf("\n\n");

    /* 1. Benchmark Raw Spin-Barrier Dispatch + Wake + Join Overhead */
    const int barrier_iters = 100000;
    double t0 = now_ms();
    for (int i = 0; i < barrier_iters; ++i) {
        lfm2_parallel_for(&pool, pool.n_threads, 1, noop_barrier_task, NULL);
    }
    double barrier_ns = ((now_ms() - t0) * 1e6) / barrier_iters;
    printf("Spin-Barrier Round-Trip Latency : %.1f ns / dispatch (%.2f us for 80 matmuls/token)\n",
           barrier_ns, (barrier_ns * 80.0) / 1000.0);

    /* 2. Benchmark AVX2 Q4_K x Q8_K MatMul [6144 x 2048] on Spin-Barrier Pool */
    const uint32_t n_cols = 2048;
    const uint32_t n_rows = 6144;
    const uint32_t nb     = n_cols / 256;

    size_t x_sz   = (n_cols * sizeof(float) + 63) & ~63ULL;
    size_t out_sz = (n_rows * sizeof(float) + 63) & ~63ULL;
    size_t q8_sz  = (nb * sizeof(block_q8_K_t) + 63) & ~63ULL;
    size_t q4_sz  = ((size_t)n_rows * nb * sizeof(block_q4_K_t) + 63) & ~63ULL;

    float        *x    = (float *)aligned_alloc(64, x_sz);
    float        *out  = (float *)aligned_alloc(64, out_sz);
    block_q8_K_t *x_q8 = (block_q8_K_t *)aligned_alloc(64, q8_sz);
    block_q4_K_t *w_q4 = (block_q4_K_t *)aligned_alloc(64, q4_sz);

    for (uint32_t i = 0; i < n_cols; ++i) x[i] = cosf((float)i * 0.013f);
    for (size_t i = 0; i < (size_t)n_rows * nb; ++i) {
        w_q4[i].d    = 0x3000;
        w_q4[i].dmin = 0x2800;
        for (int j = 0; j < 12; ++j)  w_q4[i].scales[j] = (uint8_t)((i + j * 3) & 0x3F);
        for (int j = 0; j < 128; ++j) w_q4[i].qs[j]     = (uint8_t)((i * 17 + j * 31) & 0xFF);
    }

    const int matmul_iters = 200;
    t0 = now_ms();
    for (int i = 0; i < matmul_iters; ++i) {
        lfm2_matmul_vec_avx2_mt(&pool, out, w_q4, 12, n_cols, n_rows, x, x_q8);
    }
    double dt_ms = (now_ms() - t0) / matmul_iters;
    double bw_gbs = ((double)n_rows * nb * sizeof(block_q4_K_t)) / (dt_ms * 1e6);

    printf("Q4_K x Q8_K MatMul [6144 x 2048]: %.3f ms/call | Effective Bandwidth: %.2f GB/s\n",
           dt_ms, bw_gbs);

    free(x); free(out); free(x_q8); free(w_q4);
    lfm2_threadpool_free(&pool);
    return EXIT_SUCCESS;
}
#endif
