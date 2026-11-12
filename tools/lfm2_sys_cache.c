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
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifndef LFM2_SYS_CACHE_INCLUDED
#define LFM2_SYS_CACHE_INCLUDED 1

#if defined(LFM_GGUF_H) || defined(LFM_MODEL_H)
/* Compiling inside src/ core library environment (which already includes include/gguf.h) */
#ifndef LFM2_MAX_LAYERS
#define LFM2_MAX_LAYERS 64
#endif

#ifndef LFM2_BLOCK_TYPE_DEFINED
#define LFM2_BLOCK_TYPE_DEFINED 1
typedef enum {
    LFM2_BLOCK_SHORTCONV = 0,
    LFM2_BLOCK_GQA_ATTN  = 1,
    LFM2_BLOCK_SWA_ATTN  = 2
} lfm2_block_type_t;
#endif

typedef struct {
    uint32_t          layer_idx;
    lfm2_block_type_t block_type;
} LFM2Layer;

typedef struct {
    char     arch[32];
    bool     is_lfm3;
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t n_ctx;
    uint32_t n_head_q;
    uint32_t head_dim;
    uint32_t rope_dim;
    uint32_t n_head_kv[LFM2_MAX_LAYERS];
    bool     is_swa_layer[LFM2_MAX_LAYERS];
    uint32_t sliding_window;
    uint32_t shortconv_l_cache;
    float    rms_norm_eps;
    float    rope_freq_base;
    float    rope_freq_base_swa;
    uint32_t alignment;
} LFM2Config;

typedef struct {
    int         fd;
    size_t      file_size;
    const uint8_t *mmap_base;
    const uint8_t *tensor_data_base;
    LFM2Config  cfg;
    uint64_t    n_tensors;
    GGUFTensor *tensors;
    LFM2Layer   layers[LFM2_MAX_LAYERS];
} LFM2Model;

#else

/* Standalone tools compilation: include tools/lfm2_loader.c */
#ifndef LFM2_LOADER_INCLUDED
#define LFM2_LOADER_NO_MAIN 1
#if __has_include("tools/lfm2_loader.c")
#include "tools/lfm2_loader.c"
#elif __has_include("lfm2_loader.c")
#include "lfm2_loader.c"
#else
#include "../tools/lfm2_loader.c"
#endif
#endif

#endif

/* ============================================================================
 * 1. Linux /proc/meminfo Parser, OOM Guard & Auto-Context Clamper
 * ============================================================================ */
typedef struct {
    uint64_t mem_total_bytes;
    uint64_t mem_free_bytes;
    uint64_t mem_available_bytes;
    uint64_t page_cache_bytes;
    uint64_t swap_total_bytes;
    uint64_t swap_free_bytes;
} LFM2LinuxMemInfo;

static inline bool lfm2_read_linux_meminfo(LFM2LinuxMemInfo *info) {
    if (!info) return false;
    memset(info, 0, sizeof(*info));
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return false;

    char line[256];
    uint64_t buffers_kb = 0, cached_kb = 0, sreclaim_kb = 0;

    while (fgets(line, sizeof(line), f)) {
        char key[64];
        uint64_t val_kb = 0;
        if (sscanf(line, "%63[^:]: %" SCNu64 " kB", key, &val_kb) == 2) {
            if (strcmp(key, "MemTotal") == 0)          info->mem_total_bytes     = val_kb * 1024ULL;
            else if (strcmp(key, "MemFree") == 0)      info->mem_free_bytes      = val_kb * 1024ULL;
            else if (strcmp(key, "MemAvailable") == 0) info->mem_available_bytes = val_kb * 1024ULL;
            else if (strcmp(key, "Buffers") == 0)      buffers_kb                = val_kb;
            else if (strcmp(key, "Cached") == 0)       cached_kb                 = val_kb;
            else if (strcmp(key, "SReclaimable") == 0) sreclaim_kb               = val_kb;
            else if (strcmp(key, "SwapTotal") == 0)    info->swap_total_bytes    = val_kb * 1024ULL;
            else if (strcmp(key, "SwapFree") == 0)     info->swap_free_bytes     = val_kb * 1024ULL;
        }
    }
    fclose(f);

    info->page_cache_bytes = (buffers_kb + cached_kb + sreclaim_kb) * 1024ULL;
    if (info->mem_available_bytes == 0) {
        info->mem_available_bytes = info->mem_free_bytes + info->page_cache_bytes;
    }
    return (info->mem_total_bytes > 0);
}

typedef struct {
    uint64_t model_mmap_bytes;       /* Total GGUF file size                   */
    uint64_t active_weights_bytes;   /* Active weights pulled per token        */
    uint64_t scratch_arena_bytes;    /* Activation + RoPE + BPE tables         */
    uint64_t conv_state_bytes;       /* Total ShortConv rolling state (~160KB) */
    uint64_t kv_bytes_per_token;     /* FP16 KV-cache bytes across GQA layers  */
    uint32_t requested_ctx;
    uint32_t safe_clamped_ctx;       /* Adjusted context length to prevent OOM */
    bool     was_clamped;
    bool     fits_in_ram_without_swap;
} LFM2MemoryPlan;

/* Calculates exact LFM2 hybrid memory requirements and clamps `requested_ctx`
 * so the process never triggers the Linux OOM killer on resource-constrained machines. */
static inline LFM2MemoryPlan lfm2_plan_memory_and_guard_oom(const LFM2Model *m,
                                                            uint32_t requested_ctx,
                                                            uint64_t os_reserve_mb) {
    LFM2MemoryPlan plan;
    memset(&plan, 0, sizeof(plan));
    if (!m) return plan;

    const LFM2Config *c = &m->cfg;
    const uint32_t D        = c->n_embd;
    const uint32_t head_dim = (c->head_dim > 0) ? c->head_dim
                              : ((c->n_head_q > 0) ? (D / c->n_head_q) : 64);
    const uint32_t hist_len = (c->shortconv_l_cache > 1) ? (c->shortconv_l_cache - 1) : 2;
    const uint32_t swa_cap  = ((c->sliding_window > 0) ? c->sliding_window : 256U) + 8U;

    plan.model_mmap_bytes     = m->file_size;
    plan.active_weights_bytes = m->file_size; /* Dense default; lower for MoE */
    plan.scratch_arena_bytes  = 32ULL * 1024ULL * 1024ULL; /* ~32 MB scratch + LUTs */

    uint64_t kv_per_tok      = 0;
    uint64_t fixed_swa_bytes = 0;
    uint32_t n_conv          = 0;

    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            n_conv++;
        } else if (m->layers[l].block_type == LFM2_BLOCK_SWA_ATTN) {
            /* SWA layers hold fixed ring buffer (sliding_window + 8 slots), independent of max_seq_len */
            const uint64_t kv_dim = (uint64_t)c->n_head_kv[l] * head_dim;
            fixed_swa_bytes += 2ULL * (uint64_t)swa_cap * kv_dim * sizeof(uint16_t);
        } else {
            /* FP16 K-cache + FP16 V-cache: 2 * n_head_kv[l] * head_dim * sizeof(uint16_t) */
            const uint64_t kv_dim = (uint64_t)c->n_head_kv[l] * head_dim;
            kv_per_tok += 2ULL * kv_dim * sizeof(uint16_t);
        }
    }

    plan.conv_state_bytes   = (uint64_t)n_conv * hist_len * D * sizeof(float);
    plan.kv_bytes_per_token = kv_per_tok;

    if (requested_ctx == 0) requested_ctx = (uint32_t)LFM_DEFAULT_CTX_LEN;
    if (c->n_ctx > 0 && requested_ctx > c->n_ctx) requested_ctx = c->n_ctx;
    plan.requested_ctx    = requested_ctx;
    plan.safe_clamped_ctx = requested_ctx;

    LFM2LinuxMemInfo mem;
    if (!lfm2_read_linux_meminfo(&mem)) {
        plan.fits_in_ram_without_swap = true;
        return plan;
    }

    const uint64_t reserve_bytes = os_reserve_mb * 1024ULL * 1024ULL;
    const uint64_t usable_ram = (mem.mem_available_bytes > reserve_bytes)
                                ? (mem.mem_available_bytes - reserve_bytes)
                                : (mem.mem_available_bytes / 2);

    const uint64_t fixed_heap = plan.scratch_arena_bytes + plan.conv_state_bytes + fixed_swa_bytes;
    /* Keep at least 50% of the mmap'd weights resident alongside the heap */
    const uint64_t min_model_resident = plan.model_mmap_bytes / 2;

    if (usable_ram <= fixed_heap + min_model_resident) {
        /* Extreme low memory: clamp context to minimum 512 tokens */
        plan.safe_clamped_ctx         = 512;
        plan.was_clamped              = (requested_ctx > 512);
        plan.fits_in_ram_without_swap = false;
        return plan;
    }

    /* Check if full model + requested KV cache fits in available RAM without swap */
    const uint64_t req_kv_bytes = (uint64_t)requested_ctx * plan.kv_bytes_per_token;
    if (plan.model_mmap_bytes + fixed_heap + req_kv_bytes <= usable_ram) {
        plan.fits_in_ram_without_swap = true;
        return plan;
    }

    /* Otherwise, compute maximum context that fits inside remaining anonymous RAM */
    const uint64_t max_kv_budget = usable_ram - fixed_heap - min_model_resident;
    uint64_t max_tokens = (plan.kv_bytes_per_token > 0)
                          ? (max_kv_budget / plan.kv_bytes_per_token)
                          : requested_ctx;

    /* Round down to a clean multiple of 256 tokens (minimum 512) */
    max_tokens &= ~255ULL;
    if (max_tokens < 512) max_tokens = 512;

    if (max_tokens < requested_ctx) {
        plan.safe_clamped_ctx = (uint32_t)max_tokens;
        plan.was_clamped      = true;
    }
    plan.fits_in_ram_without_swap = false;
    return plan;
}

/* Allocates a 2 MB Transparent HugePage (MADV_HUGEPAGE) backed arena on Linux
 * to eliminate CPU TLB misses during multi-megabyte KV-cache & activation sweeps. */
static inline void *lfm2_alloc_hugepage_arena(size_t size_bytes) {
    const size_t align_2mb = 2ULL * 1024ULL * 1024ULL;
    const size_t aligned_sz = (size_bytes + align_2mb - 1) & ~(align_2mb - 1);

    void *ptr = mmap(NULL, aligned_sz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return NULL;

#ifdef MADV_HUGEPAGE
    madvise(ptr, aligned_sz, MADV_HUGEPAGE);
#endif
    return ptr;
}

static inline void lfm2_free_hugepage_arena(void *ptr, size_t size_bytes) {
    if (!ptr || ptr == MAP_FAILED) return;
    const size_t align_2mb = 2ULL * 1024ULL * 1024ULL;
    const size_t aligned_sz = (size_bytes + align_2mb - 1) & ~(align_2mb - 1);
    munmap(ptr, aligned_sz);
}

/* ============================================================================
 * 2. Sequential mmap Page-Warmup with Terminal Progress Bar (Anti-HDD Thrash)
 * ============================================================================ */
static inline double lfm2_monotonic_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Sequentially faults every 4 KB page of the mmap'd GGUF file into the Linux
 * page cache before token generation starts. Prevents random disk-head seeking
 * on mechanical HDDs and renders a real-time throughput bar. */
static inline uint64_t lfm2_warmup_mmap_pages(const LFM2Model *m, FILE *log_stream, bool show_progress) {
    if (!m || !m->mmap_base || m->file_size == 0) return 0;

    /* Hint Linux VFS to trigger aggressive sequential read-ahead */
    if (m->fd >= 0) {
        posix_fadvise(m->fd, 0, (off_t)m->file_size, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(m->fd, 0, (off_t)m->file_size, POSIX_FADV_WILLNEED);
    }
    madvise((void *)m->mmap_base, m->file_size, MADV_SEQUENTIAL | MADV_WILLNEED);

    long ps = sysconf(_SC_PAGESIZE);
    const size_t page_step  = (ps > 0) ? (size_t)ps : 4096;
    const size_t chunk_step = 16ULL * 1024ULL * 1024ULL; /* Update UI every 16 MB */
    const uint8_t *base     = m->mmap_base;
    const size_t total      = m->file_size;

    volatile uint64_t sink_acc = 0;
    const double t_start = lfm2_monotonic_sec();

    size_t offset = 0;
    while (offset < total) {
        size_t chunk_end = offset + chunk_step;
        if (chunk_end > total) chunk_end = total;

        /* Touch 1 byte per 4096-byte page sequentially */
        for (size_t p = offset; p < chunk_end; p += page_step) {
            sink_acc += base[p];
        }
        offset = chunk_end;

        if (show_progress && log_stream) {
            const double elapsed = lfm2_monotonic_sec() - t_start + 1e-6;
            const double mb_done = (double)offset / (1024.0 * 1024.0);
            const double mb_tot  = (double)total  / (1024.0 * 1024.0);
            const double speed   = mb_done / elapsed;
            const int    pct     = (int)((offset * 100ULL) / total);
            const int    bars    = (pct * 24) / 100;

            fprintf(log_stream, "\rWarming GGUF into RAM: [");
            for (int b = 0; b < 24; ++b) {
                fputc((b < bars) ? '#' : '.', log_stream);
            }
            fprintf(log_stream, "] %3d%% (%6.1f / %6.1f MB @ %6.1f MB/s)",
                    pct, mb_done, mb_tot, speed);
            fflush(log_stream);
        }
    }

    if (show_progress && log_stream) {
        fprintf(log_stream, "\n");
        fflush(log_stream);
    }
    return sink_acc;
}

/* ============================================================================
 * 3. Disk-Backed Hybrid Prompt Cache (--prompt-cache) Serializer & Loader
 * ============================================================================
 * Binary Layout (.lfmcache v1):
 *   - LFM2PromptCacheHeader (64 bytes, magic + model_fingerprint + counts)
 *   - uint32_t cached_tokens[n_tokens]
 *   - For each ShortConv layer l: float conv_state[(l_cache - 1) * n_embd]
 *   - For each GQA layer l      : uint16_t k_cache_f16[n_tokens * kv_dim]
 *                                 uint16_t v_cache_f16[n_tokens * kv_dim]
 *   - uint64_t payload_fnv1a_checksum (verifies file was not truncated)
 * ============================================================================ */
#define LFM2_PROMPT_CACHE_MAGIC   0x4350324D464C3130ULL /* "01LFM2PC" */
#define LFM2_PROMPT_CACHE_VERSION 1U

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t n_tokens;           /* Number of tokens in cached prefix (saved_pos) */
    uint64_t model_fingerprint;  /* FNV-1a of model architecture & tensor table   */
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_head_q;
    uint32_t shortconv_l_cache;
    uint64_t total_payload_bytes;
    uint8_t  reserved[16];
} LFM2PromptCacheHeader;
#pragma pack(pop)

static inline uint64_t lfm2_fnv1a_update(uint64_t h, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Computes a unique 64-bit fingerprint of the loaded LFM2 model so a prompt cache
 * created by LFM2.5-350M or Q4_0 is rejected if loaded into LFM2.5-1.2B or Q4_K_M */
static inline uint64_t lfm2_compute_model_fingerprint(const LFM2Model *m) {
    if (!m) return 0;
    uint64_t h = 14695981039346656037ULL;
    h = lfm2_fnv1a_update(h, m->cfg.arch, strlen(m->cfg.arch));
    h = lfm2_fnv1a_update(h, &m->cfg.n_layers, sizeof(uint32_t));
    h = lfm2_fnv1a_update(h, &m->cfg.n_embd, sizeof(uint32_t));
    h = lfm2_fnv1a_update(h, &m->cfg.n_vocab, sizeof(uint32_t));
    h = lfm2_fnv1a_update(h, &m->cfg.n_head_q, sizeof(uint32_t));
    h = lfm2_fnv1a_update(h, &m->cfg.shortconv_l_cache, sizeof(uint32_t));
    h = lfm2_fnv1a_update(h, m->cfg.n_head_kv, sizeof(uint32_t) * m->cfg.n_layers);
    if (m->cfg.is_lfm3) {
        h = lfm2_fnv1a_update(h, &m->cfg.head_dim, sizeof(uint32_t));
        h = lfm2_fnv1a_update(h, &m->cfg.sliding_window, sizeof(uint32_t));
        h = lfm2_fnv1a_update(h, m->cfg.is_swa_layer, sizeof(bool) * m->cfg.n_layers);
    }

    if (m->tensors) {
        for (uint64_t i = 0; i < m->n_tensors; ++i) {
            h = lfm2_fnv1a_update(h, m->tensors[i].name, strlen(m->tensors[i].name));
            h = lfm2_fnv1a_update(h, &m->tensors[i].type, sizeof(uint32_t));
#if defined(LFM_GGUF_H) || defined(LFM_MODEL_H)
            h = lfm2_fnv1a_update(h, m->tensors[i].dims, sizeof(uint64_t) * 4);
#else
            h = lfm2_fnv1a_update(h, m->tensors[i].ne, sizeof(uint64_t) * 4);
#endif
        }
    }
    return h;
}

static inline bool lfm2_write_exact_checked(FILE *f, const void *buf, size_t len, uint64_t *csum) {
    if (len == 0) return true;
    if (fwrite(buf, 1, len, f) != len) return false;
    *csum = lfm2_fnv1a_update(*csum, buf, len);
    return true;
}

static inline bool lfm2_read_exact_checked(FILE *f, void *buf, size_t len, uint64_t *csum) {
    if (len == 0) return true;
    if (fread(buf, 1, len, f) != len) return false;
    *csum = lfm2_fnv1a_update(*csum, buf, len);
    return true;
}

/* Fast FP32 <-> FP16 conversion helpers for universal cache interchange */
static inline uint16_t lfm2_fp32_to_fp16_scalar(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exp = ((u.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u.u & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float lfm2_fp16_to_fp32_scalar(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0) {
        if (mant == 0) { union { uint32_t u; float f; } u = { sign }; return u.f; }
        while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x00800000;
    } else if (exp == 31) {
        union { uint32_t u; float f; } u = { sign | 0x7F800000 | mant };
        return u.f;
    }
    exp = exp + (127 - 15);
    union { uint32_t u; float f; } u = { sign | (exp << 23) | mant };
    return u.f;
}

static inline void lfm2_convert_f32_to_f16(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(dst + i), h);
    }
#elif defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + i + 0));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + i + 4));
        vst1q_u16(dst + i, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
    }
#endif
    for (; i < n; ++i) dst[i] = lfm2_fp32_to_fp16_scalar(src[i]);
}

static inline void lfm2_convert_f16_to_f32(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256 v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, v);
    }
#elif defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(src + i);
        vst1q_f32(dst + i + 0, vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h))));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(h))));
    }
#endif
    for (; i < n; ++i) dst[i] = lfm2_fp16_to_fp32_scalar(src[i]);
}

/* Saves the exact LFM2/LFM3 Hybrid State (ShortConv buffers + FP16 SWA/GQA KV-caches) */
static inline bool lfm2_prompt_cache_save_ex(const char *filepath,
                                             const LFM2Model *m,
                                             const uint32_t *tokens,
                                             uint32_t n_tokens,
                                             float * const conv_state[LFM2_MAX_LAYERS],
                                             const uint32_t *conv_ring_heads,
                                             uint16_t * const k_cache_f16[LFM2_MAX_LAYERS],
                                             uint16_t * const v_cache_f16[LFM2_MAX_LAYERS],
                                             uint32_t swa_cap) {
    if (!filepath || !m || !tokens || n_tokens == 0) return false;

    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", filepath, (int)getpid());

    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        perror("fopen(prompt_cache_save)");
        return false;
    }

    const LFM2Config *c = &m->cfg;
    const uint32_t D        = c->n_embd;
    const uint32_t head_dim = (c->head_dim > 0) ? c->head_dim
                              : ((c->n_head_q > 0) ? (D / c->n_head_q) : 64);
    const uint32_t hist_len = (c->shortconv_l_cache > 1) ? (c->shortconv_l_cache - 1) : 2;
    const uint32_t swa_win  = (c->sliding_window > 0) ? c->sliding_window : 256U;
    const uint32_t eff_cap  = (swa_cap > 0) ? swa_cap : (swa_win + 8U);
    const size_t   conv_bytes = (size_t)hist_len * D * sizeof(float);

    uint64_t payload_bytes = (uint64_t)n_tokens * sizeof(uint32_t);
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            payload_bytes += conv_bytes;
        } else if (m->layers[l].block_type == LFM2_BLOCK_SWA_ATTN) {
            const uint32_t w_valid = (n_tokens < swa_win) ? n_tokens : swa_win;
            const size_t kv_dim    = (size_t)c->n_head_kv[l] * head_dim;
            payload_bytes += 2ULL * (uint64_t)w_valid * kv_dim * sizeof(uint16_t);
        } else {
            const size_t kv_dim = (size_t)c->n_head_kv[l] * head_dim;
            payload_bytes += 2ULL * (uint64_t)n_tokens * kv_dim * sizeof(uint16_t);
        }
    }

    LFM2PromptCacheHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic               = LFM2_PROMPT_CACHE_MAGIC;
    hdr.version             = LFM2_PROMPT_CACHE_VERSION;
    hdr.n_tokens            = n_tokens;
    hdr.model_fingerprint   = lfm2_compute_model_fingerprint(m);
    hdr.n_layers            = c->n_layers;
    hdr.n_embd              = c->n_embd;
    hdr.n_head_q            = c->n_head_q;
    hdr.shortconv_l_cache   = c->shortconv_l_cache;
    hdr.total_payload_bytes = payload_bytes;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    uint64_t csum = 14695981039346656037ULL;

    /* 1. Write token ID array */
    if (!lfm2_write_exact_checked(f, tokens, (size_t)n_tokens * sizeof(uint32_t), &csum)) {
        goto fail;
    }

    float *lin_conv = conv_ring_heads ? (float *)malloc(conv_bytes) : NULL;
    if (conv_ring_heads && !lin_conv) goto fail;

    /* 2. Write Hybrid Layer States */
    for (uint32_t l = 0; l < c->n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            const float *src = conv_state[l];
            if (conv_ring_heads && hist_len > 1U) {
                const uint32_t head = conv_ring_heads[l];
                for (uint32_t s = 0; s < hist_len; ++s) {
                    const uint32_t slot = (head + s) % hist_len;
                    memcpy(lin_conv + (size_t)s * D, src + (size_t)slot * D, (size_t)D * sizeof(float));
                }
                src = lin_conv;
            }
            if (!lfm2_write_exact_checked(f, src, conv_bytes, &csum)) { free(lin_conv); goto fail; }
        } else if (m->layers[l].block_type == LFM2_BLOCK_SWA_ATTN) {
            const uint32_t w_valid   = (n_tokens < swa_win) ? n_tokens : swa_win;
            const uint32_t t_start   = n_tokens - w_valid;
            const size_t   kv_dim    = (size_t)c->n_head_kv[l] * head_dim;
            const size_t   row_bytes = kv_dim * sizeof(uint16_t);
            for (uint32_t t = t_start; t < n_tokens; ++t) {
                const uint32_t slot = t % eff_cap;
                if (!lfm2_write_exact_checked(f, k_cache_f16[l] + (size_t)slot * kv_dim, row_bytes, &csum)) {
                    free(lin_conv); goto fail;
                }
            }
            for (uint32_t t = t_start; t < n_tokens; ++t) {
                const uint32_t slot = t % eff_cap;
                if (!lfm2_write_exact_checked(f, v_cache_f16[l] + (size_t)slot * kv_dim, row_bytes, &csum)) {
                    free(lin_conv); goto fail;
                }
            }
        } else {
            const size_t kv_dim   = (size_t)c->n_head_kv[l] * head_dim;
            const size_t kv_bytes = (size_t)n_tokens * kv_dim * sizeof(uint16_t);
            if (!lfm2_write_exact_checked(f, k_cache_f16[l], kv_bytes, &csum)) { free(lin_conv); goto fail; }
            if (!lfm2_write_exact_checked(f, v_cache_f16[l], kv_bytes, &csum)) { free(lin_conv); goto fail; }
        }
    }
    free(lin_conv);

    /* 3. Write trailing 64-bit FNV-1a integrity checksum */
    if (fwrite(&csum, sizeof(csum), 1, f) != 1) goto fail;
    if (fflush(f) != 0) goto fail;
    fsync(fileno(f));
    fclose(f);

    /* Atomically replace destination cache file */
    if (rename(tmp_path, filepath) != 0) {
        unlink(tmp_path);
        return false;
    }
    return true;

fail:
    fclose(f);
    unlink(tmp_path);
    return false;
}

static inline bool lfm2_prompt_cache_save(const char *filepath,
                                          const LFM2Model *m,
                                          const uint32_t *tokens,
                                          uint32_t n_tokens,
                                          float * const conv_state[LFM2_MAX_LAYERS],
                                          uint16_t * const k_cache_f16[LFM2_MAX_LAYERS],
                                          uint16_t * const v_cache_f16[LFM2_MAX_LAYERS]) {
    return lfm2_prompt_cache_save_ex(filepath, m, tokens, n_tokens,
                                     conv_state, NULL, k_cache_f16, v_cache_f16, 0U);
}

/* Attempts to load `filepath` and match its cached token sequence against the
 * prefix of `prompt_tokens[0 .. n_prompt - 1]`.
 *
 * CRITICAL LFM2 HYBRID RULE:
 * Because ShortConv state is recurrent, the cache can be used if and only if
 * `n_prompt >= hdr.n_tokens` and `prompt_tokens[0 .. hdr.n_tokens - 1]` matches
 * the cached token prefix 100% bit-for-bit.
 *
 * On match, restores ShortConv buffers + GQA FP16 KV caches, sets `*out_matched_pos = hdr.n_tokens`,
 * and returns `true` so the caller only prefills `prompt_tokens + hdr.n_tokens`! */
static inline bool lfm2_prompt_cache_try_load_prefix_ex(const char *filepath,
                                                        const LFM2Model *m,
                                                        const uint32_t *prompt_tokens,
                                                        uint32_t n_prompt,
                                                        uint32_t max_seq_len,
                                                        float *conv_state[LFM2_MAX_LAYERS],
                                                        uint32_t *out_conv_heads,
                                                        uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                                        uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                                        uint32_t swa_cap,
                                                        uint32_t *out_matched_pos) {
    if (out_matched_pos) *out_matched_pos = 0;
    if (!filepath || !m || !prompt_tokens) return false;

    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    LFM2PromptCacheHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return false;
    }

    const uint64_t expected_fp = lfm2_compute_model_fingerprint(m);
    if (hdr.magic             != LFM2_PROMPT_CACHE_MAGIC   ||
        hdr.version           != LFM2_PROMPT_CACHE_VERSION ||
        hdr.model_fingerprint != expected_fp               ||
        hdr.n_layers          != m->cfg.n_layers           ||
        hdr.n_embd            != m->cfg.n_embd             ||
        hdr.n_tokens          == 0                         ||
        hdr.n_tokens          >  n_prompt                  ||
        hdr.n_tokens          >  max_seq_len) {
        fclose(f);
        return false;
    }

    uint64_t csum = 14695981039346656037ULL;

    /* 1. Read cached token IDs and verify exact prefix match */
    uint32_t *cached_ids = (uint32_t *)malloc((size_t)hdr.n_tokens * sizeof(uint32_t));
    if (!cached_ids) {
        fclose(f);
        return false;
    }

    if (!lfm2_read_exact_checked(f, cached_ids, (size_t)hdr.n_tokens * sizeof(uint32_t), &csum)) {
        free(cached_ids);
        fclose(f);
        return false;
    }

    if (memcmp(cached_ids, prompt_tokens, (size_t)hdr.n_tokens * sizeof(uint32_t)) != 0) {
        /* Prefix mismatch: cannot use recurrent ShortConv state */
        free(cached_ids);
        fclose(f);
        return false;
    }
    free(cached_ids);

    /* 2. Read ShortConv states and SWA/GQA FP16 KV-Caches directly into runtime buffers */
    const uint32_t D          = m->cfg.n_embd;
    const uint32_t head_dim   = (m->cfg.head_dim > 0) ? m->cfg.head_dim
                                : ((m->cfg.n_head_q > 0) ? (D / m->cfg.n_head_q) : 64);
    const uint32_t hist_len   = (m->cfg.shortconv_l_cache > 1) ? (m->cfg.shortconv_l_cache - 1) : 2;
    const uint32_t swa_win    = (m->cfg.sliding_window > 0) ? m->cfg.sliding_window : 256U;
    const uint32_t eff_cap    = (swa_cap > 0) ? swa_cap : (swa_win + 8U);
    const size_t   conv_bytes = (size_t)hist_len * D * sizeof(float);

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            if (conv_state[l] != NULL) {
                if (!lfm2_read_exact_checked(f, conv_state[l], conv_bytes, &csum)) {
                    fclose(f);
                    return false;
                }
                if (out_conv_heads) out_conv_heads[l] = 0U;
            } else {
                /* Discard if caller buffer is null */
                float dummy[1024];
                size_t rem = conv_bytes;
                while (rem > 0) {
                    size_t chunk = (rem < sizeof(dummy)) ? rem : sizeof(dummy);
                    if (!lfm2_read_exact_checked(f, dummy, chunk, &csum)) { fclose(f); return false; }
                    rem -= chunk;
                }
            }
        } else if (m->layers[l].block_type == LFM2_BLOCK_SWA_ATTN) {
            const uint32_t w_valid   = (hdr.n_tokens < swa_win) ? hdr.n_tokens : swa_win;
            const uint32_t t_start   = hdr.n_tokens - w_valid;
            const size_t   kv_dim    = (size_t)m->cfg.n_head_kv[l] * head_dim;
            const size_t   row_bytes = kv_dim * sizeof(uint16_t);
            if (k_cache_f16[l] != NULL && v_cache_f16[l] != NULL) {
                for (uint32_t t = t_start; t < hdr.n_tokens; ++t) {
                    const uint32_t slot = t % eff_cap;
                    if (!lfm2_read_exact_checked(f, k_cache_f16[l] + (size_t)slot * kv_dim, row_bytes, &csum)) {
                        fclose(f); return false;
                    }
                }
                for (uint32_t t = t_start; t < hdr.n_tokens; ++t) {
                    const uint32_t slot = t % eff_cap;
                    if (!lfm2_read_exact_checked(f, v_cache_f16[l] + (size_t)slot * kv_dim, row_bytes, &csum)) {
                        fclose(f); return false;
                    }
                }
            } else {
                uint8_t dummy[2048];
                size_t total_skip = 2ULL * (size_t)w_valid * row_bytes;
                while (total_skip > 0) {
                    size_t chunk = (total_skip < sizeof(dummy)) ? total_skip : sizeof(dummy);
                    if (!lfm2_read_exact_checked(f, dummy, chunk, &csum)) { fclose(f); return false; }
                    total_skip -= chunk;
                }
            }
        } else {
            const size_t kv_dim   = (size_t)m->cfg.n_head_kv[l] * head_dim;
            const size_t kv_bytes = (size_t)hdr.n_tokens * kv_dim * sizeof(uint16_t);
            if (k_cache_f16[l] != NULL && v_cache_f16[l] != NULL) {
                if (!lfm2_read_exact_checked(f, k_cache_f16[l], kv_bytes, &csum) ||
                    !lfm2_read_exact_checked(f, v_cache_f16[l], kv_bytes, &csum)) {
                    fclose(f);
                    return false;
                }
            } else {
                uint8_t dummy[2048];
                for (int pass = 0; pass < 2; ++pass) {
                    size_t rem = kv_bytes;
                    while (rem > 0) {
                        size_t chunk = (rem < sizeof(dummy)) ? rem : sizeof(dummy);
                        if (!lfm2_read_exact_checked(f, dummy, chunk, &csum)) { fclose(f); return false; }
                        rem -= chunk;
                    }
                }
            }
        }
    }

    /* 3. Verify trailing 64-bit checksum */
    uint64_t file_csum = 0;
    if (fread(&file_csum, sizeof(file_csum), 1, f) != 1 || file_csum != csum) {
        fprintf(stderr, "[WARN] Prompt cache %s failed checksum verification.\n", filepath);
        fclose(f);
        return false;
    }

    fclose(f);
    if (out_matched_pos) *out_matched_pos = hdr.n_tokens;
    return true;
}

static inline bool lfm2_prompt_cache_try_load_prefix(const char *filepath,
                                                     const LFM2Model *m,
                                                     const uint32_t *prompt_tokens,
                                                     uint32_t n_prompt,
                                                     uint32_t max_seq_len,
                                                     float *conv_state[LFM2_MAX_LAYERS],
                                                     uint16_t *k_cache_f16[LFM2_MAX_LAYERS],
                                                     uint16_t *v_cache_f16[LFM2_MAX_LAYERS],
                                                     uint32_t *out_matched_pos) {
    return lfm2_prompt_cache_try_load_prefix_ex(filepath, m, prompt_tokens, n_prompt, max_seq_len,
                                                conv_state, NULL, k_cache_f16, v_cache_f16, 0U,
                                                out_matched_pos);
}

/* Universal FP32 wrapper saving prompt cache when caller uses float* KV buffers */
static inline bool lfm2_prompt_cache_save_f32(const char *filepath,
                                              const LFM2Model *m,
                                              const uint32_t *tokens,
                                              uint32_t n_tokens,
                                              float * const conv_state[LFM2_MAX_LAYERS],
                                              float * const k_cache[LFM2_MAX_LAYERS],
                                              float * const v_cache[LFM2_MAX_LAYERS]) {
    if (!filepath || !m || !tokens || n_tokens == 0) return false;

    const uint32_t D        = m->cfg.n_embd;
    const uint32_t head_dim = (m->cfg.n_head_q > 0) ? (D / m->cfg.n_head_q) : 64;

    uint16_t *k_f16[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_f16[LFM2_MAX_LAYERS] = {0};

    bool ok = true;
    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_GQA_ATTN) {
            const size_t kv_dim   = (size_t)m->cfg.n_head_kv[l] * head_dim;
            const size_t count    = (size_t)n_tokens * kv_dim;
            k_f16[l] = (uint16_t *)malloc(count * sizeof(uint16_t));
            v_f16[l] = (uint16_t *)malloc(count * sizeof(uint16_t));
            if (!k_f16[l] || !v_f16[l]) { ok = false; break; }
            if (k_cache[l]) lfm2_convert_f32_to_f16(k_f16[l], k_cache[l], count);
            if (v_cache[l]) lfm2_convert_f32_to_f16(v_f16[l], v_cache[l], count);
        }
    }

    if (ok) {
        ok = lfm2_prompt_cache_save(filepath, m, tokens, n_tokens, conv_state, k_f16, v_f16);
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        free(k_f16[l]);
        free(v_f16[l]);
    }
    return ok;
}

/* Universal FP32 wrapper loading prompt cache into float* KV buffers */
static inline bool lfm2_prompt_cache_try_load_prefix_f32(const char *filepath,
                                                         const LFM2Model *m,
                                                         const uint32_t *prompt_tokens,
                                                         uint32_t n_prompt,
                                                         uint32_t max_seq_len,
                                                         float *conv_state[LFM2_MAX_LAYERS],
                                                         float *k_cache[LFM2_MAX_LAYERS],
                                                         float *v_cache[LFM2_MAX_LAYERS],
                                                         uint32_t *out_matched_pos) {
    if (out_matched_pos) *out_matched_pos = 0;
    if (!filepath || !m || !prompt_tokens) return false;

    const uint32_t D        = m->cfg.n_embd;
    const uint32_t head_dim = (m->cfg.n_head_q > 0) ? (D / m->cfg.n_head_q) : 64;

    uint16_t *k_f16[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_f16[LFM2_MAX_LAYERS] = {0};

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        if (m->layers[l].block_type == LFM2_BLOCK_GQA_ATTN) {
            const size_t kv_dim = (size_t)m->cfg.n_head_kv[l] * head_dim;
            const size_t count  = (size_t)max_seq_len * kv_dim;
            k_f16[l] = (uint16_t *)malloc(count * sizeof(uint16_t));
            v_f16[l] = (uint16_t *)malloc(count * sizeof(uint16_t));
            if (!k_f16[l] || !v_f16[l]) {
                for (uint32_t j = 0; j <= l; ++j) { free(k_f16[j]); free(v_f16[j]); }
                return false;
            }
        }
    }

    uint32_t matched = 0;
    bool ok = lfm2_prompt_cache_try_load_prefix(filepath, m, prompt_tokens, n_prompt,
                                                max_seq_len, conv_state, k_f16, v_f16, &matched);

    if (ok && matched > 0) {
        for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
            if (m->layers[l].block_type == LFM2_BLOCK_GQA_ATTN) {
                const size_t kv_dim = (size_t)m->cfg.n_head_kv[l] * head_dim;
                const size_t count  = (size_t)matched * kv_dim;
                if (k_cache[l]) lfm2_convert_f16_to_f32(k_cache[l], k_f16[l], count);
                if (v_cache[l]) lfm2_convert_f16_to_f32(v_cache[l], v_f16[l], count);
            }
        }
        if (out_matched_pos) *out_matched_pos = matched;
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        free(k_f16[l]);
        free(v_f16[l]);
    }
    return ok;
}

#endif /* LFM2_SYS_CACHE_INCLUDED */

#ifndef lfm2_aligned_alloc_64
#define lfm2_aligned_alloc_64 lfm2_aligned_alloc_64
static inline void *lfm2_aligned_alloc_64(size_t size) {
    size_t aligned_size = (size + 63) & ~((size_t)63);
    return aligned_alloc(64, aligned_size);
}
#endif

/* ============================================================================
 * 4. Standalone Verification Suite (OOM Guard, Page Warmup & Prompt Cache)
 * ============================================================================ */
#ifndef LFM2_SYS_CACHE_NO_MAIN
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("=== LFM2 v1.0 Linux OOM Guard, Page Warmup & Hybrid Prompt Cache Test ===\n\n");

    /* 1. Build a mock LFM2.5-1.2B model (16 layers: 10 ShortConv + 6 GQA, D=2048) */
    LFM2Model model;
    memset(&model, 0, sizeof(model));
    snprintf(model.cfg.arch, sizeof(model.cfg.arch), "lfm2");
    model.cfg.n_layers          = 16;
    model.cfg.n_embd            = 2048;
    model.cfg.n_ff              = 6144;
    model.cfg.n_vocab           = 65536;
    model.cfg.n_ctx             = 32768;
    model.cfg.n_head_q          = 32;
    model.cfg.shortconv_l_cache = 3;
    model.file_size             = 735ULL * 1024ULL * 1024ULL; /* 735 MB Q4_K_M */

    for (uint32_t l = 0; l < 16; ++l) {
        bool is_gqa = (l == 2 || l == 5 || l == 8 || l == 11 || l == 13 || l == 15);
        model.layers[l].block_type = is_gqa ? LFM2_BLOCK_GQA_ATTN : LFM2_BLOCK_SHORTCONV;
        model.cfg.n_head_kv[l]     = is_gqa ? 8 : 0;
    }

    /* ------------------------------------------------------------------------
     * Test 1: Live Linux /proc/meminfo Inspection & OOM Guard
     * ------------------------------------------------------------------------ */
    LFM2LinuxMemInfo sys_mem;
    bool mem_ok = lfm2_read_linux_meminfo(&sys_mem);

    LFM2MemoryPlan plan = lfm2_plan_memory_and_guard_oom(&model, 32768, 512);

    printf("1. Linux /proc/meminfo & Hybrid LFM2 Memory Planner:\n");
    if (mem_ok) {
        printf("   System Total / Available RAM : %6.1f MB / %6.1f MB (PageCache: %.1f MB)\n",
               (double)sys_mem.mem_total_bytes / (1024.0 * 1024.0),
               (double)sys_mem.mem_available_bytes / (1024.0 * 1024.0),
               (double)sys_mem.page_cache_bytes / (1024.0 * 1024.0));
    }
    printf("   ShortConv State (10 layers)  : %6.1f KB total (constant O(1) memory!)\n",
           (double)plan.conv_state_bytes / 1024.0);
    printf("   GQA FP16 KV-Cache (6 layers) : %6.1f KB / token (%.1f MB for %u ctx)\n",
           (double)plan.kv_bytes_per_token / 1024.0,
           (double)(plan.safe_clamped_ctx * plan.kv_bytes_per_token) / (1024.0 * 1024.0),
           plan.safe_clamped_ctx);
    printf("   Context Decision             : req=%u -> safe=%u (clamped=%s, zero_swap=%s)\n\n",
           plan.requested_ctx, plan.safe_clamped_ctx,
           plan.was_clamped ? "YES" : "NO",
           plan.fits_in_ram_without_swap ? "YES" : "NO");

    /* ------------------------------------------------------------------------
     * Test 2: Sequential mmap Page-Warmup Progress Bar (64 MB Synthetic Buffer)
     * ------------------------------------------------------------------------ */
    printf("2. Testing Sequential 4KB-Stride mmap Warmup Progress Bar:\n");
    model.file_size = 64ULL * 1024ULL * 1024ULL;
    void *dummy_mmap = lfm2_alloc_hugepage_arena(model.file_size);
    model.mmap_base  = (const uint8_t *)dummy_mmap;
    model.fd         = -1;

    lfm2_warmup_mmap_pages(&model, stdout, true);
    lfm2_free_hugepage_arena(dummy_mmap, model.file_size);
    model.mmap_base = NULL;
    printf("\n");

    /* ------------------------------------------------------------------------
     * Test 3: Disk-Backed Hybrid Prompt Cache (--prompt-cache) Round-Trip
     * ------------------------------------------------------------------------ */
    const uint32_t n_sys_prefix = 64; /* 64-token system prompt + tool definitions */
    const uint32_t n_full_prompt= 80; /* 64-token prefix + 16-token user question  */
    const uint32_t D            = model.cfg.n_embd;
    const uint32_t kv_dim       = 8 * 64; /* 512 */

    uint32_t full_prompt[80];
    for (uint32_t i = 0; i < n_full_prompt; ++i) full_prompt[i] = 100 + i * 7;

    float    *conv_state[LFM2_MAX_LAYERS]  = {0};
    uint16_t *k_cache_f16[LFM2_MAX_LAYERS] = {0};
    uint16_t *v_cache_f16[LFM2_MAX_LAYERS] = {0};

    for (uint32_t l = 0; l < 16; ++l) {
        if (model.layers[l].block_type == LFM2_BLOCK_SHORTCONV) {
            conv_state[l] = (float *)lfm2_aligned_alloc_64(2 * D * sizeof(float));
            for (uint32_t i = 0; i < 2 * D; ++i) {
                conv_state[l][i] = (float)(l * 1000 + i) * 0.25f;
            }
        } else {
            k_cache_f16[l] = (uint16_t *)lfm2_aligned_alloc_64(512 * kv_dim * sizeof(uint16_t));
            v_cache_f16[l] = (uint16_t *)lfm2_aligned_alloc_64(512 * kv_dim * sizeof(uint16_t));
            for (uint32_t i = 0; i < n_sys_prefix * kv_dim; ++i) {
                k_cache_f16[l][i] = (uint16_t)((l * 31 + i) & 0x7FFF);
                v_cache_f16[l][i] = (uint16_t)((l * 47 + i) & 0x7FFF);
            }
        }
    }

    const char *cache_file = "/tmp/lfm2_test_sys.lfmcache";
    double t0 = lfm2_monotonic_sec();
    bool saved = lfm2_prompt_cache_save(cache_file, &model, full_prompt, n_sys_prefix,
                                        conv_state, k_cache_f16, v_cache_f16);
    double save_ms = (lfm2_monotonic_sec() - t0) * 1000.0;

    /* Wipe runtime state completely to simulate a fresh process startup */
    for (uint32_t l = 0; l < 16; ++l) {
        if (conv_state[l])  memset(conv_state[l], 0, 2 * D * sizeof(float));
        if (k_cache_f16[l]) memset(k_cache_f16[l], 0, 512 * kv_dim * sizeof(uint16_t));
        if (v_cache_f16[l]) memset(v_cache_f16[l], 0, 512 * kv_dim * sizeof(uint16_t));
    }

    /* Reload from disk against `full_prompt` (80 tokens, first 64 match cache!) */
    uint32_t matched_pos = 0;
    t0 = lfm2_monotonic_sec();
    bool loaded = lfm2_prompt_cache_try_load_prefix(
        cache_file, &model, full_prompt, n_full_prompt, 512,
        conv_state, k_cache_f16, v_cache_f16, &matched_pos);
    double load_ms = (lfm2_monotonic_sec() - t0) * 1000.0;

    bool data_intact = saved && loaded && (matched_pos == n_sys_prefix) &&
                       (conv_state[0][13] == 13.0f * 0.25f) &&
                       (k_cache_f16[2][17] == (uint16_t)((2 * 31 + 17) & 0x7FFF)) &&
                       (v_cache_f16[15][99] == (uint16_t)((15 * 47 + 99) & 0x7FFF));

    struct stat st;
    stat(cache_file, &st);
    unlink(cache_file);

    printf("3. Disk-Backed Hybrid Prompt Cache (--prompt-cache) Verifier:\n");
    printf("   Cache File Size (64 tokens) : %.1f KB (10 ShortConv + 6 GQA FP16 KV)\n",
           (double)st.st_size / 1024.0);
    printf("   Atomic Save / Load Latency  : %.2f ms / %.2f ms\n", save_ms, load_ms);
    printf("   Matched Prefix Continuation : skipped %u/80 tokens (only prefill last %u!)\n",
           matched_pos, n_full_prompt - matched_pos);
    printf("   Checksum & Bit-Exact State  : %s\n", data_intact ? "PASS" : "FAIL");

    for (uint32_t l = 0; l < 16; ++l) {
        free(conv_state[l]);
        free(k_cache_f16[l]);
        free(v_cache_f16[l]);
    }

    if (!data_intact) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
#endif
