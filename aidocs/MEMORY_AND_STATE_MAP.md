# Memory & State Management Reference

This document details the memory layout, arena allocators, page residency controls, and OS telemetry mechanisms in LAII.

---

## 1. Linear Memory Arena Allocator (`arena.c`, `include/arena.h`)

All dynamic activations, intermediate vector buffers, and temporary forward pass tensors are managed through a preallocated, cache-line aligned linear arena ([`LFMArena`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/arena.h)).

### 1.1 Structural Invariants
```c
typedef struct {
    uint8_t *buffer;      /* Base pointer, strictly 64-byte aligned */
    size_t   capacity;    /* Total arena size in bytes              */
    size_t   offset;      /* Current allocation cursor              */
} LFMArena;
```

* **Strict 64-Byte Alignment:** Every allocation offset is padded using bitwise arithmetic:
  ```c
  size_t aligned_size = (size + 63) & ~((size_t)63);
  ```
* **Zero Runtime Overhead:** Allocation is a simple pointer bump (`offset += aligned_size`) taking $< 5\ \text{ns}$.
* **Linear Reset:** At the end of each autoregressive token step, calling `arena_reset(arena)` resets `offset = 0` with zero operating-system syscalls or memory fragmentation.
* **OOM Boundary Protection:** If `offset + aligned_size > capacity`, the allocator returns `NULL` and logs an out-of-memory error without corrupting memory.

---

## 2. Hybrid State Memory Layout

LAII splits memory into three primary regions:

```
+-----------------------------------------------------------------------------+
|                            System Physical DRAM                             |
+-----------------------------------------------------------------------------+
| 1. Read-Only Model Weights (Zero-Copy mmap)                                  |
|    - Direct virtual memory mapping of the GGUF file                         |
|    - Paged into RAM sequentially via lfm2_warmup_mmap_pages (--warmup)       |
+-----------------------------------------------------------------------------+
| 2. Persistent Engine State Slabs                                            |
|    - ShortConv Rolling Buffers: N_conv * 2 * D * sizeof(float) (~160 KB)     |
|    - GQA FP16 KV-Cache: N_gqa * 2 * N_kv_heads * head_dim * N_ctx * 2 bytes |
|    - Precomputed RoPE LUT: max_seq_len * (head_dim / 2) * 2 * sizeof(float) |
|    - BPE Hash Tables & PDA Grammar Tables (~8-16 MB)                         |
+-----------------------------------------------------------------------------+
| 3. Ephemeral Scratch Arena (LFMArena) (~32 MB)                              |
|    - Normalized input vector x_norm [D]                                     |
|    - Fused SwiGLU hidden buffers [n_ff]                                     |
|    - Pre-quantized block_q8_K activations [n_cols / 256]                    |
|    - Softmax attention running max/sum registers                            |
+-----------------------------------------------------------------------------+
```

### 2.1 ShortConv Rolling State Buffers
* In LFM2-1.2B ($D=2048$, 10 ShortConv layers):
  $$\text{Bytes} = 10 \times 2 \times 2048 \times 4\ \text{bytes} = 163,840\ \text{bytes}\ (\sim 160\ \text{KB})$$
* This state fits entirely inside the CPU's **L2/L3 cache**.
* **Constant $O(1)$ Memory:** Unlike attention layers, ShortConv memory never expands with sequence length.

### 2.2 GQA FP16 Key-Value Cache
* In LFM2-1.2B (6 GQA layers, $n_{\text{kv\_heads}} = 8, d_{\text{head}} = 64$):
  $$\text{Bytes per token} = 6 \times 2 \times 8 \times 64 \times 2\ \text{bytes (FP16)} = 12,288\ \text{bytes/token}\ (\sim 12\ \text{KB/tok})$$
* For a 4,096-token context window:
  $$\text{KV Cache Footprint} = 4,096 \times 12,288\ \text{bytes} = 48\ \text{MB}$$
* Compare to a standard 1.2B Transformer (all 16 layers attention): $\approx 128\ \text{MB}$. LAII achieves a **62.5% reduction in KV-cache size**.

---

## 3. Linux `/proc/meminfo` OOM Guard

To prevent Linux OOM killer termination on 8 GB RAM machines or memory-constrained cloud instances, [`tools/lfm2_sys_cache.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c) implements a proactive memory planner.

### 3.1 Kernel Telemetry Extraction
Directly reads `/proc/meminfo` to capture:
* `MemTotal`
* `MemAvailable` (the true headroom before swapping, unlike `MemFree`)
* Reclaimable Page Cache (`Buffers` + `Cached` + `SReclaimable`)

### 3.2 Dynamic Context Clamping Formula
```c
uint64_t reserve_bytes = os_reserve_mb * 1024ULL * 1024ULL; // Default: 512 MB
uint64_t usable_ram = (mem.mem_available_bytes > reserve_bytes)
                      ? (mem.mem_available_bytes - reserve_bytes)
                      : (mem.mem_available_bytes / 2);

uint64_t fixed_heap = plan.scratch_arena_bytes + plan.conv_state_bytes;
uint64_t min_model_resident = plan.model_mmap_bytes / 2; // Keep at least 50% weights resident

uint64_t max_kv_budget = usable_ram - fixed_heap - min_model_resident;
uint64_t max_tokens = max_kv_budget / plan.kv_bytes_per_token;

// Round down to multiple of 256 tokens, min 512
max_tokens = (max_tokens & ~255ULL);
if (max_tokens < 512) max_tokens = 512;
```

---

## 4. Transparent 2 MB HugePage Arena (`MADV_HUGEPAGE`)

When performing multi-megabyte matrix sweeps across the KV-cache, standard 4 KB OS pages result in thousands of TLB (Translation Lookaside Buffer) misses.

[`lfm2_alloc_hugepage_arena`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c#L196) allocates memory aligned to 2 MB boundaries:
```c
void *lfm2_alloc_hugepage_arena(size_t size_bytes) {
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
```

---

## 5. Sequential `mmap` Page Warmup (`--warmup`)

```text
Cold mmap:           Random 4 KB faults during Token 0 -> 20+ sec stall on mechanical disks
Sequential Warmup:   Sequential 4 KB stride sweep across 16 MB chunks -> Fast single linear read
```

* **Function:** [`lfm2_warmup_mmap_pages`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c#L230).
* **Mechanism:** Reads 1 byte per 4,096 bytes under `MADV_WILLNEED` and `POSIX_FADV_WILLNEED`.
* **Telemetry:** Emits a live `MB/s` console progress bar to `stderr`.

