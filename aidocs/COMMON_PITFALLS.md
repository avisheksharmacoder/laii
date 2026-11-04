# Common Pitfalls, Fatal Anti-Patterns & Troubleshooting

This document catalogs subtle traps, alignment bugs, and architectural misconceptions that frequently trip up AI coding agents in this codebase.

---

## ⚠️ Pitfall 1: `aligned_alloc` Size Not an Integral Multiple of Alignment

### The Trap:
In C11, `aligned_alloc(alignment, size)` requires that **`size` must be an integral multiple of `alignment`**. While some glibc versions silently allocate without checking, **AddressSanitizer strictly enforces this standard**:
```text
==ERROR: AddressSanitizer: invalid alignment requested in aligned_alloc: 64,
alignment must be a power of two and the requested size 0x490 must be a multiple of alignment
```

### The Fix:
Never call raw `aligned_alloc(64, ...)` with unpadded byte counts! Always round `size` up to the nearest multiple of 64 using `lfm2_aligned_alloc_64`:
```c
static inline void *lfm2_aligned_alloc_64(size_t size) {
    size_t aligned_size = (size + 63) & ~((size_t)63);
    return aligned_alloc(64, aligned_size);
}
```

---

## ⚠️ Pitfall 2: Confusing `dims[4]` vs `ne[4]` Tensor Dimension Accessors

### The Trap:
* The core engine ([`include/gguf.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/gguf.h)) names tensor dimension arrays `dims[4]`.
* Standalone tools ([`tools/lfm2_loader.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_loader.c)) name them `ne[4]`.
Accessing `tensor->ne[0]` in code compiled against `include/gguf.h` results in a compilation error: `‘GGUFTensor’ has no member named ‘ne’`.

### The Fix:
Check the environment with conditional compilation:
```c
#if defined(LFM_GGUF_H) || defined(LFM_MODEL_H)
    const uint64_t d0 = tensor->dims[0];
#else
    const uint64_t d0 = tensor->ne[0];
#endif
```

---

## ⚠️ Pitfall 3: Type Collisions between `include/gguf.h` and `tools/lfm2_loader.c`

### The Trap:
If you `#include "tools/lfm2_loader.c"` inside a file in `src/` (or any file that already includes `include/gguf.h`), the compiler throws duplicate enum errors (`redeclaration of enumerator ‘GGML_TYPE_F32’`) and conflicting struct declarations.

### The Fix:
Guard the inclusion of `lfm2_loader.c` so it is omitted when compiling inside the core library environment:
```c
#if !defined(LFM_GGUF_H) && !defined(LFM_MODEL_H)
#define main lfm2_loader_demo_main
#include "tools/lfm2_loader.c"
#undef main
#endif
```

---

## ⚠️ Pitfall 4: Treating ShortConv Layers as if They Have a KV-Cache

### The Trap:
Assuming that all layers have Key-Value cache lines (like standard Transformers). An agent might attempt to:
* Allocate KV memory for layer $l$ when `block_type == LFM2_BLOCK_SHORTCONV`.
* Shift or counter-rotate ShortConv memory during attention-sink context eviction.
* Rewind the KV sequence pointer `pos` during speculative decoding and expect ShortConv states to magically rewind.

### The Invariant:
* In ShortConv layers, `n_head_kv[l] == 0`. There is **zero KV-cache**.
* ShortConv state is an $O(1)$ recurrent rolling buffer ($s_0, s_1$) representing step $t-2$ and $t-1$.
* When context shifts, ShortConv requires **0 memory movement or rotation**.
* When rewinding or rolling back, ShortConv requires restoring state buffers from an [`LFM2Checkpoint`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c#L350).

---

## ⚠️ Pitfall 5: Partial Prefix Loading from a Prompt Cache (`.lfmcache`)

### The Trap:
Assuming you can load the first 300 tokens from a 500-token prompt cache.
* **Why it fails:** At step 500, the ShortConv state stored in the file represents the recurrent history of token 498 and 499. If you attempt to resume generation from token 300, the ShortConv layers will execute with token 500's history, producing gibberish.

### The Invariant:
* Prompt cache loading requires an **exact 100% full prefix match** of all $N_{\text{cached}}$ tokens stored in the `.lfmcache` file before restoring states and advancing `pos`.

---

## ⚠️ Pitfall 6: Dynamic `malloc` Inside Autoregressive Generation Loops

### The Trap:
Allocating temporary scratch vectors via `malloc` inside `lfm_forward()` or attention loops.
* **Impact:** Causes memory fragmentation, latency spikes, and violates the zero-allocation engine invariant.

### The Fix:
Always allocate scratch memory from the preallocated `LFMArena` or pass preallocated buffers from the engine struct.

---

## ⚠️ Pitfall 7: Missing `-I.` During Test Compilation

### The Trap:
Compiling test files that include tools (e.g. `#include "tools/lfm2_sys_cache.c"`) without passing `-I.` to GCC.
* **Error:** `fatal error: tools/lfm2_sys_cache.c: No such file or directory`.

### The Fix:
Always include `-I.` in the compilation command:
```makefile
bin/test_foo: tests/test_foo.c | dirs
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)
```

---

## ⚠️ Pitfall 8: Stale Object Files after Header Modifications

### The Trap:
Modifying a struct in `include/lfm_engine.h` (e.g. adding a new field to `LFMSamplerConfig`) and running `make` without updating object dependencies.
* **Impact:** `build/lfm_c_api.o` is not rebuilt, resulting in an ABI struct offset mismatch and an immediate segmentation fault during string/pointer access in tests.

### The Fix:
Ensure `Makefile` depends on headers:
```makefile
build/%.o: src/%.c $(wildcard include/*.h) | dirs
```
When in doubt, run `make clean && make all`.

