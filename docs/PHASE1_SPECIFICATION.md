# Phase 1 Specification: Reference C Engine & Mathematical Parity

## 1. Objectives

Phase 1 focuses on architectural completeness, algorithmic correctness, and zero memory leaks. It establishes a dependable reference implementation against which future SIMD and quantized kernels (Phase 2) are benchmarked and verified.

---

## 2. Component Specifications

### 2.1 Zero-Copy GGUF Loader (`include/gguf.h`, `src/gguf.c`)
- **File Format:** GGUF v2 and v3 (little-endian).
- **Header:**
  - `magic`: `0x46554747` (`GGUF`).
  - `version`: uint32 (2 or 3).
  - `n_tensors`: uint64.
  - `n_kv`: uint64.
- **Metadata Parser:**
  - Dynamic string decoding (uint64 length + UTF-8 string bytes).
  - Type-safe value decoding: `GGUF_TYPE_UINT32`, `GGUF_TYPE_INT32`, `GGUF_TYPE_FLOAT32`, `GGUF_TYPE_STRING`, `GGUF_TYPE_ARRAY`, etc.
  - Extracts model parameters:
    - `general.architecture` (e.g. `"lfm2"` or `"lfm"`)
    - `lfm2.context_length` / `general.context_length`
    - `lfm2.embedding_length` ($D$)
    - `lfm2.feed_forward_length` ($D_{\text{ffn}}$)
    - `lfm2.block_count` ($N_{\text{layers}}$)
    - `lfm2.attention.head_count` ($H_q$)
    - `lfm2.attention.head_count_kv` ($H_{kv}$)
    - `lfm2.conv_L_cache` (default $3$)
    - `lfm2.layer_types` (array of strings: `"conv"` vs `"full_attention"`)
- **Tensor Directory & Memory Mapping:**
  - Scans tensor descriptors: name, number of dimensions, dimension array, type, and byte offset relative to the data segment.
  - Computes 32-byte alignment padding.
  - Maps the entire binary into virtual memory with `mmap(PROT_READ, MAP_SHARED)`.
  - Informs the Linux kernel via `madvise(MADV_SEQUENTIAL)` for optimal read-ahead.

---

### 2.2 Memory Arena Allocator (`include/arena.h`, `src/arena.c`)
To guarantee zero heap allocations (`malloc`/`free`) during inference:
- A linear memory arena is allocated during engine initialization.
- Every allocated buffer is 64-byte aligned (matching x86_64 and ARM cache lines).
- Allocations are made via a simple bump allocator:
  ```c
  void* arena_alloc(Arena* arena, size_t size);
  void arena_reset(Arena* arena);
  ```
- Scratch memory for single-token forward pass (projections, intermediate activations, logits) uses an ephemeral arena reset after each token.

---

### 2.3 Layer Primitives & Forward Pass (`include/lfm_forward.h`, `src/lfm_forward.c`)
1. **`lfm_rmsnorm`:**
   Computes root-mean-square normalization over hidden dimension $D$ with numerical epsilon $\epsilon$.
2. **`lfm_qk_norm`:**
   Applies RMSNorm per head on query and key vectors before positional embedding.
3. **`lfm_rope`:**
   Applies rotary positional embeddings to queries and keys:
   $$x_{\text{out}}[2i] = x[2i] \cos(\theta) - x[2i+1] \sin(\theta)$$
   $$x_{\text{out}}[2i+1] = x[2i] \sin(\theta) + x[2i+1] \cos(\theta)$$
4. **`lfm_liv_shortconv_step`:**
   - Input projection: $W_{\text{in}} \cdot x_{\text{norm}} \rightarrow [3D]$
   - Split into $B, C, x \in \mathbb{R}^D$
   - Gate 1: $u = B \odot x$
   - 1D Depthwise Conv ($K=3$):
     $$\text{out}[c] = w[0][c] \cdot s_0[c] + w[1][c] \cdot s_1[c] + w[2][c] \cdot u[c] + b[c]$$
   - State update: $s_0 \leftarrow s_1$, $s_1 \leftarrow u$
   - Gate 2: $y = C \odot \text{out}$
   - Output projection: $W_{\text{out}} \cdot y$
5. **`lfm_liv_shortconv_prefill`:**
   Causal 1D convolution over sequence length $T$, propagating the state through the sequence and saving the final 2-token state into the layer ring buffer.
6. **`lfm_gqa_step`:**
   Grouped query attention across $H_q$ query heads and $H_{kv}$ key/value heads, updating the KV-cache at position $pos$.
7. **`lfm_swiglu_step`:**
   Feed-forward block:
   $$W_2 \cdot \left(\text{SiLU}(W_1 x) \odot W_3 x\right)$$
   Where $\text{SiLU}(z) = z \cdot \frac{1}{1 + e^{-z}}$.

---

### 2.4 Tokenizer & Sampling (`include/lfm_tokenizer.h`, `src/lfm_tokenizer.c`)
- Loads vocabulary strings and merges from GGUF metadata.
- Supports byte-level BPE encoding with greedy longest-match merge lookup.
- Fast token-to-string decoding with UTF-8 byte stream assembly.
- Sampling features:
  - Greedy decoding (Argmax).
  - Temperature scaling ($z_i \leftarrow z_i / T$).
  - Top-K filtering.

---

### 2.5 Validation Strategy
- Python script `tools/generate_reference_tensors.py` uses PyTorch to compute exact reference outputs for:
  1. RMSNorm and QK-Norm.
  2. RoPE rotary transformation.
  3. Depthwise Conv1d and double-gated modulation.
  4. SwiGLU activation.
  5. Multi-layer hybrid forward pass.
- C test runners in `tests/` execute the C functions on identical input vectors and assert element-wise parity with tolerance $< 10^{-4}$.
