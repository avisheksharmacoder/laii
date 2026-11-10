# Liquid Foundation Model (LFM2) Engine Architecture

## 1. Overview & Motivation

Modern Large Language Models (LLMs) based entirely on pure Transformer architectures suffer from two fundamental bottlenecks when deployed on resource-constrained CPUs without dedicated GPUs:

1. **Quadratic Complexity & Expanding KV-Cache:** Full self-attention across every layer requires maintaining Key and Value states for every generated token ($O(T)$ memory per layer, $O(T^2)$ total compute). As generation length grows, the KV cache inflates, causing memory fragmentation and cache thrashing.
2. **Memory Bandwidth Bottleneck:** In single-batch autoregressive token generation on CPUs, execution is completely memory-bandwidth bound. Every token generated requires pulling model weights from system DRAM across the memory bus into CPU cache and vector registers.

Liquid AI's **LFM2 / LFM2.5** family resolves these bottlenecks by replacing the majority of self-attention blocks with **Linear Input-Varying (LIV) Short Convolutions**, reserving standard Grouped Query Attention (GQA) for a small subset of layers.

---

## 2. Hybrid Layer Breakdown

For example, in **LFM2-1.2B** (16 layers total):
* **10 Layers are Double-Gated LIV Short Convolutions:** Local context token mixing with $O(1)$ recurrent memory and $O(1)$ generation compute per token.
* **6 Layers are Grouped Query Attention (GQA):** Sparse long-range dependency capture with RoPE and QK-Normalization.
* **All Layers use SwiGLU & RMSNorm:** Standard feed-forward networks with SiLU activation and root-mean-square normalization.

```
+---------------------------------------------------------------+
|                      LFM2 Backbone Layer                      |
+---------------------------------------------------------------+
                               |
                               v
                       [ Operator RMSNorm ]
                               |
        +----------------------+-----------------------+
        |                                              |
        v (Layer is Conv)                              v (Layer is Attention)
+-------------------------------+             +-------------------------------+
|     LIV ShortConv Mixer       |             |         GQA Mixer             |
|                               |             |                               |
| 1. in_proj -> [B, C, x]       |             | 1. q_proj, k_proj, v_proj     |
| 2. Gate 1: u = B * x          |             | 2. QK-RMSNorm (q_norm, k_norm)|
| 3. Depthwise Conv1D (k=3)     |             | 3. Rotary Positional Emb(RoPE)|
| 4. Gate 2: y = C * ConvOut    |             | 4. Grouped Query Softmax Attn |
| 5. out_proj -> [hidden_dim]   |             | 5. out_proj -> [hidden_dim]   |
+-------------------------------+             +-------------------------------+
        |                                              |
        +----------------------+-----------------------+
                               |
                               v
                         [ + Residual ]
                               |
                               v
                         [ FFN RMSNorm ]
                               |
                               v
                      [ SwiGLU FFN Block ]
                w2( silu(w1(x)) * w3(x) )
                               |
                               v
                         [ + Residual ]
                               |
                               v
                        To Next Layer
```

---

## 3. Mathematical Formulations

### 3.1 Normalization (Lfm2RMSNorm)
Given an input vector $x \in \mathbb{R}^D$:
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$$
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
Where $\gamma \in \mathbb{R}^D$ is the learnable weight vector and $\epsilon$ is typically $10^{-5}$.

---

### 3.2 LIV Short-Range Gated Convolution (Lfm2ShortConv)
The LIV short convolution block provides dynamic, input-varying linear token mixing with causal local receptive field (kernel size $K$, default $K=3$).

#### Projections and Splitting:
Given normalized input $h \in \mathbb{R}^D$:
$$\begin{pmatrix} B \\ C \\ x \end{pmatrix} = W_{\text{in}} h \quad \text{where } W_{\text{in}} \in \mathbb{R}^{3D \times D}$$
Here $B, C, x \in \mathbb{R}^D$.

#### Gate 1 (Input-Varying Modulation):
$$u = B \odot x$$
Where $\odot$ represents element-wise multiplication.

#### Depthwise 1D Causal Convolution:
During autoregressive generation at step $t$ with $K=3$, each channel $c \in [0, D-1]$ computes:
$$\text{ConvOut}[c] = w[0][c] \cdot s_{t-2}[c] + w[1][c] \cdot s_{t-1}[c] + w[2][c] \cdot u[c] + b_{\text{conv}}[c]$$
Where $s_{t-2}, s_{t-1}$ are the states from the preceding two tokens stored in a ring buffer, and $w \in \mathbb{R}^{D \times K}$ are depthwise convolution weights.

The ring buffer is then updated:
$$s_{t-2}[c] \leftarrow s_{t-1}[c], \quad s_{t-1}[c] \leftarrow u[c]$$

#### Gate 2 & Output Projection:
$$y = C \odot \text{ConvOut}$$
$$\text{Output} = W_{\text{out}} y \quad \text{where } W_{\text{out}} \in \mathbb{R}^{D \times D}$$

---

### 3.3 Grouped Query Attention (Lfm2Attention)
For the designated attention layers:

1. **Linear Projections:**
   $$Q = W_q h, \quad K = W_k h, \quad V = W_v h$$
   Where $Q \in \mathbb{R}^{H_q \times d_k}$, $K \in \mathbb{R}^{H_{kv} \times d_k}$, $V \in \mathbb{R}^{H_{kv} \times d_k}$, with $H_q / H_{kv} = \text{group ratio}$.
2. **QK-Normalization:**
   $$Q = \text{RMSNorm}(Q), \quad K = \text{RMSNorm}(K)$$
   LFM2 applies RMSNorm per head *before* rotary embedding to stabilize cross-layer activations.
3. **Rotary Positional Embedding (RoPE):**
   $$Q_{\text{rot}} = \text{RoPE}(Q, pos), \quad K_{\text{rot}} = \text{RoPE}(K, pos)$$
4. **Attention Computation:**
   $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
5. **Output Projection:**
   $$\text{Output} = W_o \cdot \text{Attn}(Q, K, V)$$

---

### 3.4 Feed-Forward Network (SwiGLU)
$$h_{\text{ffn}} = \text{RMSNorm}(x)$$
$$\text{FFN}(h_{\text{ffn}}) = W_2 \cdot \left( \text{SiLU}(W_1 h_{\text{ffn}}) \odot W_3 h_{\text{ffn}} \right)$$
Where $W_1 \in \mathbb{R}^{D_{\text{ffn}} \times D}$, $W_3 \in \mathbb{R}^{D_{\text{ffn}} \times D}$, and $W_2 \in \mathbb{R}^{D \times D_{\text{ffn}}}$.

---

## 4. Memory Footprint Analysis: LFM2 vs. Standard Transformer

Assume a 16-layer model with $D = 2048$, generating a sequence of length $L = 4096$ tokens in single-batch mode:

| Memory Component | Standard Transformer (16 Attention Layers) | LFM2 (10 LIV Conv Layers + 6 GQA Layers) | Reduction |
| :--- | :--- | :--- | :--- |
| **Attention KV-Cache** | $16 \times 2 \times 4096 \times 2048 \times 2\text{ bytes} \approx \mathbf{536.8\text{ MB}}$ | $6 \times 2 \times 4096 \times 512 \times 2\text{ bytes} \approx \mathbf{50.3\text{ MB}}$ | **~90.6% savings** |
| **Recurrent Conv State** | $0\text{ bytes}$ | $10 \times 2 \times 2048 \times 4\text{ bytes} \approx \mathbf{163.8\text{ KB}}$ (constant!) | Fixed cache footprint |
| **Total Dynamic State** | $\mathbf{\approx 536.8\text{ MB}}$ | $\mathbf{\approx 50.5\text{ MB}}$ | **~90.6% reduction** |

### CPU Cache Benefit
The entire recurrent state for all 10 convolution layers occupies only $\approx 164\text{ KB}$. This easily fits in the L2 cache (typically 512 KB to 1 MB per core) of modern x86 and ARM processors, resulting in near-zero memory bus traffic for short convolution state updates.

> [!NOTE]
> For real-world empirical benchmarks comparing LFM2.c against `llama.cpp` on `LFM2.5-1.2B` and `LFM2.5-230M` (including memory RSS reductions, cold load times, and roofline analysis), see [docs/BENCHMARKS.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/BENCHMARKS.md).

---

## 5. Real-World Liquid AI (LFM2/LFM2.5) GGUF Quirks & Loader Architecture

Generic Llama/Mistral GGUF loaders fail when reading production `LiquidAI/LFM2.5-*-GGUF` checkpoints due to four specific architectural deviations:

1. **Per-Layer Hybrid Routing via Array Metadata:**
   - Standard GGUF treats `attention.head_count_kv` as a single scalar.
   - LFM2 checkpoints store `lfm2.attention.head_count_kv` as an array of `uint32` across all layers (e.g. `[0, 0, 8, 0, 0, 8, ...]`).
   - Whenever `head_count_kv[l] == 0`, layer $l$ is an input-varying ShortConv (LIV) block. When `head_count_kv[l] > 0`, layer $l$ is a GQA attention block.
2. **ShortConv Tensor Naming:**
   - Short convolution blocks use:
     - `blk.%d.shortconv.in_proj.weight` ($[n_{\text{embd}}, 3 \times n_{\text{embd}}]$)
     - `blk.%d.shortconv.conv.weight` ($[l_{\text{cache}}, n_{\text{embd}}]$ where $l_{\text{cache}} = 3$)
     - `blk.%d.shortconv.out_proj.weight` ($[n_{\text{embd}}, n_{\text{embd}}]$)
3. **Per-Head QK-Norm in GQA Layers:**
   - In GQA blocks, LFM2 applies RMSNorm per head:
     - `blk.%d.attn_q_norm.weight` ($[\text{head\_dim}]$, typically 64 floats)
     - `blk.%d.attn_k_norm.weight` ($[\text{head\_dim}]$, typically 64 floats)
4. **Final Norm Naming & Tied Output Weights:**
   - The final output RMSNorm tensor is named **`token_embd_norm.weight`** instead of `output_norm.weight`.
   - The language model head `output.weight` is often omitted because output logits share weights directly with `token_embd.weight`.

### Zero-Copy Pointer Binding
Both the core engine (`src/lfm_model.c`) and the standalone inspector (`bin/lfm2_loader`, source `tools/lfm2_loader.c`) map the `.gguf` file using Linux `mmap` with `madvise(..., MADV_SEQUENTIAL)` and bind tensor pointers directly to 32-byte/64-byte aligned offsets in the kernel page cache without heap copying.

---

## 6. Forward Pass Engine & State Arena (`lfm2_forward.c` / `src/lfm_forward.c`)

The forward pass executes token-by-token autoregressive decoding and prompt prefilling across the hybrid layer stack:

1. **Embedding Lookup & RMS Normalization:**
   - Look up token embedding from `token_embd.weight`.
   - Apply per-layer pre-operator RMSNorm.
2. **ShortConv LIV Execution:**
   - Single fused projection into `[3 * n_embd]` to produce $B, C, \tilde{x}$.
   - Pre-conv modulation: $u = B \odot \tilde{x}$.
   - Depthwise 1D causal convolution using a 2-token rolling buffer:
     $$\text{conv\_out}[c] = w[0][c] \cdot s_{0}[c] + w[1][c] \cdot s_{1}[c] + w[2][c] \cdot u[c] + b[c]$$
   - Rolling buffer shift: $s_0 \leftarrow s_1, s_1 \leftarrow u$.
   - Post-conv gating: $y = C \odot \text{conv\_out}$, followed by output linear projection and residual addition.
3. **GQA Attention Execution:**
   - Project hidden state into $Q, K, V$.
   - Apply per-head RMSNorm to $Q$ and $K$.
   - Apply Rotary Positional Embeddings (RoPE) using half-split rotation:
     $$\text{RoPE}(x_i, x_{i+d/2}) = (x_i \cos \theta - x_{i+d/2} \sin \theta, \; x_i \sin \theta + x_{i+d/2} \cos \theta)$$
   - Cache $K$ and $V$ in the persistent KV-cache arena at current sequence position `pos`.
   - Compute causal scaled dot-product attention over cached $K, V$ positions $0 \le t \le pos$.
   - Project attention output and add residual.
4. **SwiGLU FFN:**
   - Apply FFN RMSNorm.
   - Compute gate and up-projections: $\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$.
   - Apply down-projection $W_{\text{down}}$ and add residual.
5. **Output Logits:**
   - Final RMSNorm via `token_embd_norm.weight`.
   - Compute dot product with `output.weight` (or tied `token_embd.weight`) across the 65,536 vocabulary.

---

## 7. Interactive Terminal Chat Engine & Sampler (`tools/lfm2_chat.c`)

Building an interactive terminal chat engine for **LFM2 / LFM2.5** requires matching two specifications from Liquid AI’s post-training pipeline:

### 7.1 Liquid AI Recommended Sampling Pipeline & Min-P Cutoff
LFM2/LFM2.5 checkpoints are tuned for low-temperature **Min-P** sampling:
* `temperature = 0.30`
* `min_p = 0.15`
* `top_k = 50`
* `repetition_penalty = 1.05` (sliding window of 64 tokens)

#### Logit-Space Min-P Pre-Filtering:
In LFM2/LFM2.5, vocabulary size is 65,536. Running `expf()` and `qsort()` on 65,536 tokens per step is wasteful. We use a single-pass logit cutoff:

$$\frac{\exp(z_i / T)}{\exp(z_{\max} / T)} \ge p_{\min} \iff z_i \ge z_{\max} + T \ln(p_{\min})$$

Computing $z_{\text{cutoff}} = z_{\max} + T \ln(p_{\min})$ filters 65,536 candidates down to **5 to 25 surviving candidates** in a single linear pass before any `expf()` or `qsort()` call is executed.

### 7.2 Stateful Multi-Turn ChatML Formatting
LFM2 uses `<|startoftext|>`, `<|im_start|>`, and `<|im_end|>` tags:
* **Turn 1 (First turn):** Emits `<|startoftext|><|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`.
* **Turn 2+ (Multi-turn continuation):** Emits only the incremental delta `\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`.

Because `LFM2State` holds both the 6 GQA KV caches and the 10 ShortConv 2-token rolling buffers, multi-turn chat **never** re-prefills the conversation from scratch on subsequent turns; it simply appends the new delta at the current `pos` cursor.

### 7.3 Zero-Dependency GGUF Vocabulary Loader & Tokenizer
`LFM2Vocab` extracts `tokenizer.ggml.tokens` directly from the GGUF header and implements GPT-2 byte-level unicode mapping ($0\text{x}20 \rightarrow \text{Ġ}$, $0\text{x}0\text{A} \rightarrow \text{Ċ}$) for terminal streaming and interactive decoding.

---

## 8. High-Throughput AVX2 Q8_K Activation Pre-Quantization & Integer SIMD Dot Products (`include/lfm2_simd_avx2.h`)

To eliminate the massive overhead of scalar on-the-fly dequantization when running official `Q4_K` and `Q6_K` checkpoints (e.g. `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`), the engine implements single-pass **`Q8_K` activation pre-quantization**:

1. **Amortized Quantization Cost ($O(D)$):**
   - For any matrix-vector multiplication $y = W x$, the activation vector $x \in \mathbb{R}^D$ is converted into 256-element `block_q8_K_t` super-blocks **once** before entering the parallel row loop.
   - Computes symmetric scale $d = \max(|x_i|) / 127.0$, int8 quantized values $q_8 \in [-127, 127]$, and precomputed 16-element sub-block activation sums `bsums[16]`.
2. **32 Multiply-Accumulates per Cycle via `_mm256_maddubs_epi16`:**
   - Multiplies 32 unsigned weight bytes ($u_4 \in [0, 15]$ or $u_6 \in [0, 63]$) against 32 signed $q_8$ activations in a single instruction with zero 16-bit lane saturation ($15 \times 127 \times 2 = 3810 \le 32767$, $63 \times 127 \times 2 = 16002 \le 32767$).
3. **Zero-Point Cancellation via `bsums[16]`:**
   - Both asymmetric `Q4_K` min offsets and symmetric `Q6_K` $-32$ offsets are computed directly against precomputed `bsums[16]`, completely removing runtime activation summation from the inner row loop.
4. **1-Instruction Sub-Block Scaling and Widening via `_mm256_madd_epi16`:**
   - Multiplies 16-bit intermediate dot products by sub-block scales and widens to 32-bit accumulators in a single instruction.

---

## 9. Exact GGUF BPE Rank Merging Tokenizer Engine (`tools/lfm2_bpe.c`)

Standard greedy longest-prefix matching tokenizers fail to reproduce correct tokenization on modern BPE vocabularies (e.g. GPT-2, LLaMA-3, Liquid LFM2) because BPE merges are priority-ordered rather than prefix-length-greedy. 

The inference engine upgrades tokenization to **exact priority-ordered GGUF BPE rank merging** with zero runtime heap allocations:

1. **Load-Time String Hash Table (`TokenStrSlot`):**
   - Maps all 65,536 token strings in `tokenizer.ggml.tokens` to their `uint32_t` token IDs in $O(1)$ via 64-bit FNV-1a hashing in an open-addressed power-of-two table with $\le 50\%$ load factor.
2. **Load-Time `byte_to_id[256]` Lookup Table:**
   - Every raw byte `0x00..0xFF` is mapped to its GPT-2 byte-level Unicode character (e.g. `0x20` $\rightarrow$ `Ġ`, `0x0A` $\rightarrow$ `Ċ`) and pre-resolved to its initial token ID. Runtime word seeding is a single array lookup `v->byte_to_id[byte]`.
3. **Load-Time Merge Pair Hash Table (`BPEMergeSlot`):**
   - Each `"left right"` string in `tokenizer.ggml.merges` at index `rank` is resolved into `left_id`, `right_id`, and `merged_id`, and keyed by packed 64-bit integer `((uint64_t)left_id << 32) | right_id`.
4. **Runtime Doubly-Linked List + Min-Heap ($O(N \log N)$):**
   - Pre-tokenized words are loaded into a stack-allocated doubly-linked list (`BPESymbol syms[512]`).
   - Merge candidates are enqueued into a binary min-heap ordered by `(rank, left_pos)`.
   - Lazy version counters (`ver`) invalidate stale heap entries in $O(1)$ without heap scanning or deletion.
5. **GPT-2 Pre-Tokenizer State Machine & Special Token Pre-Splitting:**
   - Pre-tokenizes contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), runs of up to 3 digits, letters, punctuation, and whitespace sequences.
   - Preserves special tags (`<|startoftext|>`, `<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`) with exact discrete boundary preservation.
   - Seamlessly falls back to prefix matching when loaded on legacy checkpoints lacking `tokenizer.ggml.merges` (e.g. `tests/lfm2_toy.gguf`).

---

## 10. C11 Lock-Free Spin-Barrier Thread Pool & Linux Futex Engine (`tools/lfm2_threadpool.c`)

To eliminate the external shared runtime dependency on OpenMP (`libgomp.so.1`) and enable pure `-static` Linux ELF builds, the inference engine implements a custom **C11 `<stdatomic.h>` + `pthread` + Linux `SYS_futex` spin-barrier thread pool** tailored specifically for low-latency autoregressive token decoding:

### 10.1 Physical Core Topology Pinning (No SMT / Hyperthreading Thrashing)
On modern x86_64 CPUs, co-scheduling heavy 256-bit SIMD matrix multiplication loops across SMT siblings (hyperthreads sharing a physical execution core) induces severe execution port contention (Port 0/1/5 FMA/ALU) and causes L1 data cache eviction.
- The pool queries `/sys/devices/system/cpu/cpuN/topology/core_id` and `physical_package_id` to discover unique physical cores.
- Each worker thread is pinned to its dedicated physical core via `sched_setaffinity`.
- Worker `0` runs directly on the main calling thread, avoiding extra OS context switches.

### 10.2 Cache-Line Aligned Atomic Spin-Barrier Layout
The `LFM2ThreadPool` struct is split across 64-byte cache lines (`_Alignas(64)`) to completely prevent false sharing between producer and consumer threads:
- **Cache-Line 1 (Producer Dispatch State):** `generation`, `stop`, `task_fn`, `task_arg`, `n_items`, `chunk_size`.
- **Cache-Line 2 (Dynamic Work-Stealing Cursor):** `next_item` (shared atomic `fetch_add`).
- **Cache-Line 3 (Completion Barrier Counter):** `n_pending` and `n_sleeping` futex wait trackers.

### 10.3 Hybrid User-Space Spin + `FUTEX_WAIT_PRIVATE` Sleep
Across the ~80 matrix multiplications in a single token forward pass, background workers spin in user-space with `_mm_pause()` for zero-syscall, sub-microsecond latency (measured at **~260 ns** round-trip dispatch latency).
- When token generation concludes or the CLI/HTTP server waits for user input, spinning workers exceed `LFM2_SPIN_ITERS` (25,000 iterations $\approx 10$ ms) and transition to zero-CPU kernel sleep via Linux `futex_wait(&pool->generation, local_gen)`.
- On the next token dispatch, the main thread wakes sleeping workers via `futex_wake(&pool->generation, n_workers)`.

### 10.4 Drop-In OpenMP-Free Parallel MatVec & GQA Attention
All `#pragma omp parallel for` pragmas across `lfm2_matmul_vec_avx2_mt` and `lfm2_forward_gqa` are replaced with `lfm2_parallel_for`:
- **MatVec:** 16-row chunks claimed dynamically via atomic `fetch_add` to eliminate thread imbalance across asymmetric matrix shapes.
- **GQA Heads:** 1-head chunks claimed dynamically across physical cores with zero heap allocations.
- **Pure Static Linux Binaries:** Compiles with `-static -pthread` into self-contained ELF executables with zero dynamic library dependencies (`ldd: not a dynamic executable`).

---

## 11. Batched Prefill Acceleration & Rank-1 Accumulator GEMM (`tools/lfm2_prefill_batch.c`)

During prompt prefill, token sequences are known in advance. Processing $B$ tokens sequentially through matrix-vector products requires reloading multi-megabyte weight tensors $B$ times from system memory, saturating CPU memory bandwidth.

### 11.1 Rank-1 Outer-Product Accumulators
`lfm2_prefill_batch.c` replaces single-token MatVec with token-block GEMM:
- **Parallel Activation Quantization:** A batch of $B$ continuous activation vectors $X \in \mathbb{R}^{B \times D}$ is pre-quantized in parallel into $B$ arrays of `block_q8_K` structs.
- **Weight Amortization:** For every row of weight matrix $W$, the weight row is loaded once into CPU vector registers and multiplied across all $B$ tokens simultaneously:
  $$Y[0..B-1, r] = \sum_{c} W[r, c] \cdot X[0..B-1, c]$$
- **Memory Bandwidth Reduction:** DRAM weight traffic drops by a factor of $B$, delivering 2.5× to 4× prefill speedups while maintaining 100% bit-exact equivalence with single-token decode.

---

## 12. 1x4 Row-Unrolled AVX2 MatVec & Register-Resident SwiGLU FFN (`tools/lfm2_decode_1x4.c`)

### 12.1 1x4 Row-Unrolled AVX2 MatVec
In single-token decoding, reading activation vector $x$ for every single row incurs repeated L1/L2 cache accesses.
- The 1x4 unrolled kernel computes 4 adjacent output rows ($r, r+1, r+2, r+3$) concurrently.
- Each 256-element block of activation $x$ is loaded once into a YMM register and reused across all 4 weight rows.
- Integer SIMD instructions (`_mm256_maddubs_epi16`, `_mm256_madd_epi16`) accumulate into four parallel 32-bit vector accumulators, hiding execution latency and maximizing port utilization.

### 12.2 Register-Resident SwiGLU FFN Fusion
In standard implementations, SwiGLU requires computing $w_1(x)$, writing it to an intermediate buffer in RAM, computing $w_3(x)$, writing to another buffer, evaluating $\text{SiLU}$, multiplying, and then reading back for $w_2$.
- The fused kernel computes both $w_1$ (gate) and $w_3$ (up) in vector registers.
- Evaluates the rational polynomial approximation of $\text{SiLU}$ directly on the $w_1$ register, multiplies by $w_3$, and leaves the activation register-resident for immediate quantizing or accumulation, cutting intermediate DRAM memory transfers by $2\times$.

---

## 13. Planar ShortConv, Precomputed RoPE LUT & Single-Pass FP16 Online Softmax (`tools/lfm2_kernels_fast.c`)

### 13.1 Planar AVX2 ShortConv 3-Tap Kernel
GGUF stores `blk.l.shortconv.conv.weight` in stride-3 layout (`[w_{d,0}, w_{d,1}, w_{d,2}]`), causing non-contiguous scalar loads during vector execution.
- **Model Loading:** The kernel transposes stride-3 weights into three 64-byte aligned planar arrays ($w_0[D]$, $w_1[D]$, $w_2[D]$).
- **Execution Loop:** An 8-wide AVX2 loop evaluates Gate 1 ($u_t = B \odot \tilde{x}$), reads 2-token rolling states $(s_0, s_1)$, computes $w_0 s_0 + w_1 s_1 + w_2 u_t$, updates rolling history in-place, evaluates Gate 2 ($y = C \odot \text{conv}$), and quantizes the result to `conv_out_q8` in L1 cache.

### 13.2 Precomputed RoPE LUT & Fused Per-Head QK-Norm
- In LFM2, `head_dim = 64` (`half_dim = 32`), which fits into eight `__m256` registers (`x_lo[0..3]` and `x_hi[0..3]`).
- Startup function `lfm2_rope_lut_init` precomputes `cos_table` and `sin_table` for all positions up to `max_seq_len`, eliminating per-token `powf`, `cosf`, and `sinf` overhead.
- Normalizes query and key heads in-place, applies half-split NeoX RoPE with `_mm256_fmsub_ps` and `_mm256_fmadd_ps`, and packs Keys/Values directly into the KV-cache as IEEE-754 `FP16` via hardware `_mm256_cvtps_ph`.

### 13.3 Single-Pass Online Softmax (FlashAttention-CPU) over FP16 KV-Cache
Standard attention requires three memory sweeps (dot products $\rightarrow$ max/exp $\rightarrow$ value accumulation) and an intermediate score buffer.
- FlashAttention-CPU keeps running accumulators in eight `__m256` registers alongside scalars $m_{\text{prev}}$ (running max logit) and $l_{\text{prev}}$ (running partition sum).
- Streams Keys and Values from `k_cache_f16` and `v_cache_f16` in a single pass using hardware `_mm256_cvtph_ps`:
  $$\text{If } s_t > m_{\text{prev}}: \quad \alpha = \exp(m_{\text{prev}} - s_t), \quad l \leftarrow l \cdot \alpha + 1, \quad \mathbf{o} \leftarrow \mathbf{o} \cdot \alpha + \mathbf{v}_t, \quad m_{\text{prev}} \leftarrow s_t$$
  $$\text{Else}: \quad \beta = \exp(s_t - m_{\text{prev}}), \quad l \leftarrow l + \beta, \quad \mathbf{o} \leftarrow \mathbf{o} + \beta \cdot \mathbf{v}_t$$
- Slashes KV memory traffic by 50% and eliminates the intermediate scratch buffer entirely.

---

## 14. Single-Binary CPUID Multi-ISA Dynamic Dispatch & Extended Quantization (`tools/lfm2_cpuid_quants.c`)

Compiling with `-march=native` introduces AVX2 instructions into the ELF entry point, leading to fatal `SIGILL` crashes on older x86_64 CPUs.

### 14.1 GCC/Clang Target Attributes & CPUID Query
- Baseline functions are built for generic `x86-64`.
- Accelerated SIMD kernels are decorated with `__attribute__((target("avx2,fma,f16c")))` and `__attribute__((target("avx2,fma,f16c,avxvnni")))`.
- At startup, [`lfm2_init_cpu_kernels()`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_cpuid_quants.c) inspects CPU capabilities via `__cpuid_count(7, 0, ...)` and `__builtin_cpu_supports`:
  - **AVX-VNNI Tier:** Selects `"AVX-VNNI (vpdpbusd) + FMA + F16C + NTA"` on modern CPUs (Intel 12th-15th Gen Alder/Raptor/Arrow Lake, AMD Zen 4/Zen 5).
  - **AVX2 Tier:** Selects `"AVX2 + FMA + F16C + NTA"` on baseline AVX2 systems (Intel Haswell through 11th Gen, AMD Zen 1-3).
  - **Scalar Fallback:** If AVX2 is absent or `LFM2_FORCE_ISA=scalar` is specified, the engine safely binds portable scalar fallbacks without `SIGILL` risk.

### 14.2 Non-Temporal Weight Prefetching (`_MM_HINT_NTA`)
Autoregressive token generation streams hundreds of megabytes of static weight pages per token.
- Standard temporal prefetching (`_MM_HINT_T0`) pollutes L2/L3 caches, displacing the active KV-cache and ShortConv states.
- The `LFM2_PREFETCH_1X4_NTA` macro streams weights into L1 using `_MM_HINT_NTA` (bypassing L2/L3 retention) while pinning shared `x_q8` activations in L1D via `_MM_HINT_T0`, improving generation speed by +5% to +10%.

### 14.3 Hardware AVX-VNNI `vpdpbusd` Fused Acceleration
- Uses `_mm256_dpbusd_epi32` to fuse unsigned uint8 by signed int8 multiply-accumulate into 32-bit integer accumulators in **1 clock cycle**, replacing the 2-instruction `_mm256_maddubs_epi16` + `_mm256_madd_epi16` sequence and eliminating 16-bit intermediate overflow.
- Covers 1x1 and 1x4 row-unrolled variants across all 6 formats: `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`, `Q4_0`, and `Q8_0`.
- Signed weights (`Q8_0`) are mapped to unsigned bytes via $u = w \oplus 0\text{x}80$, canceled in $O(1)$ with precomputed activation `bsums`.
- Non-linear weights (`IQ4_XS`) use pre-shifted unsigned grid $kvalues\_iq4nl\_u8 = kvalues\_iq4nl + 128$, executing in 1 instruction without `_mm256_sign_epi8` runtime sign-flips.

### 14.4 Quantization Formats Supported in AVX2
- **`Q4_0`:** 32 weights in 18 bytes. Nibbles are unpacked into $u_4 \in [0, 15]$ in YMM registers, multiplied via `_mm256_maddubs_epi16`, and centered in $O(1)$ by canceling the $-8$ offset with activation sub-block `bsums`.
- **`Q5_K`:** 256 weights in 176 bytes. High bits `qh[0..31]` are loaded once per super-block, combined with 4-bit nibbles to construct 5-bit weights $u_5 \in [0, 31]$, and multiplied without 16-bit register overflow.
- **`IQ4_XS`:** 256 weights in 136 bytes. Uses non-linear 16-entry importance grid lookup via byte shuffle `_mm256_shuffle_epi8` (`vpshufb`) in **1 CPU clock cycle**, converting 32 4-bit indices into non-linear `int8_t` weights instantly.

---

## 15. Runtime Control Plane: Grammar Masking, Reasoning Budgets & Attention Sinks (`tools/lfm2_runtime_features.c`)

### 15.1 Zero-Allocation Byte-Level Pushdown Automaton (PDA) Grammar Logit Masker
- **Pre-Decoded Raw Byte Vocabulary:** Decodes all 65,536 vocabulary tokens once at startup into raw byte strings (`LFM2DecodedVocab`), eliminating UTF-8 string conversions during generation.
- **28-State Byte-Level Pushdown Automaton (`LFM2Grammar`):** Implements strict syntax validation for RFC 8259 JSON objects and LFM2 Pythonic tool calls (`[func(arg="val", count=42)]<|tool_call_end|>`).
- **Speculative Logit Masking:** In $< 50\ \mu\text{s}$, copies the 64-byte grammar struct for candidate tokens and speculatively steps through token bytes. Tokens that violate syntax grammar have their logits set to `-INFINITY`.

### 15.2 `<think>` Reasoning Token Budget Controller (`LFM2ThinkController`)
- Liquid AI reasoning models generate thoughts within `<think>\n...\n</think>\n`.
- Tracks states: `LFM2_THINK_OUTSIDE`, `LFM2_THINK_ACTIVE`, `LFM2_THINK_FORCING_CLOSE`, and `LFM2_THINK_ANSWERING`.
- Enforces `max_think_tokens`. When the budget is reached without closing `</think>`, forces emission of `\n</think>\n` through the normal forward pass by boosting target token logits to $+10^{30}$ and setting others to $-\infty$, ensuring model hidden states remain completely synchronized.

### 15.3 GQA Attention-Sink KV-Cache Shifting & AVX2 RoPE Re-Rotation
Because 10 of LFM2's 16 layers are ShortConvs with fixed 2-token rolling buffers, they never run out of context. Only the 6 GQA layers have a finite `max_seq_len` KV-cache.
- **Cache Eviction:** Retains the first $n_{\text{sink}}$ tokens (system prompt), evicts $n_{\text{discard}}$ old dialogue tokens, and shifts the most recent $n_{\text{keep}}$ tokens down via `memmove`.
- **ShortConv State Invariance:** ShortConv layers require **0 memory moves or counter-rotations**.
- **Exact RoPE Counter-Rotation:** Shifting a key vector by $\Delta = n_{\text{discard}}$ requires rotating by $-\Delta \cdot \theta_i$:
  $$\begin{bmatrix} k_0' \\ k_1' \end{bmatrix} = \begin{bmatrix} k_0 \cos(\Delta\theta_i) + k_1 \sin(\Delta\theta_i) \\ k_1 \cos(\Delta\theta_i) - k_0 \sin(\Delta\theta_i) \end{bmatrix}$$
  Loads precomputed $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ from `LFM2RoPELUT` at index `n_discard` and rotates shifted Keys via AVX2 intrinsics (`_mm256_cvtph_ps`, `_mm256_fmadd_ps`, `_mm256_fmsub_ps`, `_mm256_cvtps_ph`).

---

## 16. Sparse Mixture-of-Experts Architecture (`lfm2moe`)

To scale to multi-billion parameter frontiers (`LFM2-8B-A1B`, `LFM2.5-8B-A1B`, and `LFM2.5-24B-A2B`), Liquid AI introduces a specialized hybrid dense-sparse feed-forward network (`lfm2moe`).

### 16.1 Leading Dense + Sparse MoE Layer Hierarchy
Liquid AI's MoE models do not convert all FFN layers to MoE:
- **Dense Layers (0 and 1):** The first two layers (`num_dense_layers = 2`) remain standard dense SwiGLU layers (`blk.l.ffn_gate.weight`, `blk.l.ffn_up.weight`, `blk.l.ffn_down.weight`). This stabilizes lower-layer representations across tokens.
- **Sparse MoE Layers (2 through $N-1$):** Layers 2 to $N-1$ route each token to a subset of experts:
  - `8B-A1B`: 22 MoE layers with 32 experts per layer, activating $K=4$ experts.
  - `24B-A2B`: 38 MoE layers with 64 experts per layer, activating $K=4$ experts.
  - By evaluating only 4 out of 32 or 64 experts, memory bandwidth demands during autoregressive generation match those of a 1.5B–2B dense model.

### 16.2 Sigmoid Router with Auxiliary-Loss-Free Expert Selection Bias
Unlike traditional MoE models (e.g. Mixtral) that utilize Softmax over all router logits, Liquid AI employs an auxiliary-loss-free **Sigmoid** activation function:
1. **Raw Logits:** The router project normalized hidden state $\hat{\mathbf{x}}$ onto the expert projection matrix:
   $$z_e = \mathbf{W}_{\text{gate\_inp}}[e] \cdot \hat{\mathbf{x}}, \quad e \in \{0, \dots, N_{\text{exp}}-1\}$$
2. **Sigmoid Probability:**
   $$s_e = \sigma(z_e) = \frac{1}{1 + \exp(-z_e)}$$
3. **Selection Bias for Top-$K$ Ranking:**
   To maintain expert load balancing without introducing auxiliary training loss penalties, a learned per-expert bias vector $b_e$ (from `blk.l.exp_probs_b.bias` or `blk.l.exp_probs_b`) is added strictly to the candidate ranking scores:
   $$c_e = s_e + b_e$$
   The top $K=4$ experts $\{e_0, e_1, e_2, e_3\}$ with the highest $c_e$ are chosen.
4. **Unbiased Probability Normalization:**
   Crucially, the learned bias $b_e$ is discarded when computing final expert output mixing weights. The unbiased sigmoid probabilities $s_{e_k}$ are normalized (`norm_topk_prob = true`):
   $$w_k = \frac{s_{e_k}}{\sum_{j=0}^{K-1} s_{e_j} + 10^{-20}}$$
   This ensures output scaling reflects unskewed feature relevance rather than load-balancing offsets.

### 16.3 Zero-Copy 3D Expert Tensor Slicing Over `mmap`
In GGUF files, expert weights are packed as contiguous 3D tensors:
- `ffn_gate_exps.weight`: $[D_{\text{in}}, d_{\text{ffn}}, N_{\text{exp}}]$
- `ffn_up_exps.weight`: $[D_{\text{in}}, d_{\text{ffn}}, N_{\text{exp}}]$
- `ffn_down_exps.weight`: $[d_{\text{ffn}}, D_{\text{out}}, N_{\text{exp}}]$

Because $N_{\text{exp}} = ne[2]$ is the outermost tensor dimension in row-major order:
$$\text{byte\_offset}(e) = e \cdot ne[1] \cdot \left(ne[0] \cdot \frac{\text{block\_size\_bytes}}{\text{block\_size\_weights}}\right)$$
The zero-copy tensor slicer `lfm2_slice_expert_tensor_2d` constructs a 2D tensor descriptor pointing directly into the `mmap` buffer in $O(1)$ time. At runtime:
- In `8B-A1B`, each layer stores 32 experts (~186.75 MB in Q4_K_M).
- Activating 4 experts requires reading only 23.34 MB from DRAM per token (12.5% of layer parameters).
- The remaining 28 experts remain untouched in virtual memory with zero cache pollution.

### 16.4 Fused 1x4 MoE Down-Projection & Parallel Residual Accumulation
1. Activation vector $\hat{\mathbf{x}}$ is pre-quantized to `block_q8_K` structs once per layer.
2. The router selects the top 4 experts and calculates normalized weights $w_k$.
3. For each active expert $k \in \{0..3\}$:
   - Slices 2D tensors for `w_gate`, `w_up`, and `w_down`.
   - Computes fused SwiGLU activation $\mathbf{h} = \text{SiLU}(w_{\text{gate}} \hat{\mathbf{x}}) \odot (w_{\text{up}} \hat{\mathbf{x}})$.
   - Quantizes $\mathbf{h}$ into `block_q8_K`.
4. Down-projection rows are distributed across thread-pool workers via `MoEDownAccumTaskCtx`. Each worker computes its assigned row chunk of $\mathbf{y}_k = w_{\text{down}} \mathbf{h}$, scales by $w_k$, and accumulates directly into the residual vector:
   $$\mathbf{x}_{\text{out}} = \mathbf{x} + \sum_{k=0}^{K-1} w_k \cdot \mathbf{y}_k$$

---

## 17. Hybrid State Checkpointing & Speculative Rollback (`tools/lfm2_moe_checkpoint.c`)

Standard speculative decoding frameworks assume an attention-only architecture where rejecting $M$ draft tokens is trivially accomplished by decrementing the KV-cache sequence cursor: `pos <- pos - M`.

### 17.1 The Recurrent State Overwrite Challenge
Liquid AI models are hybrid architectures:
- **GQA Layers:** State is stored in a linear key-value cache buffer indexed by token position `pos`. Discarding $M$ tokens simply requires resetting `pos`.
- **ShortConv Layers:** State is maintained in a 2-token rolling buffer $(s_0, s_1)$ per channel:
  $$s_0 \leftarrow s_1, \quad s_1 \leftarrow u_t$$
  Each forward step performs an in-place destructive update. If a speculative draft branch generates 4 tokens that are subsequently rejected by the target model, the ShortConv state contains rolling activations from the discarded tokens. Decrementing `pos` leaves the ShortConv recurrent history corrupted, causing catastrophic generation degeneration.

### 17.2 Zero-Allocation Contiguous Checkpoint Slab
The engine implements `LFM2Checkpoint`, preallocating a single 64-byte aligned memory slab storing:
1. **ShortConv Rolling States:** All $(s_0, s_1)$ state buffers across all model layers ($2 \times D \times 4$ bytes per layer; ~160 KB for 1.2B models, ~288 KB for 8B models, ~480 KB for 24B models).
2. **GQA Linear Attention Cursor:** Token position scalar `pos`.
3. **Sampling Ring History:** Rolling token history window for deduplicated repetition penalty.

### 17.3 Sub-20 µs Fast Snapshot & Rollback Protocol
- **`lfm2_checkpoint_save`:** Captures a bit-exact snapshot of all ShortConv states and GQA `pos` via vector-aligned memory copies in $< 15\ \mu\text{s}$.
- **`lfm2_checkpoint_restore`:** When speculative draft tokens are rejected, restores the clean ShortConv state and rewinds `pos` in $< 15\ \mu\text{s}$.
- Enables tree search, beam search, and speculative decoding with zero state degradation and zero heap allocations during token generation.

---

## 18. Linux `/proc/meminfo` OOM Guard & Hybrid Memory Budgeting (`tools/lfm2_sys_cache.c`)

When running large models on edge Linux systems (such as 8 GB RAM developer laptops, single-board computers, or memory-constrained VMs), spawning inference runs with large context windows risks catastrophic process termination by the Linux Out-Of-Memory (OOM) killer.

### 18.1 Mathematical Formulation of Hybrid Footprint
In pure Transformer architectures, memory is overwhelmingly dominated by the expanding KV-cache. In LFM2, the footprint combines constant recurrent state memory with sparse GQA attention:

$$\text{Mem}_{\text{total}} = \text{Mem}_{\text{mmap}} + \text{Mem}_{\text{scratch}} + \text{Mem}_{\text{conv\_states}} + N_{\text{ctx}} \cdot \text{BytesPerToken}_{\text{KV}}$$

Where:
* $\text{Mem}_{\text{mmap}}$ is the model file size mapped via `mmap`.
* $\text{Mem}_{\text{scratch}}$ is the activation arena, RoPE lookup tables, and BPE hash structures (~32 MB).
* $\text{Mem}_{\text{conv\_states}} = N_{\text{conv\_layers}} \cdot (K-1) \cdot D \cdot 4\ \text{bytes}$ is the **constant $O(1)$** ShortConv rolling state buffer (~160 KB for 1.2B models).
* $\text{BytesPerToken}_{\text{KV}} = \sum_{l \in \text{GQA}} 2 \cdot N_{\text{kv\_heads}}^{(l)} \cdot d_{\text{head}} \cdot 2\ \text{bytes}$ across the sparse GQA layers in FP16 precision.

### 18.2 Direct `/proc/meminfo` Headroom Evaluation
Rather than querying non-portable libc hooks or imprecise `sysconf` totals, `lfm2_read_linux_meminfo` directly parses `/proc/meminfo` to extract:
* `MemTotal`: Hardware physical RAM.
* `MemAvailable`: Kernel's estimate of memory available for starting new applications without swapping (accounting for non-reclaimable slabs).
* `Buffers` + `Cached` + `SReclaimable`: Kernel page cache pages that can be reclaimed under memory pressure.
* `SwapTotal` and `SwapFree`: Swap partition utilization.

### 18.3 Dynamic Context Window Clamping
Given an OS reserve threshold $R_{\text{reserve}}$ (default: 512 MB):
$$\text{RAM}_{\text{usable}} = \max\left(\text{Mem}_{\text{available}} - R_{\text{reserve}}, \frac{\text{Mem}_{\text{available}}}{2}\right)$$
$$\text{Budget}_{\text{KV}} = \text{RAM}_{\text{usable}} - \text{Mem}_{\text{scratch}} - \text{Mem}_{\text{conv\_states}} - \frac{\text{Mem}_{\text{mmap}}}{2}$$

If the requested context length $N_{\text{ctx}}$ exceeds the budget:
$$N_{\text{safe}} = \left\lfloor \frac{\text{Budget}_{\text{KV}}}{\text{BytesPerToken}_{\text{KV}}} \right\rfloor$$
$N_{\text{safe}}$ is rounded down to a clean multiple of 256 tokens (clamped to a safe minimum of 512). This guarantees that the engine dynamically adapts to available physical RAM, preventing swapping and OOM crashes.

### 18.4 Transparent 2 MB HugePage Arena (`MADV_HUGEPAGE`)
To eliminate Translation Lookaside Buffer (TLB) thrashing during multi-megabyte KV-cache and activation matrix sweeps, `lfm2_alloc_hugepage_arena` allocates anonymous memory aligned to 2 MB boundaries (`2ULL * 1024ULL * 1024ULL`) and issues `madvise(ptr, size, MADV_HUGEPAGE)`.

---

## 19. Sequential `mmap` Page-Warmup Progress Bar (`--warmup`)

When an inference model is mapped with `mmap(MAP_PRIVATE)`, the kernel populates pages lazily on demand. On mechanical hard drives (HDDs), NFS network mounts, or systems with a cold Linux page cache:
* Autoregressive decoding at Token 0 incurs thousands of random 4 KB page faults across gigabytes of weights.
* Random disk seek latency balloons Token 0 latency to 20+ seconds, creating an impression of an engine stall.

### 19.1 Sequential Page-Fault Sweeping
`lfm2_warmup_mmap_pages` eliminates this stall by turning random seek thrashing into a single high-throughput linear read:
1. Issues `madvise(ptr, size, MADV_WILLNEED)` and `posix_fadvise(fd, 0, size, POSIX_FADV_WILLNEED)` to inform the kernel readahead subsystem.
2. Sweeps through the mapped file in 16 MB windows, reading **1 byte per 4,096 bytes** (stride of 4 KB):
   ```c
   volatile uint8_t dummy = 0;
   for (size_t off = chunk_start; off < chunk_end; off += 4096) {
       dummy ^= ((const volatile uint8_t *)mmap_ptr)[off];
   }
   ```
3. Calculates real-time window transfer rates and renders a live terminal progress bar:
   ```text
   Warming GGUF into RAM: [############............]  50% (  32.0 /   64.0 MB @ 40622.6 MB/s)
   ```
This guarantees that all weight pages are resident in physical RAM before the first prompt token enters prefill.

---

## 20. Disk-Backed Hybrid Prompt Cache (`--prompt-cache`, `.lfmcache` v1)

Standard LLM engines cache system prompts and conversational history by serializing KV-cache buffers to disk. In LFM2, the presence of recurrent ShortConv layers imposes strict architectural rules on caching.

### 20.1 The LFM2 Prefix-Cache Rule: Exact-Prefix vs Partial-Prefix
* **Standard Transformers:** If a cache holds 500 tokens and a new prompt matches the first 300 tokens, the engine can slice and load just the first 300 KV-cache rows, because self-attention is position-indexed and independent across positions.
* **LFM2 Recurrent ShortConv Layers:** 10 of 16 layers (in LFM2-1.2B) are recurrent ShortConv mixers. Their state is a 2-token rolling buffer $(s_0, s_1)$ updated sequentially in-place at step $t$:
  $$s_0 \leftarrow s_1, \quad s_1 \leftarrow u_t$$
  The state saved in the cache file at step 500 represents the model's recurrent state **at step 500**, not step 300. Truncating the KV-cache to 300 while restoring ShortConv states from 500 causes immediate mathematical corruption.
* **Invariant Enforced:** `lfm2_prompt_cache_try_load_prefix` enforces an **exact prefix match** of all $N_{\text{cached}}$ tokens stored in the cache file. If and only if the new prompt starts with all $N_{\text{cached}}$ tokens bit-for-bit, it restores the ShortConv rolling buffers and the first $N_{\text{cached}}$ FP16 KV rows in $< 1\ \text{ms}$, advancing `pos` directly to $N_{\text{cached}}$ so prefill evaluates only the newly appended user message.

### 20.2 Binary File Specification (`.lfmcache` v1)
A `.lfmcache` file is a zero-dependency binary format designed for memory-mapped efficiency:

```text
+---------------------------------------------------------------+
|                      Header (64 Bytes)                        |
| - Magic: 0x4C464D4341434845 ("LFMCACHE") [uint64_t]           |
| - Version: 1 [uint32_t]                                       |
| - Reserved / Header Size: 64 [uint32_t]                       |
| - Model Fingerprint: FNV-1a 64-bit [uint64_t]                 |
| - Payload Checksum: FNV-1a 64-bit [uint64_t]                  |
| - Cached Token Count: N [uint32_t]                            |
| - Padding to 64 bytes                                         |
+---------------------------------------------------------------+
|                      Token ID Array                           |
| - uint32_t tokens[N]                                          |
+---------------------------------------------------------------+
|                 ShortConv Recurrent Buffers                   |
| - float conv_state[N_conv][2 * D]                             |
+---------------------------------------------------------------+
|                   FP16 GQA KV-Cache Rows                      |
| - uint16_t k_cache[N_gqa][N * kv_dim]                         |
| - uint16_t v_cache[N_gqa][N * kv_dim]                         |
+---------------------------------------------------------------+
```

### 20.3 Atomic Writes & Model Fingerprinting
1. **64-Bit FNV-1a Model Fingerprint:** Hashes model architecture string, hidden dimension $D$, layer count $L$, vocab size $V$, ShortConv kernel length $K$, and the full layer routing table `n_head_kv[0..L-1]`. Loading a cache created for a different model or quantization variant is detected and rejected instantly.
2. **Atomic Writes via POSIX `rename()`:** Caches are written to `<path>.tmp.<pid>` and committed via atomic `rename(tmp, target)`. Power loss or `Ctrl+C` termination never corrupts existing disk caches.
3. **Payload Checksum Validation:** An FNV-1a 64-bit hash covers all rolling states and KV bytes, rejecting disk-level corruption or truncated reads.

---

## 21. Public C11 ABI & High-Performance Language Bindings (`include/lfm2.h`, `src/lfm2_api.c`, `python/lfm2.py`)

To expose the hybrid engine to C, C++, Rust, Python, Go, and Zig without leaking internal structs (`GGUFTensor`, `block_q8_K_t`, `LFM2ThreadPool`), the public ABI uses **opaque handles** and **fixed-width Plain Old Data (POD) configuration structs**, built with `-fvisibility=hidden`.

### 21.1 Opaque Engine Handles & Memory Isolation
* `lfm2_model_t`: Read-only model weights mapped directly from GGUF storage via `mmap`. Shareable across multiple independent execution contexts.
* `lfm2_context_t`: Stateful inference session maintaining the linear activation scratch arena, the recurrent ShortConv 2-token rolling buffers, the sparse FP16 GQA KV-cache, the RoPE precomputed LUT, the PDA grammar logit masker, the `<think>` reasoning controller, and the sliding-window UTF-8 boundary buffer.
* `lfm2_checkpoint_t`: Fast snapshot capturing ShortConv rolling states and KV-cache `pos`.

### 21.2 Pull-Based UTF-8 Streaming (`lfm2_context_next_utf8`)
Conventional C LLM wrappers expose a blocking callback loop:
$$\texttt{lfm\_generate(ctx, callback, user\_data)}$$
To turn this into a Python generator (`for token in ctx.stream():`), a language wrapper must spawn a background worker thread, allocate an asynchronous `queue.Queue`, handle inter-thread exception propagation, and pay thread synchronization overhead on every token.

`lfm2_context_next_utf8` replaces this with a **pure single-threaded pull-based step function**:
1. When called, the engine steps the autoregressive forward pass.
2. The generated token is pushed into the embedded sliding-window UTF-8 boundary buffer (`LFM2StreamFilter`).
3. If the token boundary falls in the middle of a multi-byte UTF-8 codepoint (e.g. `0xF0 0x9F` of a 4-byte emoji) or matches a proper prefix of an active stop string (e.g. `"\nUse"` matching `"\nUser:"`), `lfm2_context_next_utf8` **automatically steps subsequent tokens internally** until:
   * A complete, valid UTF-8 sequence is ready, or
   * A stop sequence has triggered completion (`LFM2_DONE`).
4. Result: Python's `for chunk in ctx.stream_chat():` is a single-threaded, zero-overhead generator that guarantees zero mojibake replacement characters (``) and zero stop-prefix leakages.

### 21.3 Hybrid Context Branching (`ctx.branch()`)
In agentic coding, tree-search exploration (MCTS), and speculative decoding, an agent explores hypothetical responses or speculative token drafts before deciding whether to commit or discard them.
* In standard Transformers, discarding 10 tokens requires simply decrementing `pos = pos - 10`.
* In LFM2, decrementing `pos` leaves the recurrent ShortConv rolling buffers corrupted with the discarded tokens.
* The Python SDK provides `with ctx.branch():`, which internally calls `lfm2_checkpoint_create(ctx)`. On block exit (or exception), `lfm2_checkpoint_restore(ctx, ckpt)` is invoked in the `finally:` block, restoring the ShortConv buffers and `pos` in $< 15\ \mu\text{s}$ with bit-exact reproducibility.

### 21.4 High-Throughput L2-Normalized Embeddings (`lfm2_embed_text`)
Computes normalized semantic embeddings over input prompts:
* **Causal Last-Token Pooling (`LFM2_POOL_LAST_TOKEN`):** Extracts the final token's normalized hidden state $h_{N-1} \in \mathbb{R}^D$ and projects through the output RMSNorm:
  $$\mathbf{e} = \frac{\text{RMSNorm}(h_{N-1})}{\|\text{RMSNorm}(h_{N-1})\|_2}$$
* **Mean Pooling (`LFM2_POOL_MEAN`):** Averages the normalized hidden states across all tokens $[0 \dots N-1]$ and applies L2 normalization:
  $$\mathbf{e} = \frac{\frac{1}{N} \sum_{t=0}^{N-1} \text{RMSNorm}(h_t)}{\|\frac{1}{N} \sum_{t=0}^{N-1} \text{RMSNorm}(h_t)\|_2}$$
Vectorized using AVX2 FMA for instant semantic similarity indexing.

---

## 22. Non-Blocking Linux `epoll` OpenAI HTTP/SSE Server (`tools/lfm2_server.c`, `lfm2_server.c`)

To serve high-concurrency local inference without the multi-threading overhead or memory footprint of multi-worker Python web frameworks (such as FastAPI/Uvicorn), the engine incorporates a zero-dependency C11 server utilizing Linux edge-triggered I/O event multiplexing (`sys/epoll.h`).

```text
               +-------------------------------------------+
               |              Linux epoll_wait             |
               +-------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
         [EPOLLIN on listen_fd]                 [Active SSE Generation]
               |                                           |
    accept() -> non-blocking fd                lfm2_context_next_utf8()
    set TCP_NODELAY + EPOLLET                   Send data: {...}\n\n (MSG_NOSIGNAL)
    assign LFM2HttpConn slot                   Immediate return to epoll_wait(0)
               |                                           |
               +---------------------+---------------------+
                                     |
                         [EPOLLRDHUP on client fd]
                                     |
                        Instant Generation Abort
                        Reclaim LFM2HttpConn slot
                        CPU cores immediately freed
```

### 22.1 Zero-Stall Cooperative Event Loop (`epoll_wait` + Single-Token Stepping)
Conventional single-threaded HTTP servers block inside a loop for several seconds when generating long responses:
$$\texttt{while (!done) step\_token();}$$
During this period, no new client connections can be accepted, and active client disconnects cannot be detected.
`lfm2_run_http_server` replaces this with **cooperative token-step scheduling**:
1. When a streaming chat completion request arrives, the connection transitions to `CONN_STREAMING_SSE`, and the server sends the HTTP 200 SSE header.
2. The server records `srv->active_sse_slot = slot`.
3. In the main event loop, the `epoll_wait` timeout is dynamically adjusted:
   $$\text{timeout} = \begin{cases} 0\ \text{ms} & \text{if } \text{active\_sse\_slot} \ge 0 \\ 200\ \text{ms} & \text{otherwise} \end{cases}$$
4. On each event loop tick, `server_step_active_sse` calls `lfm2_context_next_utf8()` **exactly once**, formatting a single verified UTF-8 token chunk into an OpenAI SSE event `data: {"choices":[{"delta":{"content":"..."}}]}\n\n`.
5. The frame is dispatched immediately via `send(fd, ..., MSG_NOSIGNAL)` with `TCP_NODELAY` active ($< 50\ \mu\text{s}$ kernel socket flush), and the event loop immediately polls `epoll_wait` again.
6. Generation runs at full CPU speed while the server remains completely non-blocking and receptive to I/O events.

### 22.2 Instant Client-Disconnect Cancellation (`EPOLLRDHUP`)
If a user closes their browser tab, clicks "Stop Generating", or interrupts a `curl` process (`Ctrl+C`):
1. The kernel signals `EPOLLRDHUP | EPOLLHUP | EPOLLERR` on the client socket.
2. Because the event loop polls with `timeout = 0` between every single token step, `epoll_wait` catches the disconnect event on the **very next token boundary** ($\le 25\ \text{ms}$).
3. The server immediately invokes `server_close_conn(srv, slot)`, releasing the client descriptor and resetting `active_sse_slot = -1`.
4. Subsequent generation steps are halted instantaneously, preventing wasted CPU cycles and core thrashing on dead connections.

### 22.3 Pre-Allocated Connection Pool & Zero-Allocation Socket Framing
To satisfy **Architectural Invariant 2** (Zero dynamic allocations during generation):
* All connections are managed in a fixed-size statically allocated array:
  $$\texttt{LFM2HttpConn conns[LFM2\_MAX\_CONNS] (32 slots)}$$
* Each slot pre-allocates a 64 KB request buffer (`req_buf`) and a 128 KB response buffer (`resp_buf`).
* Embedding vector responses (up to 4,096 floating-point values) serialize directly into a server-owned scratch buffer (`srv->json_scratch`), completely eliminating `malloc` and `free` during request handling.

### 22.4 OpenAI API Compatibility & Embedded Web Studio
The server exposes full OpenAI REST API v1 compatibility and an embedded dark-mode browser studio:
1. `GET /`: Embedded zero-dependency HTML/CSS/JavaScript Chat Studio (`LFM2_INDEX_HTML`) with live token-per-second telemetry and auto-scrolling message stream.
2. `GET /health`: JSON engine readiness check (`{"status":"ok"}`).
3. `GET /v1/models`: Model metadata reporting active ISA (`AVX2 + FMA + F16C`), layer counts, and embedding dimensions.
4. `POST /v1/chat/completions`: Supports both streaming Server-Sent Events (`stream: true`) and non-streaming responses (`stream: false`), temperature, top-k, min-p, repetition penalty, stop strings, and grammar constraints. Emits `"delta":{"reasoning_content":"..."}` when generating inside `<think>` tags.
5. `POST /v1/embeddings`: High-throughput semantic text embedding generation using causal last-token or mean pooling.
6. `OPTIONS *`: HTTP CORS preflight handler responding with 204 No Content for seamless web client integration.
7. **Anti-Slowloris Monotonic Deadlines:** Discards idle connections in `CONN_READING_REQ` exceeding 10.0 seconds.

---

## 23. Mobile Edge & Android Scoped Storage Integration (`lfm2_loader.c`, `src/lfm2_api.c`, `python/lfm2.py`)

Deploying local on-device inference on Android smartphones (e.g. via Kivy, Flutter, or native apps) presents unique OS sandbox and memory lifecycle challenges:

### 23.1 Android Scoped Storage & File Descriptor Zero-Copy Loading (`lfm2_model_load_from_fd`)
On Android 10+ (API level 29+), applications cannot access arbitrary raw filesystem paths like `/sdcard/model.gguf`. Models selected by users via Android SAF (`ACTION_OPEN_DOCUMENT`) are provided strictly as content URIs (`content://...`) and open file descriptors (`ParcelFileDescriptor.getFd()`).
* **File Descriptor Duplication (`dup(fd)`):** `lfm2_load_gguf_from_fd` duplicates the descriptor using `dup(fd)` so the C engine owns an independent lifecycle reference, allowing the calling Java/Kotlin/Python layer to safely manage or close its original descriptor.
* **Direct Virtual Memory Mapping:** Maps the descriptor directly into virtual memory via `mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0)` with `madvise(MADV_SEQUENTIAL)`. Zero byte copies are made from Android storage to user space.
* **Pure Python stdlib Integration:** Exposed cleanly in [`python/lfm2.py`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/python/lfm2.py) as `LFM2Model.from_fd(fd)`.

### 23.2 Surviving the Android Low Memory Killer (`lmkd`) via `MADV_DONTNEED`
When an Android app transitions to the background (`onPause` / `onStop`), Android's `lmkd` daemon aggressively terminates high-memory processes to reclaim RAM.
* **Transparent Page Cache Eviction:** [`lfm2_model_release_page_cache`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2.h) issues `madvise(mmap_addr, mmap_size, MADV_DONTNEED)`.
* **Instant RSS Drop to Zero:** Tells the Linux kernel that the physical backing pages for the mapped `.gguf` file can be discarded immediately. The virtual memory address space remains intact, but the Resident Set Size (RSS) drops to zero, preventing `lmkd` kills.
* **Seamless Resume:** When the user brings the app back to the foreground (`onResume`), the kernel automatically page-faults the model weights back into RAM on demand without reloading or re-initializing the engine.
* **Python Binding:** Exposed directly via `model.release_page_cache()`.

### 23.3 Dynamic 16 KB Page Compatibility (Android 15+)
Starting with Android 15, Android devices transition from legacy 4 KB memory pages to **16 KB physical pages**. Hardcoding 4096-byte alignment or page strides causes page-boundary misalignment faults and crashes.
* Both [`tools/lfm2_loader.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_loader.c) and [`tools/lfm2_sys_cache.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c) query the hardware page size dynamically at runtime via `sysconf(_SC_PAGESIZE)`:
  ```c
  long page_sz = sysconf(_SC_PAGESIZE);
  size_t stride = (page_sz > 0) ? (size_t)page_sz : 4096;
  ```
* Seamlessly supports 4 KB, 16 KB, and 64 KB Linux/Android kernel page configurations.

---

## 24. Unified Cross-Platform Asymmetric `big.LITTLE` Thread Pool (`tools/lfm2_threadpool.c`)

Autoregressive token generation on asymmetric multi-core ARM64 SoCs (Snapdragon, MediaTek Dimensity, Google Tensor, Samsung Exynos) suffers from the **`big.LITTLE` straggler problem**:
* An 8-core mobile SoC typically contains:
  * **1 Prime Core** (e.g. Cortex-X4 @ 3.3 GHz, EAS capacity 1024)
  * **3 to 4 Performance Cores** (e.g. Cortex-A720 @ 2.8 GHz, EAS capacity 800)
  * **3 to 4 Efficiency / LITTLE Cores** (e.g. Cortex-A520 @ 1.8 GHz, in-order pipeline, EAS capacity 320)
* If an inference engine spawns worker threads across all 8 cores, the fast Big cores finish their matrix chunks in $< 0.8\ \text{ms}$ and sit spinning at the barrier waiting for the slow in-order Little cores. This yields **slower** throughput than running on 3–4 cores while burning double the power and causing severe thermal throttling.

```text
+-------------------------------------------------------------------------+
|                  Unified Cross-Platform Core Dispatch                   |
+-------------------------------------------------------------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v (Homogeneous PC: x86_64)                          v (Heterogeneous Mobile: ARM64)
+------------------------------------+              +------------------------------------+
|  Topology: All cores equal speed   |              |  Topology: Sysfs EAS Capacity      |
|  Rule: N_phys - 1 cores            |              |  Rule: N_big - 1 cores (e.g. 3)    |
|  Leaves 1 physical core for OS/UI  |              |  Leaves 1 Big for UI, Little for OS|
|  Ceiling: N_phys - 1               |              |  Strict Ceiling: N_total - 2       |
+------------------------------------+              +------------------------------------+
```

### 24.1 Dynamic Sysfs EAS Capacity & Frequency Scanner (`LFM2CpuTopology`)
The thread pool scans the Linux sysfs hierarchy at startup:
1. Reads Energy-Aware Scheduler (EAS) capacity from `/sys/devices/system/cpu/cpuN/cpu_capacity` ($0 \dots 1024$).
2. Reads maximum clock frequency from `/sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq` (kHz).
3. Classifies each CPU core into tiers:
   - `LFM2_CORE_PRIME`: Capacity $\ge 900$ or Frequency $\ge 90\%$ of peak.
   - `LFM2_CORE_PERF`: Capacity $\ge 600$ or Frequency $\ge 65\%$ of peak.
   - `LFM2_CORE_LITTLE`: Low-power efficiency core.
4. Sorts cores by compute capacity in descending order (`recommended_cpus`).

### 24.2 Cluster-Aware Reservation & Thermal Blackout Mitigation
* **Homogeneous PC Invariant ($N_{\text{phys}} - 1$):** On PCs, reserves $N_{\text{phys}} - 1$ cores (e.g., 4 physical cores $\to$ 3 workers), leaving 1 dedicated physical core for OS tasks, browser audio, and UI rendering.
* **Mobile Asymmetric Rule ($N_{\text{big}} - 1$):** On heterogeneous SoCs, automatically sets thread count to $N_{\text{big}} - 1$ (e.g., on a 1 Prime + 3 Perf + 4 Little phone, reserves 3 Big cores for inference: 1 Prime + 2 Perf). This leaves 1 high-performance Big core for Kivy's OpenGL UI thread and 100% of the Little cores for Android OS background services.
* **Thermal Blackout & Vdroop Guard ($N_{\text{total}} - 2$):** Clamps maximum user-requested threads to $N_{\text{total}} - 2$ (e.g., 6 cores maximum on an 8-core phone). Completely eliminates full-chip saturation that causes thermal throttling, Vdroop voltage drops, and sudden device shutdowns.
* **Worker 0 on Prime Core:** Pins Worker 0 (the calling thread) to the fastest Prime core (`recommended_cpus[0]`), accelerating single-threaded sequential kernels ([`RMSNorm`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm2_api.c), ShortConv 3-tap depthwise shift, RoPE, and sampling).

### 24.3 Cross-Platform Architecture Primitives & Flush-to-Zero
* **Hardware Flush-to-Zero (`FPCR.FZ`):** On ARM64 workers, sets bit 24 (`FZ`) and bit 19 (`FZ16`) of the Floating-Point Control Register (`fpcr`). Prevents denormal floating-point numbers in ShortConv rolling states from triggering 100-cycle microcode assist traps.
* **Cross-Platform Spin Pause:** Implements `lfm2_cpu_pause()` using `_mm_pause()` on x86_64 and `asm volatile("yield" ::: "memory")` on ARM64.
* **Adaptive Spin Loops:** Uses 25,000 iterations on x86_64 desktop CPUs vs 4,000 iterations on ARM64 mobile CPUs before falling back to `SYS_futex`, dramatically preserving battery life and reducing thermal dissipation.

---

## 25. Native ARM64 NEON + DotProd (`vdotq_s32`) + FP16 SIMD Suite (`tools/lfm2_simd_arm64.c`, `include/lfm2_simd_arm64.h`)

On 64-bit Android / mobile processors (`aarch64`), quantized matrix-vector inference leverages three specific architectural advantages over x86_64 AVX2:

### 25.1 32 Architectural 128-bit Vector Registers (`v0..v31`)
* While x86 AVX2 is constrained to 16 vector registers (`ymm0..ymm15`), ARM64 provides 32 full 128-bit vector registers.
* In 1x4 row-unrolled microkernels (`lfm2_vec_dot_q4_K_q8_K_1x4_neon_dp`), all 4 row accumulators, pre-quantized `Q8_K` activations, weight rows, scale multipliers, and min offsets remain entirely register-resident with **zero stack spills**.

### 25.2 Single-Cycle Signed Dot Products (`vdotq_s32` / `SDOT`)
* x86 AVX2 relies on `_mm256_maddubs_epi16`, which requires unsigned weights and signed activations, widening only to 16-bit intermediate integers. Centering offsets and avoiding 16-bit lane saturation requires complex `bsums[16]` adjustments and a second `_mm256_madd_epi16` instruction.
* ARMv8.2-A's `vdotq_s32` multiplies 16 pairs of signed `int8_t` values and accumulates them directly into four signed `int32_t` lanes in 1 clock cycle.
* Slashes instruction count by ~40% and reaches **25–45 tokens/sec** for 1.2B models on modern mobile SoCs, fully saturating LPDDR5 bandwidth.

### 25.3 1-Cycle Table Lookup via NEON `TBL` (`vqtbl1q_s8`)
* Sub-blocks in `Q6_K` and `IQ4_XS` are exactly 16 weights, mapping 1-to-1 onto a single 16-byte `int8x16_t` NEON register.
* `vqtbl1q_s8` evaluates non-linear `kvalues_iq4nl` table lookups for 16 weights in a single cycle without cross-lane shuffles.

### 25.4 Dynamic Runtime CPU Detection via `getauxval(AT_HWCAP)`
* Queries Linux `getauxval(AT_HWCAP)` and `getauxval(AT_HWCAP2)` for `HWCAP_ASIMDDP` (DotProd), `HWCAP_FPHP` (FP16), and `HWCAP2_I8MM`.
* Accelerated functions use `__attribute__((target("arch=armv8.2-a+dotprod+fp16")))`.
* Automatically dispatches to hardware `SDOT` on modern phones while falling back to portable scalar loops on pre-2018 devices, guaranteeing zero `SIGILL` crashes across Android versions.

---

## 26. Zero-Weight Prompt-Lookup Speculative Decoding (`tools/lfm2_speculative_pld.c`, `include/lfm2_speculative_pld.h`)

On desktop PCs with dual-channel DDR4 or DDR5 RAM (~25–50 GB/s bandwidth), single-token autoregressive decoding ($M = 1$) is strictly memory-bandwidth bound. Evaluating 65,536-row quantized weight matrices over and over limits throughput to 35–45 tokens/sec regardless of CPU core count.

Prompt-Lookup Speculative Decoding (PLD) breaks through this memory ceiling **without loading a secondary draft model into RAM**:

### 26.1 $O(1)$ Hierarchical N-Gram Drafting (`LFM2NgramIndex`)
* In code generation, RAG retrieval QA, JSON schemas, and multi-turn conversations, 40% to 70% of generated tokens already appeared verbatim in the prompt or prior context.
* `LFM2NgramIndex` hashes every 3-gram and 2-gram into a 16,384-slot power-of-2 L1-resident table using 64-bit mixer hashes.
* Before generating, the engine matches the current suffix against prior occurrences and copies up to $K \le 7$ continuation tokens as candidate draft tokens in **$< 1\ \mu\text{s}$** with zero matrix multiplies.

### 26.2 Amortized Verification via Batched GEMM ($B = K + 1 \le 8$)
* The pending token and $K$ drafts are packed into a micro-batch of size $B = K + 1 \le 8$ and evaluated via `lfm2_forward_speculative_batch`.
* The multi-threaded AVX2 kernel streams model weights (`Q4_K`, `Q6_K`) from RAM **once** and multiplies them across all $B$ tokens in L1 cache.
* Evaluating 6 tokens takes only $\sim 1.25\times$ the time of evaluating 1 token.

### 26.3 Per-Slot ShortConv Rolling Trace (`conv_trace[b][l]`)
* To solve the hybrid recurrent/convolution state rewind problem without recomputation, `lfm2_forward_shortconv_spec_batch` snapshots each layer's 16 KB conv state after causal token step $b$ into `conv_trace[b][l]`.
* When verification halts at slot $m$ (where draft $m$ is rejected), `lfm2_restore_shortconv_trace_slot` snaps `conv_trace[m][l]` back into `conv_state[l]` in **$8\ \mu\text{s}$**, achieving zero-recomputation causal rollbacks.

### 26.4 100% Bit-Exact Sequence Parity Contract
* Every draft token is verified against the target model's output logits.
* Accepted drafts advance position; rejected drafts are replaced by the target model's true sampled token.
* The emitted token sequence is **mathematically identical** to standard single-token autoregressive decoding.





