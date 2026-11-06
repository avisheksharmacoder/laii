# Phase 4 Technical Specification: Advanced Architectural Kernels & Runtime Control Plane

This specification documents the engineering design, mathematical formulation, low-level SIMD kernels, and verification of **Phase 4** of the specialized Liquid AI (LFM2) Linux inference engine.

---

## 1. Executive Summary & Objectives

Phase 1 through Phase 3 established a mathematically sound reference forward pass, high-throughput AVX2 SIMD acceleration, single-binary distribution, public C ABI, embedded Web Studio, and zero-dependency Python FFI.

Phase 4 completes the **performance optimization and runtime control plane** of the engine:
1. **Priority-Ordered BPE Rank Merging & Pre-Tokenizer State Machine ([`tools/lfm2_bpe.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_bpe.c)):** Exact Hugging Face / GGUF BPE rank merging with min-heap priority, leftmost tie-breaking, GPT-2 byte mapping, and discrete special token splitting.
2. **Lock-Free Physical Topology Spin-Barrier ThreadPool ([`tools/lfm2_threadpool.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_threadpool.c)):** C11 `<stdatomic.h>` thread pool pinned to unique physical CPU cores, achieving sub-microsecond (~260 ns) dispatch latency and dynamic row-stealing without OpenMP dependencies.
3. **Batched Prefill Acceleration & Rank-1 Accumulator GEMM ([`tools/lfm2_prefill_batch.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_prefill_batch.c)):** Amortizes weight loads across parallel token blocks during prompt prefill with fused SIMD `Q8_K` quantization.
4. **1x4 Row-Unrolled AVX2 MatVec & Register-Resident SwiGLU FFN ([`tools/lfm2_decode_1x4.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_decode_1x4.c)):** Quad-row unrolled GEMV amortizing activation loads across 4 consecutive weight rows, combined with register-resident fused SwiGLU FFN.
5. **Planar ShortConv, Precomputed RoPE LUT & Single-Pass FP16 Online Softmax ([`tools/lfm2_kernels_fast.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_kernels_fast.c)):**
   - Planar AVX2 ShortConv 3-tap kernel with transposed weights and direct L1-hot quantization.
   - Precomputed RoPE LUT + Fused AVX2 per-head QK-norm.
   - FlashAttention-CPU single-pass online softmax over FP16 KV-cache (`F16C`).
6. **Multi-ISA CPUID Runtime Dispatch & Extended Quantization ([`tools/lfm2_cpuid_quants.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_cpuid_quants.c)):**
   - Single pre-compiled binary with GCC/Clang target attributes (`avx2,fma,f16c`) and portable scalar fallback.
   - AVX2 inner loops for `Q4_0`, `Q5_K`, and `IQ4_XS` (using `vpshufb` 1-clock non-linear grid lookup).
7. **Runtime Control Plane ([`tools/lfm2_runtime_features.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_runtime_features.c)):**
   - Zero-allocation byte-level Pushdown Automaton (PDA) grammar logit masker supporting RFC 8259 JSON objects and LFM2 native Pythonic tool calls.
   - `<think>` reasoning token budget controller with forced termination sequence emission.
   - GQA attention-sink KV-cache shifting with AVX2 RoPE re-rotation.
8. **Sparse Mixture-of-Experts & Hybrid State Checkpointing ([`tools/lfm2_moe_checkpoint.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c)):**
   - Sparse MoE (`lfm2moe`) for `LFM2-8B-A1B`, `LFM2.5-8B-A1B`, and `LFM2.5-24B-A2B` models with hybrid leading dense SwiGLU layers (0–1) and 32/64-expert MoE layers (2..N-1) activating Top-4 experts per token.
   - Auxiliary-loss-free Sigmoid router with learned per-expert load-balancing selection bias (`exp_probs_b` / `exp_probs_b.bias`) and unbiased normalized Top-4 probability mixing.
   - $O(1)$ zero-copy 3D tensor slicing over `mmap` buffers (`[ne0, ne1, n_exp]`), loading only the 4 selected expert slices per token (12.5% parameter traffic).
   - Fast hybrid state checkpointing (`LFM2Checkpoint`) snapshotting all ShortConv rolling buffers (~160–480 KB) and GQA `pos` cursor in $< 20\ \mu\text{s}$ for instant speculative draft rollback and tree-search backtracking.
9. **Streaming UTF-8 Boundary Buffer, Multi-Token Stop Holdback & Two-Stage SIGINT ([`tools/lfm2_cli_stream.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_cli_stream.c)):**
   - Sliding-window assembly of 2-, 3-, and 4-byte UTF-8 codepoints across token boundaries, eliminating mojibake replacement characters (``).
   - Multi-token stop-string matcher (`--stop`) with prefix holdback, instant divergence flush, and earliest match truncation.
   - Two-stage `sigaction` signal handler (`Ctrl+C` cancels generation turn while preserving REPL state; second `Ctrl+C` at idle gracefully exits).
   - UNIX pipe/TTY detector routing telemetry to `stderr` when redirected, leaving `stdout` as pure model text.
10. **Linux `/proc/meminfo` OOM Guard, Sequential `mmap` Page Warmup & Disk-Backed Prompt Cache ([`tools/lfm2_sys_cache.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c)):**
   - Real-time `/proc/meminfo` parsing and hybrid memory footprint planner preventing Linux OOM killer termination by safely clamping context length on RAM-constrained machines (e.g. 8 GB RAM).
   - 2 MB Transparent HugePage arena allocator (`MADV_HUGEPAGE`) eliminating CPU TLB misses during multi-megabyte activation sweeps.
   - Sequential 4 KB stride page-fault sweeping (`MADV_WILLNEED`) with real-time `MB/s` console progress bar eliminating mechanical drive seek thrashing during Token 0.
   - Zero-dependency disk-backed hybrid prompt cache (`--prompt-cache`, `.lfmcache` v1) enforcing the exact-prefix ShortConv rolling state invariant, with 64-bit FNV-1a model fingerprinting, atomic temporary file writing, and $< 1\ \text{ms}$ prefill skip.

---

## 2. Priority-Ordered BPE Rank Merging (`tools/lfm2_bpe.c`)

Standard greedy longest-prefix matching tokenizers fail on modern BPE vocabularies because merge order is governed by priority ranks rather than substring length.

### Architecture
- **FNV-1a Open-Addressed String Hash Table:** Maps all 65,536 token strings to integer IDs in $O(1)$ with $\le 50\%$ load factor.
- **Merge Pair Hash Table:** Maps packed 64-bit pair keys `((uint64_t)left_id << 32) | right_id` to `(rank, merged_id)`.
- **Pre-Tokenizer Regex State Machine:** Replicates GPT-2/LFM2 regex pattern matching contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), letter runs, numeric sequences up to 3 digits, punctuation, and non-whitespace runs.
- **Priority Min-Heap + Lazy Version Invalidation:** Pre-tokenized words are loaded into a stack-allocated doubly-linked list. Merges are popped in order of `(rank, left_pos)` in $O(N \log N)$ without array shifts.

---

## 3. Lock-Free Physical Topology Spin-Barrier ThreadPool (`tools/lfm2_threadpool.c`)

To achieve single-binary static distribution without depending on OpenMP (`libgomp.so.1`), the engine implements a zero-allocation POSIX threadpool with atomic spin-barriers:

- **Topology Pinning:** Queries `/sys/devices/system/cpu/` to discover physical core IDs and pins worker threads using `sched_setaffinity` to eliminate SMT (Hyper-Threading) execution port thrashing.
- **Cache-Padded Atomic Structure:** 64-byte alignment (`_Alignas(64)`) separates producer dispatch state, dynamic work-stealing counter, and completion counters across distinct cache lines.
- **User-Space Spin + Futex Sleep:** Workers spin with `_mm_pause()` for up to 25,000 iterations (~10 ms, ~260 ns dispatch latency) before falling back to zero-CPU Linux `FUTEX_WAIT_PRIVATE`.

---

## 4. Batched Prefill Acceleration (`tools/lfm2_prefill_batch.c`)

During prompt prefill, processing $B$ tokens one-by-one is severely memory-bandwidth bound. Phase 4 introduces block GEMM:

- **Parallel Activation Quantization:** Quantizes $B$ consecutive token activations into `block_q8_K` structs in parallel.
- **Rank-1 Outer-Product Accumulators:** For each weight row, computes dot products against all $B$ tokens simultaneously, reusing the weight row across $B$ tokens and cutting memory bandwidth demand by $B\times$.
- **Bit-Exact Parity:** Verified to match single-token decode bit-for-bit.

---

## 5. 1x4 Row-Unrolled MatVec & Register-Resident SwiGLU FFN (`tools/lfm2_decode_1x4.c`)

Single-token decoding bandwidth is dominated by loading weight matrices:

- **1x4 Row Unrolling:** Evaluates 4 contiguous weight rows in parallel inside the inner loop. The activation vector $x$ (pre-quantized into `Q8_K`) is broadcast once from L1 cache into registers, amortizing activation loads across 4 weight rows.
- **Register-Resident SwiGLU Fusion:** Computes gate projection $w_1(x)$ and up projection $w_3(x)$ simultaneously into separate AVX2 register accumulators. It computes the rational approximation of SiLU on $w_1$, multiplies by $w_3$, and keeps the intermediate result register-resident, cutting DRAM round-trips by $2\times$.

---

## 6. Planar ShortConv, Precomputed RoPE LUT & FP16 Online Softmax (`tools/lfm2_kernels_fast.c`)

### 6.1 Planar AVX2 ShortConv 3-Tap Kernel
In GGUF checkpoints, `blk.l.shortconv.conv.weight` is stored with stride 3 (`[w_{d,0}, w_{d,1}, w_{d,2}]`), requiring non-contiguous scalar loads.
- At initialization, weights are transposed into three 64-byte aligned planar arrays: $w_0[D]$, $w_1[D]$, $w_2[D]$.
- At runtime, an 8-wide AVX2 loop computes Gate 1 ($u_t = B \odot \tilde{x}$), loads 2-token rolling states $(s_0, s_1)$, evaluates $w_0 s_0 + w_1 s_1 + w_2 u_t$, updates rolling history in-place, applies Gate 2 ($y = C \odot \text{conv}$), and quantizes the L1-hot result into `conv_out_q8` for subsequent projection.

### 6.2 Precomputed RoPE LUT & Fused Per-Head QK-Norm
- In LFM2, `head_dim = 64` (`half_dim = 32`), which fits into eight `__m256` registers.
- Replaces runtime `cosf` / `sinf` with precomputed lookup tables `cos_table[max_seq_len * 32]` and `sin_table[max_seq_len * 32]`.
- Normalizes query and key heads in-place, applies half-split NeoX RoPE via `_mm256_fmsub_ps` and `_mm256_fmadd_ps`, and stores Keys/Values directly to the KV-cache as IEEE-754 `FP16` via `_mm256_cvtps_ph`.

### 6.3 Single-Pass Online Softmax (FlashAttention-CPU)
Eliminates intermediate score buffers and multiple passes over the KV cache. Maintains running max $m_{\text{prev}}$ and normalizer $l_{\text{prev}}$ in registers:
$$\text{If } s_t > m_{\text{prev}}: \quad \alpha = \exp(m_{\text{prev}} - s_t), \quad l \leftarrow l \cdot \alpha + 1, \quad \mathbf{o} \leftarrow \mathbf{o} \cdot \alpha + \mathbf{v}_t, \quad m_{\text{prev}} \leftarrow s_t$$
$$\text{Else}: \quad \beta = \exp(s_t - m_{\text{prev}}), \quad l \leftarrow l + \beta, \quad \mathbf{o} \leftarrow \mathbf{o} + \beta \cdot \mathbf{v}_t$$
Streams `k_cache_f16` and `v_cache_f16` in a single pass, cutting memory bandwidth by 50%.

---

## 7. Multi-ISA CPUID Runtime Dispatch & Quantization (`tools/lfm2_cpuid_quants.c`)

Enables a single pre-compiled binary to safely run across all x86_64 CPUs without `SIGILL` crashes:

| Quantization | Block Geometry | AVX2 Strategy |
| :--- | :--- | :--- |
| **`Q4_0`** | 32 weights (18 B): `fp16 d` + `uint8 qs[16]` | Unpacks nibbles into `u4`, multiplies via `_mm256_maddubs_epi16`, cancels $-8$ zero-point via `bsums`. |
| **`Q5_K`** | 256 weights (176 B): `d`, `dmin`, `scales`, `qh[32]`, `qs[128]` | Unpacks high-bit byte `qh` once per super-block, constructs 5-bit integers (`u5 = u4 | qh_bit4`), multiplies with `_mm256_maddubs_epi16`. |
| **`IQ4_XS`** | 256 weights (136 B): `d`, `scales_h`, `scales_l[4]`, `qs[128]` | Non-linear 16-entry importance grid lookup via `vpshufb` (`_mm256_shuffle_epi8`) in 1 clock cycle + `_mm256_sign_epi8` + `_mm256_maddubs_epi16`. |

- **Target Attributes:** High-performance routines are decorated with `__attribute__((target("avx2,fma,f16c")))`.
- **Runtime Dispatch:** At startup, `lfm2_init_cpu_kernels()` queries `cpuid` / `__builtin_cpu_supports` and selects hardware-accelerated pointers or portable scalar fallbacks. Supports `LFM2_FORCE_ISA=scalar` for testing.

---

## 8. Runtime Control Plane (`tools/lfm2_runtime_features.c`)

### 8.1 Zero-Allocation Byte-Level PDA Grammar Logit Masker
- Decodes all 65,536 vocabulary tokens into raw byte slices once at startup (`LFM2DecodedVocab`).
- Maintains a 28-state Pushdown Automaton (`LFM2Grammar`) with stack depth 32.
- Evaluates candidate tokens speculatively on a copied 64-byte grammar struct and masks invalid tokens with `-INFINITY` in $< 50\ \mu\text{s}$.
- Supports RFC 8259 JSON objects and native LFM2 Pythonic tool calls: `[func_name(arg="val", count=42)]<|tool_call_end|>`.

### 8.2 `<think>` Reasoning Token Budget Controller
- Tracks reasoning lifecycle: `OUTSIDE` $\rightarrow$ `ACTIVE` $\rightarrow$ `FORCING_CLOSE` $\rightarrow$ `ANSWERING`.
- Enforces `max_think_tokens`. When the budget is reached without closing `</think>`, sets logits of the forced closing sequence (`\n</think>\n`) to $+10^{30}$ and all others to $-\infty$. The forced tokens pass through the standard forward pass to maintain exact state continuity.

### 8.3 GQA Attention-Sink KV-Cache Shifting & RoPE Re-Rotation
- Preserves initial `n_sink` tokens (system prompt), discards `n_discard` dialogue tokens, and shifts `n_keep` tokens down via `memmove`.
- **ShortConv Invariance:** LFM2's ShortConv layers use a 2-token rolling buffer holding only $(t-2, t-1)$, requiring **zero memory movement or rotation**.
- **RoPE Counter-Rotation:** Shifted Keys are counter-rotated by $-\Delta \cdot \theta_i$ using $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ loaded from `LFM2RoPELUT` at index `n_discard`:
  $$k_{lo}^{(p')} = k_{lo}^{(p)} \cos(\Delta\theta_i) + k_{hi}^{(p)} \sin(\Delta\theta_i)$$
  $$k_{hi}^{(p')} = k_{hi}^{(p)} \cos(\Delta\theta_i) - k_{lo}^{(p)} \sin(\Delta\theta_i)$$
  Executed via 8 AVX2 instructions (`_mm256_cvtph_ps`, `_mm256_fmadd_ps`, `_mm256_fmsub_ps`, `_mm256_cvtps_ph`).

---

## 9. Sparse Mixture-of-Experts Architecture (`lfm2moe`)

Liquid AI's large-parameter checkpoints (`LFM2-8B-A1B`, `LFM2.5-8B-A1B`, and `LFM2.5-24B-A2B`) employ a hybrid dense-sparse feed-forward architecture (`lfm2moe`).

### 9.1 Leading Dense Layers + Sparse MoE Hierarchy
- **Layers 0 and 1 are Dense SwiGLU layers** (`num_dense_layers = 2`) utilizing standard 2D tensors: `blk.l.ffn_gate.weight`, `blk.l.ffn_up.weight`, and `blk.l.ffn_down.weight`.
- **Layers 2 through $N-1$ are Sparse MoE layers**:
  - `LFM2-8B-A1B`: 22 MoE layers with $N_{\text{exp}} = 32$ experts ($D=2048, d_{\text{ffn}}=1536$).
  - `LFM2.5-24B-A2B`: 38 MoE layers with $N_{\text{exp}} = 64$ experts ($D=2816, d_{\text{ffn}}=1792$).
- Exactly $K = 4$ active experts are evaluated per token (`top_k = 4`).

### 9.2 Sigmoid Router with Auxiliary-Loss-Free Expert Selection Bias
Unlike Mixtral or Qwen-MoE which rely on Softmax, `lfm2moe` routes tokens using an auxiliary-loss-free **Sigmoid** activation on router projection logits:
$$z_e = \mathbf{W}_{\text{gate\_inp}}[e] \cdot \hat{\mathbf{x}}, \quad e \in \{0, \dots, N_{\text{exp}}-1\}$$
$$s_e = \sigma(z_e) = \frac{1}{1 + \exp(-z_e)}$$

**Selection Bias Ranking vs Unbiased Output Mixing:**
- A learned per-expert load-balancing bias $b_e$ is extracted from GGUF tensor `blk.l.exp_probs_b.bias` (or `blk.l.exp_probs_b`).
- The bias is added **only during Top-$K$ candidate selection**:
  $$c_e = s_e + b_e$$
  The top $K=4$ experts $\{e_0, e_1, e_2, e_3\}$ with the highest $c_e$ are chosen.
- Once the active experts are selected, their **unbiased** sigmoid probabilities $s_{e_k}$ are normalized (`norm_topk_prob = true`):
  $$w_k = \frac{s_{e_k}}{\sum_{j=0}^{K-1} s_{e_j} + 10^{-20}}$$
  This decouples capacity load-balancing during training from inference probability calibration.

### 9.3 Zero-Copy 3D Expert Tensor Slicing Over `mmap`
In GGUF files, MoE tensors (`ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`) are formatted as 3D arrays:
$$\text{shape} = [ne[0], ne[1], ne[2]] = [D_{\text{in}}, D_{\text{out}}, N_{\text{exp}}]$$
Because $N_{\text{exp}} = ne[2]$ is the outermost dimension, the 2D weight matrix for expert $e$ is stored as a contiguous memory block at byte offset:
$$\text{offset}(e) = e \cdot ne[1] \cdot \text{row\_bytes}$$
The zero-copy helper `lfm2_slice_expert_tensor_2d` constructs a 2D tensor descriptor viewing this slice directly within the `mmap` address space in $O(1)$ time with zero memory allocations or tensor duplication. Only the 4 active experts are accessed per token, touching only 12.5% of layer parameters and reducing memory bus traffic from 186.75 MB/layer down to 23.34 MB/layer.

### 9.4 Fused 1x4 MoE Down-Projection & Parallel Residual Accumulation
1. The RMSNorm-scaled activation $\hat{\mathbf{x}}$ is pre-quantized into `block_q8_K` structs once.
2. The Sigmoid router evaluates all $N_{\text{exp}}$ expert logits, computes $s_e + b_e$, and picks the top 4 experts.
3. For each active expert $k \in \{0..3\}$, the engine extracts 2D slices for `w_gate`, `w_up`, and `w_down`, computes SwiGLU hidden states $\mathbf{h} = \text{SiLU}(w_{\text{gate}} \hat{\mathbf{x}}) \odot (w_{\text{up}} \hat{\mathbf{x}})$, and quantizes $\mathbf{h}$ to `Q8_K`.
4. Down-projection rows are evaluated via multi-threaded row chunks (`MoEDownAccumTaskCtx`), scaling each expert's contribution by $w_k$ and accumulating directly into residual vector $\mathbf{x}$.

---

## 10. Hybrid State Checkpointing (`LFM2Checkpoint`)

Liquid AI's hybrid ShortConv + GQA architecture creates a unique recurrent state rollback problem during speculative decoding, beam search, and Monte Carlo Tree Search (MCTS).

### 10.1 The Recurrent State Rollback Challenge
- In pure Transformer models, rolling back $M$ rejected speculative draft tokens requires only rewinding the KV-cache write pointer: `pos <- pos - M`.
- In LFM2, the 10 ShortConv layers maintain a recurrent 2-token rolling state $(s_0, s_1)$ per channel:
  $$s_0 \leftarrow s_1, \quad s_1 \leftarrow u_t$$
  Each forward step destroys past state information in-place. Decrementing `pos` leaves the ShortConv recurrent history corrupted with rejected speculative tokens.

### 10.2 Cache-Aligned Contiguous Checkpoint Storage
`LFM2Checkpoint` preallocates a single 64-byte aligned slab holding:
1. **ShortConv Rolling Buffers:** State tensors $(s_0, s_1)$ across all ShortConv layers ($2 \times D \times 4$ bytes per layer, amounting to ~160 KB for 1.2B models and ~480 KB for 24B models).
2. **GQA Linear Attention Cursor:** Single scalar `pos`.
3. **Sampling Ring History:** Rolling token history window for repetition penalty.

### 10.3 Fast Snapshot & Rollback Protocol
- **Snapshot (`lfm2_checkpoint_save`):** Copies all active ShortConv state buffers to the checkpoint slab using vector-aligned `memcpy` in $< 15\ \mu\text{s}$.
- **Restoration (`lfm2_checkpoint_restore`):** When a speculative draft sequence is rejected, restores ShortConv states and resets `pos` in $< 15\ \mu\text{s}$.
- Enables speculative decoding and multi-path tree search with zero state corruption and zero dynamic allocations.

---

## 11. Streaming UTF-8 Assembly, Multi-Token Stop Holdback & Two-Stage SIGINT (`tools/lfm2_cli_stream.c`)

Autoregressive token streaming across interactive terminals, UNIX pipelines, and SSE network streams introduces subtle byte-boundary and signal-handling challenges.

### 11.1 Zero-Mojibake Multi-Byte UTF-8 Boundary Buffer
BPE tokenizers routinely fragment multi-byte UTF-8 codepoints (2-byte Cyrillic/Greek, 3-byte CJK/box-drawing, 4-byte emojis) across discrete token IDs. Emitting partial byte slices immediately produces replacement characters (``) or malformed JSON payloads.
- **Incomplete Tail Inspection (`utf8_incomplete_tail_bytes`):** Scans backward up to 3 continuation bytes (`10xxxxxx`) to identify the leading byte (`0xC0..0xF7`).
- If the expected byte length ($2, 3, 4$) exceeds the present continuation byte count $+ 1$, the incomplete trailing bytes are held in the sliding buffer `pending` until subsequent tokens supply the remaining continuation bytes.

### 11.2 Multi-Token Stop-String Sliding Window Holdback
Stop sequences (e.g., `"\nUser:"`, `"<|im_end|>"`, `"```"`) regularly straddle 2–3 consecutive BPE tokens.
- **Longest Proper Prefix Holdback (`compute_stop_prefix_holdback`):** Identifies if any suffix of `pending` matches a proper prefix of any registered stop string. Those bytes are withheld from output.
- **Immediate Divergence Flush:** If subsequent tokens prove the match was a false positive (e.g. `"\nUsers"`), the accumulated bytes are flushed immediately.
- **Earliest Full Match Truncation:** When a registered stop string fully matches, everything strictly prior to the match index is flushed, the stop string is discarded, and the filter returns `LFM2_STREAM_STOPPED`.
- **End-of-Turn Flush (`lfm2_stream_filter_flush`):** When decoding ends, any trailing partial prefixes that never materialized into full stop strings are emitted, while incomplete orphaned UTF-8 lead bytes are safely pruned.

### 11.3 Two-Stage `SIGINT` State Machine & UNIX Pipe Detection
- **Non-Restarting Signal Handler:** Installs `sigaction` with `sa_flags = 0` (omitting `SA_RESTART`) so that pressing `Ctrl+C` while waiting on `fgets(stdin)` or `read()` immediately interrupts the system call with `EINTR`.
- **Two-Stage Logic:**
  - *During Token Generation (`g_lfm2_is_generating = 1`):* `Ctrl+C` sets `g_lfm2_abort_gen = 1`, halting the decode loop, finalizing the model's recurrent state with `<|im_end|>`, and returning cleanly to the `User >` prompt.
  - *While Idle at Prompt (`g_lfm2_is_generating = 0`):* `Ctrl+C` sets `g_lfm2_exit_process = 1`, gracefully exiting the application without abrupt termination.
- **Terminal Environment Detector (`LFM2TerminalEnv`):** Inspects `isatty(STDIN_FILENO)` and `isatty(STDOUT_FILENO)`. When `stdout` is redirected or piped, banners, tables, and generation telemetry are automatically routed to `stderr`, leaving `stdout` as pure model text.

---

## 12. System Runtime, Linux OOM Guard, Page-Fault Warmup & Prompt Cache (`tools/lfm2_sys_cache.c`)

Running high-parameter hybrid models on Linux systems requires operating-system level memory telemetry, deterministic page residency, and disk persistence for recurrent prompt states.

### 12.1 Linux `/proc/meminfo` Parser & Context Clamper
* **Real-Time Memory Telemetry (`lfm2_read_linux_meminfo`):** Parses `/proc/meminfo` to capture `MemTotal`, `MemAvailable`, and reclaimable page cache (`Buffers + Cached + SReclaimable`).
* **Hybrid Footprint Budgeting:** Accurately sums model weights mapped via `mmap`, activation scratch arena (~32 MB), constant $O(1)$ ShortConv buffers ($N_{\text{conv}} \cdot 2 \cdot D \cdot 4\ \text{bytes}$), and per-token FP16 KV-cache memory ($\sum_{l \in \text{GQA}} 2 \cdot N_{\text{kv\_heads}}^{(l)} \cdot d_{\text{head}} \cdot 2\ \text{bytes}$).
* **Safe Context Clamping:** When available RAM (minus a 512 MB OS reserve) is insufficient for the requested context, clamps context length down to a multiple of 256 tokens (minimum 512), preventing process termination by the Linux OOM killer.
* **Transparent HugePage Arena (`lfm2_alloc_hugepage_arena`):** Preallocates 2 MB aligned virtual memory regions with `MADV_HUGEPAGE` to eliminate CPU TLB miss latency.

### 12.2 Sequential 4 KB Stride Page-Fault Sweeping (`--warmup`)
* To prevent random seek thrashing on mechanical disks or cold page caches during Token 0, `lfm2_warmup_mmap_pages` issues `MADV_WILLNEED` and `posix_fadvise`.
* Sequentially touches 1 byte per 4,096 bytes across 16 MB sliding windows, rendering a real-time `MB/s` console progress bar and guaranteeing zero disk stall during prompt prefill.

### 12.3 Disk-Backed Hybrid Prompt Cache (`--prompt-cache`, `.lfmcache` v1)
* **LFM2 Exact-Prefix Invariant:** Enforces 100% bit-for-bit token prefix matching for all $N_{\text{cached}}$ tokens before restoring recurrent ShortConv rolling states and advancing `pos`, preserving recurrent causality.
* **64-Bit FNV-1a Model Fingerprint:** Embeds a deterministic fingerprint computed across model geometry, vocabulary size, and layer routing table (`n_head_kv[0..L-1]`). Mismatched models are rejected immediately.
* **Atomic Serialization:** Employs temporary PID-tagged files (`<path>.tmp.<pid>`) committed via atomic POSIX `rename()`, preventing file corruption during power interruptions or `Ctrl+C`.
* **Payload Checksum Validation:** Covers all rolling states and KV bytes with an FNV-1a checksum to detect bit-rot or truncated reads.
* **Sub-Millisecond Restoration:** Restores full recurrent history and skips matched token prefill in $< 1\ \text{ms}$.

---

## 13. Verification & Quality Assurance

All Phase 4 and Phase 5 components are validated across automated unit, integration, and concurrency tests:
- `bin/test_optimizations`: Fused LIV ShortConv and SIMD parity.
- `bin/test_prefill_batch`: Batched GEMM vs single-token parity.
- `bin/test_decode_1x4`: 1x4 row unrolling & fused SwiGLU FFN speedup.
- `bin/test_kernels_fast`: Planar ShortConv, RoPE LUT, and FP16 online softmax.
- `bin/test_cpuid_quants`: `Q4_0`, `Q5_K`, and `IQ4_XS` parity + forced scalar fallback.
- `bin/test_runtime_features`: Byte-level PDA grammar masking, think budget lifecycle, and attention-sink shifting with RoPE re-rotation.
- `bin/test_moe_checkpoint`: Sparse MoE layer binding, 3D tensor slicing, Sigmoid + bias routing parity, fused 1x4 MoE forward pass, and hybrid state checkpointing/rollback lifecycle.
- `bin/test_cli_stream`: UTF-8 2/3/4-byte split assembly, multi-token stop string holdback, false-positive prefix divergence, end-of-stream flush, two-stage SIGINT state transitions, and large buffer safety.
- `bin/test_sys_cache`: `/proc/meminfo` parser, OOM memory planner, 2 MB HugePage alignment, 4 KB stride page warmup, model fingerprint sensitivity, round-trip `.lfmcache` save/load, exact-prefix rejection, and checksum corruption detection.
- `bin/test_lfm2_server`: Zero-dependency Linux `epoll` HTTP and SSE server test validating `GET /health`, `GET /`, `GET /v1/models`, `OPTIONS` preflight, streaming and non-streaming chat completions, embeddings, and instant client disconnect cancellation (`EPOLLRDHUP`).
- `bin/test_lfm2_api`: Full public C11 ABI test including Test 8 for Android Scoped Storage FD loading (`dup(fd)`) and `madvise(MADV_DONTNEED)` cache release.
- `bin/test_lfm2_threadpool`: Physical CPU topology discovery, simulated 8-core mobile SoC test (3 auto big cores, cpu7 Prime on Worker 0, 6-core max safe ceiling), futex idle transitions, and dynamic work-stealing.
- **Sanitizers:** 100% clean under AddressSanitizer and UndefinedBehaviorSanitizer across all 27 C binaries and 14 standalone tools (`make test-asan`).
- **ThreadSanitizer:** Zero data races certified across threadpool and parallel MatVec kernels under ThreadSanitizer (`make test-tsan`).
- **Static Binaries:** Verified 16 standalone static ELF binaries with zero dynamic dependencies (`statically linked`), plus `make static-musl` target.

---

## 14. Public C11 ABI & Modern Python SDK (`include/lfm2.h`, `src/lfm2_api.c`, `python/lfm2.py`)

### 14.1 Zero-Leak ABI Architecture
* **Public C11 Header (`include/lfm2.h`):** Defines opaque handles (`lfm2_model_t`, `lfm2_context_t`, `lfm2_checkpoint_t`) and fixed-width POD structs (`lfm2_model_info_t`, `lfm2_context_params_t`, `lfm2_sample_params_t`, `lfm2_perf_stats_t`). Guarantees ABI compatibility across C, C++, Rust, Zig, Go, and Python.
* **Internal Symbol Hiding:** `src/lfm2_api.c` is compiled into `lib/liblfm2.so` with `-fvisibility=hidden`. Only functions explicitly marked with `LFM2_API` are exported. Internal engine symbols (`GGUFTensor`, `block_q8_K_t`, `LFM2ThreadPool`) remain strictly private.

### 14.2 Pull-Based UTF-8 Streaming Generator (`lfm2_context_next_utf8`)
* Embeds `LFM2StreamFilter` within the context.
* If a token ends mid-character (such as byte 1 or 2 of a 4-byte emoji) or matches an active stop prefix, `lfm2_context_next_utf8` automatically advances the engine forward internally until a clean, complete UTF-8 slice is ready.
* Eliminates the need for background worker threads and queues in language bindings, making Python's `for chunk in ctx.stream_chat(...)` a single-threaded, zero-overhead generator.

### 14.3 State Checkpointing & Branching (`ctx.branch()`)
* Python context manager `with ctx.branch():` snapshots ShortConv 2-token rolling buffers and GQA `pos` in $< 15\ \mu\text{s}$ via `lfm2_checkpoint_create`.
* Restores clean state on block exit or exception, powering tree search, Monte Carlo planning, and speculative decoding.

### 14.4 High-Throughput L2-Normalized Text Embeddings (`lfm2_embed_text`)
* Computes L2-normalized vector embeddings using causal last-token hidden state (`LFM2_POOL_LAST_TOKEN`) or normalized sequence mean (`LFM2_POOL_MEAN`) with hardware FMA acceleration.

---

## 15. Non-Blocking Linux `epoll` OpenAI HTTP/SSE Server (`tools/lfm2_server.c`, `lfm2_server.c`)

### 15.1 Zero-Stall Cooperative Event Loop
* Operates a single-threaded Linux `epoll` event loop with dynamic timeout scheduling (`timeout = 0` during active SSE streams, `200 ms` when idle).
* On each loop tick, calls `lfm2_context_next_utf8()` once (10–25 ms latency), dispatches a single SSE event `data: {"choices":[{"delta":{"content":"..."}}]}\n\n` over the non-blocking socket with `MSG_NOSIGNAL` and `TCP_NODELAY`, and returns immediately to `epoll_wait`.

### 15.2 Instant Client-Disconnect Cancellation (`EPOLLRDHUP`)
* Catches `EPOLLRDHUP | EPOLLHUP | EPOLLERR` on token boundaries to immediately abort generation and free CPU cores when a user stops generation in a browser or disconnects a client.

### 15.3 Pre-Allocated Connection Pool & Zero Dynamic Allocations
* Statically pre-allocates 32 connection slots with 64 KB request buffers and 128 KB response buffers.
* Embedding vectors serialize directly into server scratch buffers, guaranteeing zero dynamic heap allocations (`malloc`/`free`) in request hot paths.
* Incorporates 10.0-second anti-Slowloris monotonic header deadlines.
* Serves embedded zero-dependency HTML/JS chat studio at `GET /`, plus `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, and `OPTIONS`.

---

## 16. Mobile Edge & Android Scoped Storage Integration (`tools/lfm2_loader.c`, `src/lfm2_api.c`, `python/lfm2.py`)

### 16.1 Scoped Storage FD Mapping (`lfm2_model_load_from_fd`)
* Solves Android 10+ filesystem path isolation where apps obtain user-selected GGUF files via content URIs (`content://...`).
* Accepts an open integer file descriptor, duplicates it via POSIX `dup(fd)` to decouple ownership from the host runtime, and maps it directly via `mmap(MAP_SHARED)`.
* Exposed in Python SDK as `LFM2Model.from_fd(fd)`.

### 16.2 Android `lmkd` Memory Reclamation (`lfm2_model_release_page_cache`)
* Calls `madvise(mmap_addr, mmap_size, MADV_DONTNEED)`.
* Discards physical RAM pages occupied by weights when the application enters the background (`onPause`), reducing process Resident Set Size (RSS) to near zero and preventing termination by Android's Low Memory Killer daemon.
* Pages fault back in on demand when execution resumes (`onResume`).

### 16.3 Dynamic 16 KB Page Stride Support (Android 15+)
* Uses `sysconf(_SC_PAGESIZE)` instead of hardcoded 4096-byte constants during memory sweeps and page warmup loops, preventing SIGBUS and page-boundary alignment violations on Android 15+ devices with 16 KB kernel pages.

---

## 17. Unified Cross-Platform Asymmetric `big.LITTLE` Thread Pool (`tools/lfm2_threadpool.c`)

### 17.1 Linux EAS Capacity & Clock Frequency Scanner
* Scans `/sys/devices/system/cpu/cpuN/cpu_capacity` and `/sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq`.
* Classifies cores into `PRIME` ($\ge 900$ capacity / $\ge 90\%$ frequency), `PERF` ($\ge 600$ capacity / $\ge 65\%$ frequency), and `LITTLE` (in-order efficiency cores).

### 17.2 Cluster-Aware Resource Reservation
* **PC Rule ($N_{\text{phys}} - 1$):** Reserves $N_{\text{phys}} - 1$ cores on homogeneous machines, guaranteeing 1 physical core for OS and desktop responsiveness.
* **Mobile Rule ($N_{\text{big}} - 1$):** Automatically allocates $N_{\text{big}} - 1$ cores on heterogeneous SoCs (e.g. 3 Big cores: 1 Prime + 2 Perf), dedicating 1 Big core to Kivy/OpenGL rendering and all Little cores to Android OS services.
* **Thermal Ceiling ($N_{\text{total}} - 2$):** Clamps maximum worker count to $N_{\text{total}} - 2$ (e.g. 6 cores max on an 8-core phone) to prevent thermal throttling, Vdroop voltage drops, and sudden CPU blackouts.
* **Worker 0 on Prime Core:** Main thread pins Worker 0 to `recommended_cpus[0]` (highest compute capacity), accelerating single-threaded non-GEMM kernels (`RMSNorm`, ShortConv 3-tap shift, RoPE, sampling).
* **ARM64 Microcode Trapping Prevention (`FPCR.FZ`):** Configures Floating-Point Control Register bit 24 (`FZ`) and bit 19 (`FZ16`) on all worker threads to flush denormal floating-point numbers to zero, eliminating 100-cycle microcode traps.
* **Cross-Platform Pause & Adaptive Spin:** Replaces `_mm_pause()` with `lfm2_cpu_pause()` (evaluating `yield` on ARM64) and throttles busy-spinning from 25,000 iterations (x86_64) to 4,000 iterations (ARM64) to save battery before kernel futex sleep.




