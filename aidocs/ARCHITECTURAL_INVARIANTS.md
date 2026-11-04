# Architectural Invariants & Mathematical Formulations

This document provides a formal mathematical and architectural reference for all layers, kernels, and state mechanics in LAII. Every coding agent must adhere strictly to these invariants.

---

## 1. Linear Input-Varying (LIV) Short Convolutions

ShortConv layers provide linear $O(1)$ token mixing. In LFM2-1.2B, 10 of 16 layers are ShortConv.

### 1.1 Projections & Gating Equations
Given normalized input vector $h \in \mathbb{R}^D$:
$$\begin{pmatrix} B \\ C \\ x \end{pmatrix} = W_{\text{in}} h \quad \text{where } W_{\text{in}} \in \mathbb{R}^{3D \times D}$$

1. **Gate 1 (Input-Varying Modulation):**
   $$u_t = B \odot x \quad (u_t \in \mathbb{R}^D)$$
2. **Causal Depthwise 1D Convolution (Kernel Size $K=3$):**
   $$\text{conv}_t = w_0 \odot s_{0} + w_1 \odot s_{1} + w_2 \odot u_t$$
   Where $w_0, w_1, w_2 \in \mathbb{R}^D$ are convolution channel weights, and $s_0, s_1 \in \mathbb{R}^D$ are the channel history states from steps $t-2$ and $t-1$.
3. **Destructive State Update:**
   $$s_0 \leftarrow s_1, \quad s_1 \leftarrow u_t$$
4. **Gate 2 (Output Activation Modulation):**
   $$y_t = C \odot \text{conv}_t \quad (y_t \in \mathbb{R}^D)$$
5. **Output Projection:**
   $$z_t = W_{\text{out}} y_t \quad \text{where } W_{\text{out}} \in \mathbb{R}^{D \times D}$$

### 1.2 Planar Transposition Invariant
In GGUF files, convolution weights are stored interleaved with stride 3:
$$\text{Tensor Layout: } [w_{0,0}, w_{0,1}, w_{0,2}, \dots, w_{D-1,0}, w_{D-1,1}, w_{D-1,2}]$$
* **The Optimization:** At initialization, weights are transposed into three 64-byte aligned planar arrays ($w_0[D]$, $w_1[D]$, $w_2[D]$).
* **AVX2 Loop:** Allows loading 8 consecutive channels per vector instruction (`_mm256_load_ps`) without strided gathers.

---

## 2. Grouped Query Attention (GQA) & FlashAttention-CPU

GQA layers capture long-range dependencies across positions. In LFM2-1.2B, 6 of 16 layers are GQA ($n_{\text{head\_q}} = 32, n_{\text{head\_kv}} = 8, d_{\text{head}} = 64$).

### 2.1 Rotary Positional Embedding (NeoX Half-Split)
RoPE is applied to queries and keys after per-head RMSNorm:
For coordinate $i \in [0, 31]$:
$$\theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{freq} = \text{pos} \cdot \theta_i$$
$$q_{\text{rot}}[i] = q[i] \cos(\text{freq}) - q[i + 32] \sin(\text{freq})$$
$$q_{\text{rot}}[i + 32] = q[i + 32] \cos(\text{freq}) + q[i] \sin(\text{freq})$$
* **Precomputed LUT Invariant:** Precomputed in [`LFM2RoPELUT`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_kernels_fast.c#L180) to eliminate runtime `powf`, `cosf`, and `sinf`.
* **Half-Split SIMD Invariant:** Evaluated via AVX2 `_mm256_fmsub_ps` and `_mm256_fmadd_ps`.

### 2.2 Single-Pass Online Softmax (FlashAttention-CPU)
Instead of materializing $T \times T$ attention matrix scores, the engine streams FP16 Keys and Values in a single pass over the KV-cache using running max $m$ and running normalizer $l$:
$$\alpha = \exp(m_{\text{prev}} - \max(m_{\text{prev}}, s_t)), \quad \beta = \exp(s_t - \max(m_{\text{prev}}, s_t))$$
$$l_{\text{new}} = l_{\text{prev}} \cdot \alpha + \beta$$
$$\mathbf{o}_{\text{new}} = \mathbf{o}_{\text{prev}} \cdot \alpha + \beta \cdot \mathbf{v}_t$$
Slashes KV DRAM traffic by 50% and completely eliminates intermediate DRAM score allocations.

---

## 3. Sparse Mixture-of-Experts (`lfm2moe`)

Used in `LFM2-8B-A1B` (32 experts, Top-4) and `LFM2.5-24B-A2B` (64 experts, Top-4).

### 3.1 Leading Dense Layer Rule
* **Layers 0 and 1 are ALWAYS dense SwiGLU layers** (`num_dense_layers = 2`).
* **Layers 2 through $N-1$ are Sparse MoE layers**.
* Agents must never attempt to bind MoE routing or 3D tensor slicing to layers 0 or 1.

### 3.2 Sigmoid Routing with Learned Selection Bias
Unlike Mixtral or Qwen which use Softmax over all router logits, Liquid AI uses **Sigmoid**:
1. **Router Projection:**
   $$z_e = \mathbf{W}_{\text{gate\_inp}}[e] \cdot \hat{\mathbf{x}}, \quad e \in \{0, \dots, N_{\text{exp}}-1\}$$
2. **Sigmoid Probability:**
   $$s_e = \sigma(z_e) = \frac{1}{1 + \exp(-z_e)}$$
3. **Learned Bias for Candidate Selection Only:**
   $$c_e = s_e + b_e \quad \text{where } b_e \text{ is from tensor } \texttt{blk.l.exp\_probs\_b.bias}$$
   The top $K=4$ experts with the highest $c_e$ are selected: $\{e_0, e_1, e_2, e_3\}$.
4. **Unbiased Probability Normalization:**
   The bias $b_e$ is discarded during output mixing. Unbiased sigmoid probabilities $s_{e_k}$ are normalized:
   $$w_k = \frac{s_{e_k}}{\sum_{j=0}^{K-1} s_{e_j} + 10^{-20}}$$

### 3.3 Zero-Copy 3D Tensor Slicing Over `mmap`
In GGUF files, MoE expert weights are packed as contiguous 3D tensors:
$$\text{shape} = [ne[0], ne[1], ne[2]] = [D_{\text{in}}, D_{\text{out}}, N_{\text{exp}}]$$
Because $N_{\text{exp}} = ne[2]$ is the outermost dimension in row-major order:
$$\text{byte\_offset}(e) = e \cdot ne[1] \cdot \text{row\_bytes}$$
* **Invariant:** Never duplicate, malloc, or copy expert tensors. Slice directly over the `mmap` pointer in $O(1)$ time with [`lfm2_slice_expert_tensor_2d`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c#L130).

---

## 4. Prompt Caching & Recurrent Invariants (`.lfmcache`)

### 4.1 The Exact-Prefix Rule
```text
Standard Transformer:
Prompt: [A, B, C, D, E] -> Cache saved at E.
New:    [A, B, C, X, Y] -> Valid! Can slice [A, B, C] from KV-cache.

Liquid AI (LFM2 Hybrid):
Prompt: [A, B, C, D, E] -> Cache saved at E.
ShortConv State: Reflects (D, E).
New:    [A, B, C, X, Y] -> INVALID!
ShortConv state for C requires (B, C), which was overwritten at step E.
THEREFORE: Exact-prefix match of ALL N_cached tokens is MANDATORY!
```

* **Invariant:** `lfm2_prompt_cache_try_load_prefix` validates that tokens $[0 \dots N_{\text{cached}}-1]$ in the new prompt match the cached tokens bit-for-bit.
* **Prefill Skip:** When matched, ShortConv states and the first $N_{\text{cached}}$ FP16 KV rows are restored in $< 1\ \text{ms}$, and the prefill cursor `pos` jumps straight to $N_{\text{cached}}$.

### 4.2 Atomic Cache Serialization
* Cache files must be written to `<path>.tmp.<pid>` and finalized using atomic POSIX `rename()`.
* Every cache file embeds a 64-bit FNV-1a fingerprint of model geometry and layer routing table, plus a 64-bit payload checksum over all states.

---

## 5. Hybrid State Checkpointing (`LFM2Checkpoint`)

* **Speculative Decoding & MCTS Backtracking:** When draft tokens are rejected, decremented `pos` alone does NOT restore ShortConv states.
* **Preallocated Slab:** [`LFM2Checkpoint`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c#L350) preallocates a contiguous 64-byte aligned slab holding:
  1. ShortConv rolling buffers $(s_0, s_1)$ across all ShortConv layers (~160–480 KB).
  2. GQA position scalar `pos`.
  3. Repetition penalty ring buffer.
* **Speed:** Captures and restores clean states in $< 15\ \mu\text{s}$ using vector-aligned `memcpy`.

---

## 6. Public C11 ABI & Pull-Based UTF-8 Generator Invariants

### 6.1 Opaque Encapsulation & Dynamic Symbol Hiding
* **Opaque Handles:** Pointers to `lfm2_model_t`, `lfm2_context_t`, and `lfm2_checkpoint_t` are opaque. Zero internal types (`GGUFTensor`, `block_q8_K_t`, `LFM2ThreadPool`) are declared in the public header [`include/lfm2.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2.h).
* **Symbol Hiding:** `lib/liblfm2.so` is compiled with `-fvisibility=hidden` and `LFM2_API` attribute (`__attribute__((visibility("default")))`). Only the 27 official C11 API entry points are exported.
* **Fixed-Width Structs:** All public structs use fixed-width integer and float types (`uint32_t`, `int32_t`, `float`, `uint64_t`) to ensure binary layout parity across C, C++, Rust, and Python `ctypes`.

### 6.2 Pull-Based UTF-8 Streaming Generator
* **The Callback Anti-Pattern:** A blocking callback loop (`lfm_generate(ctx, callback)`) requires foreign language wrappers (such as Python) to spin up a worker thread and a synchronized `queue.Queue` to yield tokens. This adds thread synchronization latency and complex cross-thread exception propagation.
* **Pull-Based Invariant:** `lfm2_context_next_utf8` embeds `LFM2StreamFilter` within `lfm2_context_t`. If a generated token ends mid-codepoint (e.g. byte 1 of a 4-byte UTF-8 emoji) or matches a rolling stop prefix (`"\nUser"`), `lfm2_context_next_utf8` steps internal forward passes until complete, safe UTF-8 bytes are ready. This makes Python generator streaming single-threaded, high-throughput, and mojibake-free.

### 6.3 Context Branching & State Rollback
* When performing speculative tree-search or agent exploration in Python:
  ```python
  with ctx.branch():
      # Speculative tokens generated here mutate ShortConv states & pos
      ...
  # States automatically restored on exit
  ```
* Under the hood, `ctx.branch()` calls `lfm2_checkpoint_create` on enter and `lfm2_checkpoint_restore` on exit, guaranteeing bit-exact restoration of both GQA `pos` and ShortConv 2-token rolling buffers in $< 15\ \mu\text{s}$.

### 6.4 Zero-Copy Raw Logit Distribution Inspection
* `lfm2_context_get_logits_ptr()` returns a read-only `const float*` directly pointing to `ctx->state.logits` (`n_vocab` elements).
* **Zero Allocations & Zero Copies:** Accessing logits introduces 0 heap allocations and 0 buffer copies ($O(1)$ pointer return).
* Enables custom external samplers, log-probability calculation, speculative verification, and the 5-tier parity verification suite (`tests/verify_parity.py`).

---

## 7. Non-Blocking Linux `epoll` OpenAI Server Invariants

### 7.1 Zero-Stall Cooperative Token Stepping
* **Single-Token Step Invariant:** The event loop must **never** block inside a multi-token generation loop for seconds. An active streaming client is marked `CONN_STREAMING_SSE`. On each `epoll_wait` tick, the server executes `lfm2_context_next_utf8()` **once**, flushes the SSE event with `MSG_NOSIGNAL` and `TCP_NODELAY`, and immediately re-enters `epoll_wait(0)`.
* This ensures that generation runs at 100% CPU speed while the server event loop remains non-blocking and instantly responsive to network I/O.

### 7.2 Instant Client Disconnect Cancellation (`EPOLLRDHUP`)
* If a client closes their socket, drops connection, or hits `Ctrl+C` on a streaming request, `epoll_wait` catches `EPOLLRDHUP | EPOLLHUP | EPOLLERR` on the very next token boundary ($\le 25\ \text{ms}$).
* The server calls `server_close_conn()` and resets `active_sse_slot = -1` immediately, halting generation and freeing CPU cores rather than computing tokens into a dead socket.

### 7.3 Zero-Allocation Request State Machine
* All connection buffers are pre-allocated in `LFM2HttpConn conns[32]`: 64 KB for request headers/body, 128 KB for outbound responses.
* Zero dynamic heap allocations (`malloc`/`free`) are permitted during request processing or embedding generation.
* Anti-Slowloris monotonic timeouts (10.0s) actively sweep and close stalled connections in `CONN_READING_REQ`.

---

## 8. Production Hardening & Numerical Boundary Invariants

### 8.1 Flush Denormals to Zero (`FTZ`) & Denormals Are Zero (`DAZ`)
* In recurrent convolution buffers (`conv_state`) and online softmax exponentials, floating-point activations frequently decay into the IEEE 754 subnormal/denormal range ($10^{-38}$ to $10^{-45}$).
* On modern x86_64 CPUs, arithmetic operations on denormals trigger microcode assist traps taking ~100+ cycles per instruction, causing random token latency spikes during long chats.
* **The Invariant:** `lfm2_set_denormals_zero_mode()` sets bit 15 (`FTZ`) and bit 6 (`DAZ`) in the CPU's MXCSR register via `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` and `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)`.
* This is executed at the entry point of the main thread (`main`, `lfm2_model_load`, `lfm2_context_create`, `threadpool_init`) and at the entry point of **every background pthread worker** (`lfm2_worker_entry`, `worker_thread_fn`), guaranteeing that all threads execute with zero microcode penalty assist stalls.

### 8.2 Non-256 Divisible Column Dimensions ($N_{\text{cols}} \pmod{256} \ne 0$) Tail Padding
* `block_q8_K_t` requires 256 activation floats per block. When tensor dimensions are not multiples of 256 (e.g. $N_{\text{cols}} = 100, 128, 300, 2688$):
  * The number of quantized blocks allocated is $N_b = \lfloor (N_{\text{cols}} + 255) / 256 \rfloor$.
  * For the final remainder block ($N_{\text{cols}} \pmod{256} > 0$), `lfm2_quantize_row_q8_K_avx2` copies the remaining floats into a 32-byte aligned zero-padded buffer `float pad[256]` before AVX2 block quantization.
  * This guarantees that activation elements beyond $N_{\text{cols}}$ are quantized to strictly 0 with precomputed `bsums` matching the quantized sub-block slice, preventing out-of-bounds page faults and uninitialized memory traps in dot products.
* In dot product kernels (`lfm2_vec_dot_q4_K_q8_K_avx2`, `lfm2_vec_dot_q6_K_q8_K_avx2`), loops iterate over all $N_b$ blocks. In `lfm2_vec_dot_q8_0_q8_K_avx2`, the sub-block loop clamps at $(N_{\text{cols}} - b \cdot 256 + 31) / 32$ to prevent reading past the $(N_{\text{cols}} + 31) / 32$ blocks allocated for `block_q8_0_t`.



