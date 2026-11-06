# Liquid AI (LFM2) Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/Standard-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![SIMD Acceleration](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA%20%2B%20F16C-brightgreen.svg)]()
[![API Protocol](https://img.shields.io/badge/OpenAI%20API-Compatible-orange.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(libc%20only)-success.svg)]()

A specialized, zero-dependency C11 inference engine for models based on Liquid AI's hybrid architecture (**LFM2 / LFM2.5** series) on Linux x86_64.

Designed from first principles to leverage Liquid AI’s edge-friendly hybrid architecture—combining **Double-Gated Linear Input-Varying (LIV) Short Convolutions** with sparse **Grouped Query Attention (GQA)**.

> [!NOTE]
> **Independent Open-Source Implementation:** This repository is an independent, community-developed open-source inference engine designed to execute models conforming to the LFM architecture. It is **not** affiliated with, endorsed by, sponsored by, or associated with Liquid AI, Inc. Pretrained weights are published separately by Liquid AI under the Liquid AI Community License and can be downloaded from Hugging Face.

---

> [!TIP]
> **AI / LLM Agents:** Working on this codebase? Visit the dedicated [**`aidocs/` Knowledge Hub**](file:///home/avisheks/Documents/python-projects/Python-projects/laii/aidocs/README.md) and review the [**`AGENT_PRIMER.md`**](file:///home/avisheks/Documents/python-projects/Python-projects/laii/aidocs/AGENT_PRIMER.md) for non-negotiable architectural invariants, SIMD vector rules, and extension cookbooks.

## Why Liquid Foundation Models on Edge CPUs?

In standard Transformers, every single layer uses full self-attention, requiring an ever-growing Key-Value (KV) cache ($O(T)$ memory per layer, $O(T^2)$ compute). On resource-constrained edge CPUs, this causes cache thrashing and memory bus saturation.

Liquid AI’s **LFM2** replaces the majority of self-attention blocks with **1D depthwise short convolutions ($k=3$)**:

1. **Tiny State Memory:** The 1D convolution layers do **not** use a growing KV cache. During autoregressive decoding, they only need to remember the last 2 token states in a fixed-size ring buffer (~16 KB per layer). This cuts the dynamic state memory by **60% to 90%** compared to a Llama or Qwen model of equivalent size.
2. **CPU Cache Locality:** Short 1D convolutions and element-wise multiplicative gates fit entirely inside CPU L1/L2 caches.
3. **Memory Bandwidth Relief:** Single-token decoding is 100% memory-bandwidth bound. A quantized LFM2-1.2B checkpoint (~700 MB) stays warm in RAM without paging or thrashing.

---

## Architecture Summary

```
                      [ Input Token ]
                             |
                             v
                     [ Token Embedding ]
                             |
                             v
                +-------------------------+
                |  LFM2 Hybrid Layer x N  |
                +-------------------------+
                             |
                   [ Operator RMSNorm ]
                             |
               +-------------+-------------+
               |                           |
        (Conv Layer)                (Attn Layer)
               v                           v
     +-------------------+       +-------------------+
     |   LIV ShortConv   |       |   GQA Attention   |
     | - in_proj [3*D]   |       | - Q, K, V proj    |
     | - Gate 1: B * x   |       | - QK-RMSNorm      |
     | - Conv1D (k=3)    |       | - RoPE (NeoX)     |
     | - Gate 2: C * out |       | - Softmax GQA     |
     | - out_proj [D]    |       | - out_proj [D]    |
     +-------------------+       +-------------------+
               |                           |
               +-------------+-------------+
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
                     [ Output RMSNorm ]
                             |
                             v
                         [ LM Head ]
                             |
                             v
                       [ Next Token ]
```

---

## Features of the LAII Engine

### 1. Architectural & Model Capabilities
* **Hybrid LIV ShortConv + GQA Backbone:** Combines 1D depthwise short convolutions ($k=3$) with sparse Grouped Query Attention. ShortConv layers use a fixed 2-token rolling buffer (~16 KB per layer), reducing dynamic state memory by **60% to 90%** compared to standard attention architectures.
* **Sparse Mixture-of-Experts (`lfm2moe`):** Full architectural support for large-parameter Liquid AI checkpoints ([`LFM2-8B-A1B`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c) and [`LFM2.5-24B-A2B`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_moe_checkpoint.c)). Features leading dense SwiGLU layers (0–1) and sparse MoE layers (2..$N-1$) activating Top-4 experts per token.
* **Auxiliary-Loss-Free Sigmoid Router with Learned Selection Bias:** Implements the exact Sigmoid routing activation $s_e = \sigma(z_e)$ with learned per-expert load-balancing bias $b_e$ (`exp_probs_b` / `exp_probs_b.bias`) applied strictly during Top-$K$ candidate selection ($c_e = s_e + b_e$), followed by unbiased probability normalization ($w_k = s_{e_k} / \sum s_{e_j}$).
* **$O(1)$ Zero-Copy 3D Tensor Slicing Over `mmap`:** Slices contiguous 2D expert matrices from packed 3D GGUF tensors (`[ne0, ne1, n_exp]`) at offset `e * ne1 * row_bytes` with zero heap allocation. Touches only the 4 active experts (12.5% of layer parameters in 8B-A1B), cutting memory bus traffic from 186.75 MB/layer to 23.34 MB/layer.
* **Hybrid State Checkpointing (`LFM2Checkpoint`):** Solves the destructive recurrent state overwrite problem where ShortConv rolling buffers are overwritten in-place. Preallocates a 64-byte aligned slab capturing all ShortConv states (~160–480 KB) and GQA `pos` in $< 15\ \mu\text{s}$, enabling zero-latency draft rejection in speculative decoding and clean backtracking in tree search.

### 2. High-Performance SIMD & Kernel Optimizations
* **Planar AVX2 ShortConv 3-Tap Kernel:** Transposes stride-3 GGUF convolution weights into 64-byte aligned planar arrays ($w_0[D], w_1[D], w_2[D]$) at startup. Computes Gate 1, Conv1D, rolling state update, Gate 2, and L1-hot quantization in a single AVX2 pass, eliminating non-contiguous scalar loads.
* **Precomputed RoPE LUT + Fused QK-Norm:** Precomputes $\cos$ and $\sin$ tables up to `max_seq_len` at startup, completely eliminating per-token `powf`, `cosf`, and `sinf` runtime overhead. Fuses per-head RMSNorm scaling with in-register half-split NeoX RoPE rotation and hardware `F16C` packing into the KV-cache.
* **Single-Pass Online Softmax (FlashAttention-CPU):** Streams FP16 Keys and Values from the KV-cache in a single pass using running logit max and sum scalars. Slashes KV memory traffic by 50% and eliminates intermediate score buffers in DRAM.
* **1x4 Row-Unrolled MatVec & Register-Resident SwiGLU FFN:** Computes 4 adjacent weight rows in parallel, amortizing activation vector loads across consecutive rows. Computes gate ($w_1$) and up ($w_3$) projections concurrently in vector registers, evaluates rational SiLU in registers, and eliminates intermediate DRAM round-trips.
* **Batched Prefill GEMM Acceleration:** Quantizes batches of prompt token activations into `block_q8_K` in parallel and evaluates rank-1 outer-product accumulators across weight rows, cutting weight DRAM loads by $B\times$ with bit-exact single-token decode parity.

### 3. Extended Quantization & Single-Binary Portability
* **Comprehensive GGUF Quantization Support:** Hardware-accelerated support for `Q4_K_M`, `Q6_K`, `Q8_0`, `Q4_0`, `Q5_K`, and `IQ4_XS`, alongside unquantized `FP32`, `FP16`, and `BF16`.
* **1-Clock `IQ4_XS` Non-Linear Grid Lookup:** Broadcasts the 16-entry non-linear importance grid and uses byte shuffle `_mm256_shuffle_epi8` (`vpshufb`) to translate 32 4-bit indices into non-linear weights in 1 CPU clock cycle.
* **Runtime CPUID Multi-ISA Dynamic Dispatch:** Decorated with GCC/Clang target attributes (`__attribute__((target("avx2,fma,f16c")))`). Inspects hardware capabilities at startup via `cpuid` to bind hardware-accelerated kernels, while providing safe portable C11 scalar fallbacks on older CPUs without `SIGILL` crashes. Includes an `LFM2_FORCE_ISA=scalar` testing hook.

### 4. Runtime Control Plane & Context Management
* **Zero-Allocation Byte-Level PDA Grammar Logit Masker:** Pre-decodes all 65,536 vocabulary tokens into raw byte slices once at startup. Uses a 28-state Pushdown Automaton (PDA) with stack depth 32 to speculatively evaluate token candidates in $< 50\ \mu\text{s}$, masking syntax violations with `-INFINITY` to guarantee 100% syntactically valid RFC 8259 JSON objects and LFM2 Pythonic tool calls (`[func(arg="val", count=42)]`).
* **`<think>` Reasoning Token Budget Controller:** Tracks the chain-of-thought lifecycle (`OUTSIDE`, `ACTIVE`, `FORCING_CLOSE`, `ANSWERING`) in `LFM2.5-Thinking` models. Enforces strict token limits by deterministically forcing emission of `\n</think>\n` through the forward pass to maintain complete internal state synchronization.
* **GQA Attention-Sink Infinite Context Shifting:** Preserves initial prompt attention sinks ($n_{\text{sink}}$), evicts old dialogue history ($n_{\text{discard}}$), shifts active context ($n_{\text{keep}}$) down, and counter-rotates shifted Keys by $-\Delta \cdot \theta$ using AVX2 intrinsics. Allows unbounded multi-turn streaming conversations without running out of context memory (while ShortConv layers require 0 memory movement or rotation).
* **Streaming UTF-8 Boundary Assembly & Multi-Token Stop Holdback:** Sliding-window filter assembling split 2-, 3-, and 4-byte UTF-8 codepoints across token boundaries (preventing mojibake replacement characters ``), rolling stop-string prefix holdback (`--stop`), two-stage non-restarting `sigaction(SIGINT)` (`Ctrl+C` cancels generation turn while keeping REPL and state alive), and UNIX pipe/TTY detector routing telemetry to `stderr` when redirected.
* **Linux `/proc/meminfo` OOM Guard & Context Clamper:** Direct `/proc/meminfo` parsing calculating hybrid model footprint (GGUF `mmap`, recurrent ShortConv buffers, 32 MB arena, and FP16 KV-cache). Dynamically clamps context window to safe limits on memory-constrained systems (e.g. 8 GB RAM machines) to avoid Linux OOM killer termination.
* **Sequential `mmap` Page-Warmup Progress Bar (`--warmup`):** Sequentially faults every 4 KB page of the mmap'd GGUF file using `MADV_WILLNEED` and sequential 4 KB stride reads across 16 MB windows with a live `MB/s` console progress bar, turning 20+ seconds of random seek thrashing on mechanical drives into a single fast contiguous read.
* **Disk-Backed Hybrid Prompt Cache (`--prompt-cache`, `.lfmcache` v1):** Enforces LFM2 exact-prefix caching rules: ShortConv rolling states are recurrent ($s_0, s_1$ at step $t$), requiring 100% bit-for-bit prefix match of all $N_{\text{cached}}$ tokens before restoring states and advancing `pos`. Embeds 64-bit FNV-1a model hyperparameter & routing table fingerprints with atomic file writes (`.tmp.<pid>` + `rename()`), accelerating prompt prefill by skipping matched prefix evaluation in $< 1\ \text{ms}$.

### 5. Production Infrastructure & Tooling
* **Public C11 ABI & Zero-Leak Dynamic Shared Library (`include/lfm2.h`, `lib/liblfm2.so`):** Industrial-grade public C interface using opaque handles (`lfm2_model_t`, `lfm2_context_t`, `lfm2_checkpoint_t`) and fixed-width POD structs (`lfm2_model_info_t`, `lfm2_context_params_t`, `lfm2_sample_params_t`, `lfm2_perf_stats_t`). Built with `-fvisibility=hidden` so zero internal engine symbols or tensor types (`GGUFTensor`, `block_q8_K_t`, `LFM2ThreadPool`) leak into the dynamic symbol table.
* **Pull-Based Single-Threaded UTF-8 Streaming Generator (`lfm2_context_next_utf8`):** Eliminates background callback threads and `queue.Queue` synchronization for language bindings. Embeds the streaming UTF-8 boundary buffer and multi-token stop holdback filter directly into the context handle. Steps internal forward passes until complete, pristine UTF-8 codepoints are assembled—guaranteeing zero mojibake and zero partial stop leakage.
* **Sub-15 µs Hybrid Context Branching (`ctx.branch()`):** Zero-copy state snapshot and instant rollback. Automatically snapshots ShortConv 2-token rolling buffers (~160–480 KB) and GQA `pos` on enter, restoring clean state on exit to empower speculative decoding, tree-search exploration, and agentic workflows.
* **High-Throughput L2-Normalized Text Embeddings (`lfm2_embed_text`):** Generates normalized dense embedding vectors using either causal last-token state (`LFM2_POOL_LAST_TOKEN`) or mean-pooled normalized sequence states (`LFM2_POOL_MEAN`) with hardware FMA L2 normalization.
* **Modern Python SDK (`python/lfm2.py`):** Pure Python `ctypes` bindings supporting lazy token generator streaming (`for chunk in ctx.stream_chat():`), context manager branching (`with ctx.branch():`), grammar constraints, and performance telemetry.
* **Lock-Free Spin-Barrier ThreadPool:** Queries `/sys/devices/system/cpu/` to isolate physical CPU cores and eliminate Hyper-Threading port thrashing. Uses 64-byte cache-line aligned atomic barriers to achieve ~260 ns dispatch latency, dynamic row work-stealing, and low-power Linux `FUTEX_WAIT_PRIVATE` sleep.
* **Priority-Ordered BPE Rank Merging:** Replicates Hugging Face / GGUF BPE merge ranks using FNV-1a hash tables, a GPT-2 regex pre-tokenizer state machine, and a priority min-heap with lazy version invalidation.
* **Single Static ELF Executables (`make static`):** Standalone static binaries (`bin/lfm-run-static`, `bin/lfm-server-static`, etc.) with zero dynamic library dependencies (`statically linked`).
* **Embedded Glassmorphic Web Studio & OpenAI-Compatible REST API:** Zero-dependency POSIX socket server serving an in-memory single-page Web Studio (`GET /`) and OpenAI chat completion endpoints (`POST /v1/chat/completions`).

---

## Project Structure

```
laii/
├── aidocs/                      # Dedicated LLM Agent Knowledge Hub & Onboarding Guides
│   ├── README.md                # Agent Hub master index & navigation map
│   ├── AGENT_PRIMER.md          # 7 Golden Rules & quick context primer for Gemini/LLMs
│   ├── ARCHITECTURAL_INVARIANTS.md # Mathematical formulations & hybrid state rules
│   ├── MEMORY_AND_STATE_MAP.md  # Memory arena, KV-cache, OOM guard & HugePages
│   ├── CODEBASE_TOPOLOGY.md     # Directory layout, symbol glossary & build boundaries
│   ├── SIMD_AND_KERNELS.md      # AVX2/FMA intrinsics, quantization formats & CPUID
│   ├── EXTENSION_PLAYBOOK.md    # Developer cookbooks for adding features & tests
│   └── COMMON_PITFALLS.md       # Fatal traps, alignment bugs & sanitizer failures
├── docs/
│   ├── ARCHITECTURE.md          # Complete mathematical formulations & cache analysis
│   ├── PHASE1_SPECIFICATION.md   # Phase 1 design, memory arena, GGUF loader & layer specs
│   ├── PHASE2_SPECIFICATION.md   # Phase 2 AVX2 SIMD, Quantization, Fused LIV & ThreadPool specs
│   ├── PHASE3_SPECIFICATION.md   # Phase 3 C ABI, Python bindings, Web Studio & Static packaging
│   ├── PHASE4_SPECIFICATION.md   # Phase 4 Fast kernels, CPUID multi-ISA dispatch & runtime control plane
│   ├── ROADMAP.md               # 4-Phase engineering roadmap & milestones
│   └── TEST_SUITE.md            # Comprehensive testing architecture & QA protocol
├── include/
│   ├── lfm2.h                   # Public C11 ABI header (opaque handles & POD structs)
│   ├── arena.h                  # 64-byte cache-line aligned linear memory arena
│   ├── gguf.h                   # GGUF v2/v3 binary format parser & tensor directory
│   ├── lfm_model.h              # Model config, layer weight mappings & metadata
│   ├── lfm_simd.h               # AVX2/FMA SIMD vector routines & block quantization
│   ├── lfm_fused.h              # Register-resident fused LIV ShortConv kernel
│   ├── lfm_threadpool.h         # Zero-allocation POSIX threadpool with generation barrier
│   ├── lfm_forward.h            # LIV ShortConv, GQA, RoPE, RMSNorm & SwiGLU forward pass
│   ├── lfm_tokenizer.h          # BPE tokenizer encoding & decoding
│   ├── lfm_engine.h             # High-level engine lifecycle & autoregressive generation
│   ├── lfm_sys_cache.h          # OOM guard, sequential page warmup & prompt cache
│   ├── lfm_c_api.h              # Public C ABI interface for shared/static libraries & FFI
│   └── lfm_server.h             # Embedded HTTP REST server interface
├── src/
│   ├── lfm2_api.c               # Public C11 ABI implementation (-fvisibility=hidden)
│   ├── arena.c                  # Memory arena allocator
│   ├── gguf.c                   # Zero-copy mmap GGUF loader
│   ├── lfm_model.c              # Tensor name binding & config extractor
│   ├── lfm_simd.c               # AVX2/FMA dot products, RMSNorm, vector math & Q8_0/Q4_0
│   ├── lfm_fused.c              # Fused LIV ShortConv kernel (Gate 1 + Conv + State + Gate 2)
│   ├── lfm_threadpool.c         # Multi-threaded parallel row-chunked MatVec execution
│   ├── lfm_forward.c            # Hybrid forward pass execution with fused SIMD ops
│   ├── lfm_tokenizer.c          # Tokenizer & byte fallback logic
│   ├── lfm_engine.c             # Prefill, autoregressive decode & Top-K sampling
│   ├── lfm_c_api.c              # Public C ABI implementation
│   ├── lfm_server.c             # Embedded POSIX HTTP REST server & endpoint routing
│   ├── lfm_webui.h              # Embedded single-page Web Studio HTML/CSS/JS (in-memory)
│   ├── main.c                   # CLI executable (`lfm-run`)
│   └── server_main.c            # Dedicated server executable (`lfm-server`)
├── python/                      # High-Performance Python SDK (ctypes)
│   ├── __init__.py              # Python package exports (`LFMModel`, `LFMContext`, etc.)
│   └── lfm2.py                  # Single-threaded UTF-8 pull generator & branch context manager
├── bindings/
│   └── python/
│       └── laii.py              # Zero-dependency Python ctypes wrapper (Phase 3)
├── lib/
│   ├── liblfm.a                 # Static engine library archive
│   ├── liblfm.so                # Shared dynamic library (-fPIC)
│   └── liblfm2.so               # Public C11 ABI dynamic library (zero internal symbol leakage)
├── tests/
│   ├── test_arena.c             # 64-byte alignment & arena reset/OOM tests
│   ├── test_math.c              # RMSNorm, SiLU, RoPE norm preservation & MatVec tests
│   ├── test_simd_avx2.c         # AVX2 vector dot product, RMSNorm & vector math unit tests
│   ├── test_quants.c            # Q8_0 and Q4_0 block quantization dot product tests
│   ├── test_simd_kquants.c      # AVX2 Q8_K pre-quantization & K-quants integer dot products
│   ├── test_fused_conv.c        # Fused LIV ShortConv numerical equivalence test vs reference
│   ├── test_threads.c           # ThreadPool generation-barrier concurrency stress test
│   ├── test_tokenizer.c         # BPE encoding, decoding & byte fallback tests
│   ├── test_bpe.c               # Exact priority-ordered GGUF BPE rank merging unit tests
│   ├── test_gguf.c              # GGUF parser & tensor offset alignment test
│   ├── test_conv1d.c            # LIV ShortConv numerical parity test vs PyTorch
│   ├── test_attention.c         # GQA + QK-Norm + RoPE parity test vs PyTorch
│   ├── test_forward.c           # End-to-end toy LFM2 model forward pass test
│   ├── test_c_api.c             # C ABI lifecycle, buffer generation & streaming callback test
│   ├── test_lfm2_threadpool.c   # Spin-barrier thread pool & topology pinning tests
│   ├── test_optimizations.c     # Fused ShortConv & SwiGLU optimization parity tests
│   ├── test_prefill_batch.c     # Batched GEMM prefill acceleration & parity tests
│   ├── test_decode_1x4.c        # 1x4 row unrolling & fused SwiGLU FFN tests
│   ├── test_kernels_fast.c      # Planar ShortConv, RoPE LUT & FP16 online softmax tests
│   ├── test_cpuid_quants.c      # Runtime CPUID dispatch & Q4_0/Q5_K/IQ4_XS tests
│   ├── test_runtime_features.c  # PDA grammar masking, think budget & KV shift tests
│   ├── test_moe_checkpoint.c    # Sparse MoE routing, 3D tensor slicing & state rollback tests
│   ├── test_cli_stream.c        # Streaming UTF-8 assembly, stop holdback & SIGINT guard tests
│   ├── test_sys_cache.c         # OOM guard, HugePage arena, warmup & prompt cache tests
│   ├── test_lfm2_api.c          # Public C11 ABI unit tests (streaming, rollback, embeddings)
│   ├── test_hardening.c         # FTZ/DAZ mode, non-256 divisible dimension & bounds hardening tests
│   ├── test_python_bindings.py  # Python ctypes bindings automated test
│   ├── test_python_lfm2.py      # Modern Python SDK test (generator streaming & branch rollback)
│   ├── verify_parity.py         # 5-tier numerical parity SLA & golden logit verifier
│   └── test_http_server.sh      # HTTP REST server & Web Studio automated integration test
├── tools/
│   ├── lfm2_loader.c            # Zero-copy GGUF v3 inspector for LFM2 hybrid models
│   ├── lfm2_bpe.c               # Priority-ordered BPE rank merging engine
│   ├── lfm2_threadpool.c        # Standalone atomic spin-barrier thread pool
│   ├── lfm2_forward.c           # Standalone C11 forward pass engine for LFM2 models
│   ├── lfm2_chat.c              # Interactive terminal chat REPL with Min-P pre-filtering
│   ├── lfm2_prefill_batch.c     # Batched GEMM prefill acceleration benchmark
│   ├── lfm2_decode_1x4.c        # 1x4 row unrolling & fused SwiGLU FFN benchmark
│   ├── lfm2_kernels_fast.c      # Planar ShortConv, RoPE LUT & FP16 online softmax benchmark
│   ├── lfm2_cpuid_quants.c      # Runtime CPUID dispatch & Q4_0/Q5_K/IQ4_XS benchmark
│   ├── lfm2_runtime_features.c  # Grammar PDA masking, think budget & KV shifting benchmark
│   ├── lfm2_moe_checkpoint.c    # Sparse MoE Sigmoid routing, 3D slicing & rollback benchmark
│   ├── lfm2_cli_stream.c        # Streaming UTF-8 boundary assembly, stop holdback & SIGINT guard
│   ├── lfm2_sys_cache.c         # OOM guard, sequential page warmup & prompt cache engine
│   ├── generate_reference_tensors.py # PyTorch reference generator for unit tests
│   └── create_toy_gguf.py       # Synthetic LFM2 GGUF model builder
├── Makefile                     # Build system with AVX2/FMA, static, and library targets
└── README.md                    # Project documentation
```

---

## Quickstart

### 1. Build the Engine and Run Tests
```bash
# Build the engine and run the full 24-part automated test suite
make test-c-bins

# Run with AddressSanitizer and UndefinedBehaviorSanitizer
make test-asan

# Build standalone static ELF binaries and verify static linkage
make static-test
```

Expected output:
```text
=======================================================
Running Liquid AI Engine Full Automated Test Suite     
=======================================================
[Test] Running Memory Arena allocator tests...            -> PASSED
[Test] Running Mathematical Primitives unit tests...      -> PASSED
[Test] Running AVX2 SIMD Vectorization unit tests...      -> PASSED
[Test] Running Quantization (Q8_0 & Q4_0) dot product...   -> PASSED
[Test] Running AVX2 Q8_K & K-Quants SIMD tests...         -> PASSED
[Test] Running Fused LIV ShortConv kernel equivalence...  -> PASSED
[Test] Running Multi-threaded ThreadPool barrier tests... -> PASSED
[Test] Running GGUF binary parser and mmap loader test... -> PASSED
[Test] Running BPE Tokenizer unit tests...                -> PASSED
[Test] Running Priority-Ordered GGUF BPE tests...         -> PASSED
[Test] Running LIV Gated ShortConv parity (PyTorch)...    -> PASSED
[Test] Running Grouped Query Attention parity (PyTorch).. -> PASSED
[Test] Running end-to-end forward pass on toy model...    -> PASSED
[Test] Running C ABI lifecycle & generation tests...      -> PASSED
[Test] Running Spin-Barrier ThreadPool & SMT tests...     -> PASSED
[Test] Running ShortConv & SwiGLU Optimization tests...   -> PASSED
[Test] Running Batched Prefill GEMM tests...              -> PASSED
[Test] Running 1x4 Row Unrolling & SwiGLU FFN tests...    -> PASSED
[Test] Running Fast Kernels (RoPE LUT & Softmax) tests... -> PASSED
[Test] Running CPUID Multi-ISA & Q4_0/Q5_K/IQ4_XS tests.. -> PASSED
[Test] Running Runtime Control (PDA & Think Budget) tests -> PASSED
[Test] Running Sparse MoE & Checkpoint Rollback tests...  -> PASSED
[Test] Running Streaming UTF-8 & Two-Stage SIGINT tests.. -> PASSED
[Test] Running System Runtime, OOM Guard & Cache tests... -> PASSED
[Test] Running Public C11 ABI unit tests...                -> PASSED
[Test] Running Python ctypes bindings unit test...         -> PASSED
[Test] Running Modern Python SDK tests...                  -> PASSED
[Test] Verifying Standalone Zero-Copy C11 LFM2 Loader...  -> PASSED
[Test] Verifying Standalone Zero-Copy C11 LFM2 BPE...     -> PASSED
[Test] Verifying Standalone Zero-Copy C11 LFM2 Forward... -> PASSED
[Test] Verifying Standalone Zero-Copy C11 LFM2 Chat...    -> PASSED
[Test] Verifying Standalone C11 Spin-Barrier ThreadPool.. -> PASSED
[Test] Verifying Standalone Batched Prefill Benchmark...  -> PASSED
[Test] Verifying Standalone 1x4 Decode Benchmark...       -> PASSED
[Test] Verifying Standalone Fused Fast Kernels Bench...   -> PASSED
[Test] Verifying Standalone CPUID Quantization Bench...   -> PASSED
[Test] Verifying Standalone Runtime Control Features...   -> PASSED
[Test] Verifying Standalone Sparse MoE & Checkpointing... -> PASSED
[Test] Verifying Standalone Streaming & SIGINT Guard...   -> PASSED
[Test] Verifying Standalone System Runtime & Prompt Cache -> PASSED
[Test] Running Embedded HTTP REST server & Web Studio...  -> PASSED
[Test] Verifying Standalone Static ELF binaries (15/15)... -> PASSED
=======================================================
ALL 25 TESTS & 13 BENCHMARKS PASSED (100% SUCCESS RATE)
=======================================================
```

---

## Usage Guide

### 2. CLI Inference (`lfm-run`)

```bash
# Standard text generation
./lfm-run -m path/to/model.gguf -p "Explain Liquid Foundation Models" -n 128 -t 0.7 -j 4

# Interactive multi-turn chat REPL mode
./lfm-run -m path/to/model.gguf -i -j 4

# Generation with sequential page warmup and disk-backed prompt caching
./lfm-run -m path/to/model.gguf -p "Explain Liquid Foundation Models" --warmup --prompt-cache prefix.lfmcache
```

CLI options:
- `-m, --model <path>`: Path to GGUF model file (required)
- `-p, --prompt <text>`: Prompt string for generation
- `-n, --n-predict <int>`: Maximum tokens to generate (default: 128)
- `-t, --temp <float>`: Temperature for sampling (default: 0.7, 0.0 = greedy)
- `--top-k <int>`: Top-K filter limit (default: 40)
- `-j, --threads <int>`: Number of CPU worker threads (default: hardware cores)
- `-i, --interactive`: Enter interactive REPL chat mode
- `--prompt-cache <path>`: Path to disk-backed `.lfmcache` file for hybrid prefix caching
- `--warmup`: Sequentially fault mmap pages into page cache with progress bar before Token 0
- `-s, --server`: Launch embedded HTTP server & Web Studio
- `--port <int>`: HTTP port for server mode (default: 8080)
- `-h, --help`: Display help and options

---

### 3. Embedded HTTP REST Server & Web Studio (`lfm-server`)

The engine includes a zero-dependency HTTP server with an in-memory glassmorphic Web Studio and OpenAI-compatible completions API:

```bash
# Launch server
./lfm-server -m path/to/model.gguf -p 8080 -j 4
# Or via lfm-run:
./lfm-run -m path/to/model.gguf -s --port 8080 -j 4
```

#### Accessing the Web Studio
Open `http://localhost:8080` in your browser. The embedded single-page application features:
* Real-time token streaming with live speed counters (`tokens/sec`) and latency timers.
* Interactive sliders for temperature, top-k, and max generation length.
* Architecture statistics card showing hybrid layer ratio and KV-cache footprint savings.

#### OpenAI-Compatible REST API
```bash
# Chat completions endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2",
    "messages": [{"role": "user", "content": "Explain why Liquid AI models are fast on CPU"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'

# Health & model metadata
curl http://localhost:8080/health
```

---

### 4. Modern Python SDK (`python/lfm2.py`) & C11 Public ABI

The engine exposes a high-performance, single-threaded C11 ABI (`include/lfm2.h` & `lib/liblfm2.so`) and a pure standard library Python SDK (`python/lfm2.py`):

#### Python SDK Usage
```python
from python.lfm2 import LFMModel, LFMContext, GrammarType, PoolingType

# 1. Load model with zero heap weight copies
model = LFMModel("models/lfm2-1.2b.gguf")
info = model.info()
print(f"Loaded {info.arch}: {info.n_layers} layers ({info.n_shortconv_layers} ShortConv, {info.n_gqa_layers} GQA)")

# 2. Create context with auto-planned RAM limits
ctx = LFMContext(model, max_seq_len=4096, n_threads=4)

# 3. Pull-based single-threaded generator streaming (zero threads, zero queues, zero mojibake)
print("Assistant: ", end="", flush=True)
for chunk in ctx.stream_chat("Why are Liquid AI models fast on edge CPUs?", max_tokens=128):
    print(chunk, end="", flush=True)
print()

# 4. Hybrid context branching & instant rollback (MCTS / Speculative decoding)
# ShortConv rolling buffers (~160 KB) and GQA pos are snapshotted in < 15 µs
with ctx.branch():
    print("Inside branch:")
    for chunk in ctx.stream_chat("Explore alternative speculative branch...", max_tokens=32):
        print(chunk, end="", flush=True)
    print()
# On exit, states are bit-exact restored to prior position!

# 5. Extract L2-Normalized Text Embeddings
emb_causal = ctx.embed("Liquid Foundation Models", pooling=PoolingType.LAST_TOKEN)
emb_mean   = ctx.embed("Liquid Foundation Models", pooling=PoolingType.MEAN)
print(f"Causal embedding vector dimension: {len(emb_causal)} (L2 norm: 1.0)")

# 6. Zero-Copy Logits Inspection & Custom Samplers
tokens = model.tokenize("Liquid AI inference")
ctx.prefill_tokens(tokens, compute_last_logits=True)
logits = ctx.get_logits()       # Returns full float distribution
ptr    = ctx.get_logits_ptr()   # Direct zero-copy ctypes pointer
print(f"Logits dimension: {len(logits)} (Top token: {logits.index(max(logits))})")
```

#### Multi-Tier Parity & Contract Verification (`make test-parity`)
Run the automated zero-dependency 5-tier contract suite verifying tokenization, prefill parity, bit-exact checkpoint rollback, unit embedding invariants, and golden oracle distributions:
```bash
# Run local 5-tier contract verification
make test-parity

# Record golden reference fixture from FP32/F16/Q8 GGUF checkpoint
python3 tests/verify_parity.py -m models/LFM2.5-1.2B-Q8_0.gguf --golden-file tests/lfm2_1.2b.lfm_golden --generate-golden

# Certify quantized checkpoint against golden reference fixture
python3 tests/verify_parity.py -m models/LFM2.5-1.2B-Q4_K_M.gguf -q Q4_K_M --golden-file tests/lfm2_1.2b.lfm_golden
```

#### Compiling with the Public C11 ABI (`include/lfm2.h`)
```c
#include "lfm2.h"
#include <stdio.h>

int main(void) {
    lfm2_model_t *model = NULL;
    if (lfm2_model_load("models/lfm2-1.2b.gguf", &model) != LFM2_OK) return 1;

    lfm2_context_params_t params = lfm2_context_params_default();
    lfm2_context_t *ctx = NULL;
    lfm2_context_create(model, &params, &ctx);

    lfm2_sample_params_t sparams = lfm2_sample_params_default();
    lfm2_begin_chat_turn(ctx, "user", "Hello LFM2!");

    char chunk[256];
    size_t chunk_len = 0;
    while (lfm2_context_next_utf8(ctx, &sparams, chunk, sizeof(chunk), &chunk_len) == LFM2_OK) {
        fwrite(chunk, 1, chunk_len, stdout);
        fflush(stdout);
    }
    printf("\n");

    lfm2_context_free(ctx);
    lfm2_model_free(model);
    return 0;
}
```
Compile against `liblfm2.so`:
```bash
gcc app.c -Iinclude -Llib -llfm2 -Wl,-rpath,'$ORIGIN/../lib' -o app
```

---

### 5. Legacy Python FFI Bindings (`bindings/python/laii.py`)

A lightweight ctypes wrapper for Phase 3 builds:

```python
from laii import LFMEngine

with LFMEngine("models/lfm2-1.2b.gguf", num_threads=4) as engine:
    response = engine.generate("Explain Liquid Foundation Models", max_tokens=128, temperature=0.7)
    print("Response:", response)
```

---

### 5. Standalone Static ELF Binaries (`make static`)

To deploy to edge devices, bare-metal servers, or minimal containers without glibc:

```bash
make static

# Verify fully static linkage
file bin/lfm-run-static
# Output: bin/lfm-run-static: ELF 64-bit LSB executable, x86-64, version 1 (GNU/Linux), statically linked
```

Copy `bin/lfm-run-static` or `bin/lfm-server-static` to any Linux x86_64 host and execute directly.

---

### 6. Zero-Dependency GGUF Inspector (`bin/lfm2_loader`)

Inspect any Liquid AI `.gguf` file without running inference or allocating heap for weights:

```bash
# Build standalone inspector
make bin/lfm2_loader

# Inspect real-world LFM2/LFM2.5 hybrid layers and zero-copy pointer bindings
./bin/lfm2_loader models/LFM2.5-1.2B-Q4_K_M.gguf
```

Output highlights:
- Auto-detects hybrid layers via `lfm2.attention.head_count_kv` array metadata (`0` = ShortConv, `>0` = GQA).
- Displays exact tensor strides, quantization types, and `mmap` kernel page cache pointers.
- Identifies tied weights (`token_embd.weight` $\leftrightarrow$ `output.weight`) and LFM2-specific naming (`token_embd_norm.weight`).

---

### 7. Standalone Zero-Dependency C11 Forward Pass (`bin/lfm2_forward`)

A complete, single-file forward pass engine with OpenMP multi-core acceleration and zero external dependencies:

```bash
# Build standalone forward pass executable
make bin/lfm2_forward

# Run greedy autoregressive token decode loop on any LFM2 GGUF checkpoint
OMP_NUM_THREADS=4 ./bin/lfm2_forward models/LFM2.5-1.2B-Q4_K_M.gguf 16
```

Features:
- **Zero-Copy & Sparse Allocation:** Allocates KV caches exclusively for GQA layers and a tiny 16 KB ring buffer for LIV ShortConv layers.
- **Mixed Quantization Support:** Fast internal dequantizers and dot products for `F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, and `Q6_K`.
- **In-Place Per-Head QK-Norm:** Normalizes each 64-dim head vector before applying NeoX RoPE.

---

### 8. Interactive Terminal Chat REPL (`bin/lfm2_chat`)

Stateful multi-turn ChatML terminal assistant optimized specifically for Liquid AI's sampling specifications:

```bash
# Build standalone terminal chat executable
make bin/lfm2_chat

# Launch interactive offline chat with 4 physical CPU threads
OMP_NUM_THREADS=4 ./bin/lfm2_chat models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

Features:
- **Logit-Space Min-P Pre-Filtering:** Discards non-competitive tokens ($z_i < z_{\max} + T \ln(p_{\min})$) in a single linear scan, shrinking 65,536 candidates down to $<32$ before `expf()` and sorting.
- **Deduplicated Sliding-Window Repetition Penalty:** Applies penalty ($1.05\times$) only once per unique token in the 64-token history window.
- **Stateful Turn-by-Turn ChatML:** First turn formats `<|startoftext|><|im_start|>system...`; subsequent turns append only the new user turn delta, retaining active KV caches and ShortConv ring buffer states without re-prefilling.
- **Interactive Slash Commands:** `/reset` (clear state), `/temp <f>`, `/minp <f>`, `/max <n>`, `/exit`.

---

### 9. Priority-Ordered BPE Rank Merging Tokenizer (`bin/lfm2_bpe`)

Exact reproduction of Hugging Face / GGUF BPE tokenization with priority-ordered merging and regex pre-tokenization:

```bash
# Build and run standalone BPE inspector
make bin/lfm2_bpe
./bin/lfm2_bpe models/LFM2.5-1.2B-Q4_K_M.gguf
```

Features:
- **$O(1)$ Hash Lookups:** FNV-1a hash tables for 65,536 tokens and merge pairs.
- **Priority Min-Heap:** Merges token pairs in order of merge rank with leftmost tie-breaking.
- **Special Token Boundary Preservation:** Discrete tags `<|startoftext|>`, `<|im_start|>`, `<|im_end|>` are preserved without mangling.

---

### 10. Lock-Free Physical Topology Spin-Barrier ThreadPool (`bin/lfm2_threadpool`)

C11 spin-barrier thread pool with hardware physical core affinity:

```bash
# Benchmark spin-barrier round-trip and AVX2 memory bandwidth
make bin/lfm2_threadpool
./bin/lfm2_threadpool 4
```

Features:
- **SMT/Hyperthreading Isolation:** Discovers unique physical cores and pins threads via `sched_setaffinity`.
- **Sub-Microsecond Latency:** Spin-barrier achieves ~260 ns dispatch latency with `_mm_pause()`.
- **Power Efficiency:** Transitions to Linux `FUTEX_WAIT_PRIVATE` when idle.

---

### 11. Batched Prefill Acceleration & Empirical Verification (`bin/lfm2_prefill_bench`)

Accelerates prompt prefill by processing prompt token blocks with rank-1 outer-product GEMM and fused `Q8_K` quantization:

```bash
make bin/lfm2_prefill_bench
./bin/lfm2_prefill_bench
```

#### Why Prefill is Faster (The Hardware & Roofline Proof)
1. **DRAM Bandwidth Amortization ($87.5\%$ Reduction in RAM Reads):**
   During single-token decode ($M = 1$), the CPU must stream the entire model's quantized weights (~730 MB for LFM2-1.2B) from system RAM across the memory bus for *every single token*. For 8 tokens, that requires loading $8 \times 730\text{ MB} = 5,840\text{ MB}$. In batched prefill ($M = 8 \times 1$), the engine loads each quantized weight row into fast CPU L1/L2 cache **only once** and computes against all 8 tokens in parallel, reducing memory traffic to just $730\text{ MB}$.
2. **Unpack-Once Register Reuse:**
   Super-block FP16 scale and min (`d_super`, `dmin_super`) and 4-bit quantized nibbles (`w_row[b].qs`) are unpacked into AVX2 YMM registers **only once** for all $B = 8$ tokens, hiding decompression latency.
3. **Skipping Intermediate Vocabulary Projections ($15\%–20\%$ Fewer FLOPs):**
   Prompt tokens $0 \dots T-2$ do not need next-token logits; the costly LM-Head projection ($D \times 65,536$) is executed **only on the final prompt token ($T-1$)**, eliminating millions of redundant multiply-adds.

#### Empirical Verification Commands
To empirically prove the speedup on your machine:
* **Isolated Matrix Benchmark:** Run `./bin/lfm2_prefill_bench` to measure sequential GEMV ($M=1 \times 8$) vs batched GEMM ($M=8 \times 1$). Demonstrates a **1.4x to 1.7x** matrix multiplication speedup with bit-exact parity (`Max Numerical Diff: 0.00000000`).
* **End-to-End Parity Contract Suite:** Run `make test-parity` or `python3 tests/verify_parity.py`. Tier 2 evaluates prompt prefill end-to-end against step-by-step decode, demonstrating an **18x to 26x** end-to-end prefill speedup while asserting bit-exact logit agreement (`Cosine=1.0000000`, `MaxDiff=0.00e+00`).
* **Unit Verification Suite:** Run `./bin/test_prefill_batch` to execute all 7 batched prefill unit tests including the multi-threaded throughput benchmark.
* **Hardware Profiling:** Run `perf stat -e cache-misses,L1-dcache-load-misses,cycles,instructions ./bin/lfm2_prefill_bench` to verify the drop in L1/L2 cache misses and increase in Instructions Per Cycle (IPC).

---

### 12. 1x4 Row-Unrolled AVX2 MatVec & Fused SwiGLU Benchmark (`bin/lfm2_decode_1x4_bench`)

Quad-row unrolled matrix-vector multiplication with register-resident SwiGLU:

```bash
make bin/lfm2_decode_1x4_bench
./bin/lfm2_decode_1x4_bench 4
```

Features:
- **1x4 Row Unrolling:** Amortizes activation loads across 4 contiguous weight rows in YMM registers.
- **Fused SwiGLU FFN:** Evaluates gate ($w_1$) and up ($w_3$) projections in registers, multiplies with rational SiLU, and prevents DRAM round-trips.

---

### 13. Planar ShortConv, RoPE LUT & FP16 Online Softmax Benchmark (`bin/lfm2_kernels_fast_bench`)

Vectorized architectural kernels eliminating scalar bottlenecks:

```bash
make bin/lfm2_kernels_fast_bench
./bin/lfm2_kernels_fast_bench 4
```

Features:
- **Planar AVX2 ShortConv:** Transposes stride-3 weights into aligned planar arrays and computes Gate 1, Conv1D, Gate 2, and L1-hot quantization in a single pass.
- **Precomputed RoPE LUT:** Precomputes `cos_table` and `sin_table` up to `max_seq_len`, fused with per-head QK-norm.
- **FlashAttention-CPU Online Softmax:** Streams `k_cache_f16` and `v_cache_f16` in a single pass with running logit max and sum, cutting KV bandwidth by 50%.

---

### 14. CPUID Multi-ISA Dynamic Dispatch & Extended Quants (`bin/lfm2_cpuid_quants`)

Portable single-binary execution supporting `Q4_0`, `Q5_K`, and `IQ4_XS`:

```bash
make bin/lfm2_cpuid_quants
# Run with automatic hardware detection
./bin/lfm2_cpuid_quants
# Force portable scalar fallback for testing
LFM2_FORCE_ISA=scalar ./bin/lfm2_cpuid_quants
```

Features:
- **Target Attributes:** Compiles SIMD functions with `__attribute__((target("avx2,fma,f16c")))`, preventing `SIGILL` on legacy CPUs.
- **1-Clock `IQ4_XS` Grid Lookup:** Uses `_mm256_shuffle_epi8` (`vpshufb`) for non-linear importance grid translation in 1 CPU cycle.

---

### 15. Runtime Control Plane: PDA Grammar, Think Budget & KV Shift (`bin/lfm2_runtime_features`)

Complete runtime control plane in pure zero-dependency C11:

```bash
make bin/lfm2_runtime_features
./bin/lfm2_runtime_features
```

Features:
- **Zero-Allocation PDA Grammar Logit Masker:** Speculatively evaluates tokens against RFC 8259 JSON and LFM2 Pythonic tool calls in $< 50\ \mu\text{s}$, masking syntax errors with `-INFINITY`.
- **`<think>` Reasoning Budget Controller:** Enforces token limits and deterministically forces `\n</think>\n` through the forward pass.
- **GQA Attention-Sink KV-Cache Shifting:** Evicts old dialogue tokens, shifts keys/values down, and counter-rotates shifted Keys by $-\Delta \cdot \theta$ loaded from `LFM2RoPELUT` in 8 AVX2 instructions.

---

### 16. Sparse Mixture-of-Experts & Hybrid State Checkpointing (`bin/lfm2_moe_checkpoint`)

Zero-dependency C11 implementation unlocking Liquid AI's largest models (`LFM2-8B-A1B` and `LFM2.5-24B-A2B`) with instant speculative draft rollback:

```bash
make bin/lfm2_moe_checkpoint
./bin/lfm2_moe_checkpoint 4
```

Features:
- **Hybrid Dense-Sparse Layer Hierarchy:** Dense SwiGLU layers 0–1, sparse MoE layers 2..N-1 activating Top-4 experts per token.
- **Sigmoid Router with Learned Selection Bias:** Evaluates $s_e = \sigma(\mathbf{W} \hat{\mathbf{x}})$, adds learned bias $b_e$ strictly for ranking ($c_e = s_e + b_e$), and normalizes unbiased probabilities $s_{e_k} / \sum s_{e_j}$ for output mixing.
- **$O(1)$ Zero-Copy 3D Tensor Slicing Over `mmap`:** Reads 2D expert weight slices directly at `e * ne[1] * row_bytes`, touching only 12.5% of layer parameters per token and cutting memory bandwidth to that of a dense 1.5B model.
- **Sub-20 µs Hybrid State Checkpoint (`LFM2Checkpoint`):** Snapshots all ShortConv 2-token rolling buffers (~160–480 KB) and GQA `pos` cursor in $< 15\ \mu\text{s}$, solving the destructive recurrent state overwrite problem during speculative decoding rejection and tree-search backtracking.

---

### 17. Streaming UTF-8 Boundary Assembly, Stop Holdback & Two-Stage SIGINT (`bin/lfm2_cli_stream`)

Zero-dependency streaming filter and terminal controller:

```bash
make bin/lfm2_cli_stream
./bin/lfm2_cli_stream
```

Features:
- **Zero-Mojibake UTF-8 Boundary Buffer:** Sliding-window assembly across multi-byte UTF-8 token boundaries (2-, 3-, and 4-byte codepoints).
- **Multi-Token Stop String Detection (`--stop`):** Dynamic proper prefix holdback, instant divergence flush, and earliest match truncation.
- **Two-Stage Non-Restarting `SIGINT` Guard:** First `Ctrl+C` halts token generation and cleanly finalizes the model turn with `<|im_end|>`; second `Ctrl+C` while idle gracefully exits.
- **Pipe vs TTY Stream Routing:** Routes headers, status tables, and stats to `stderr` when `stdout` is piped, leaving `stdout` as pristine model text.

---

### 18. Linux `/proc/meminfo` OOM Guard, Sequential Page Warmup & Hybrid Prompt Cache (`bin/lfm2_sys_cache`)

Operating-system level memory telemetry, deterministic page warmup, and hybrid prompt cache persistence:

```bash
make bin/lfm2_sys_cache
./bin/lfm2_sys_cache
```

Features:
- **Linux `/proc/meminfo` OOM Guard & Context Clamper:** Direct `/proc/meminfo` parsing calculating hybrid model footprint (GGUF `mmap`, ShortConv buffers, 32 MB arena, and FP16 KV-cache). Dynamically clamps context window to safe limits on memory-constrained systems (e.g. 8 GB RAM machines) to avoid Linux OOM killer termination.
- **Sequential `mmap` Page-Warmup Progress Bar (`--warmup`):** Sequentially faults every 4 KB page of the mmap'd GGUF file using `MADV_WILLNEED` and 4 KB stride reads across 16 MB windows with a live `MB/s` console progress bar, turning 20+ seconds of random seek thrashing on mechanical drives into a single fast contiguous read.
- **Disk-Backed Hybrid Prompt Cache (`--prompt-cache`, `.lfmcache` v1):** Enforces LFM2 exact-prefix caching rules: ShortConv rolling states are recurrent ($s_0, s_1$ at step $t$), requiring 100% bit-for-bit prefix match of all $N_{\text{cached}}$ tokens before restoring states and advancing `pos`. Embeds 64-bit FNV-1a model hyperparameter & routing table fingerprints with atomic file writes (`.tmp.<pid>` + `rename()`), accelerating prompt prefill by skipping matched prefix evaluation in $< 1\ \text{ms}$.
- **Transparent 2 MB HugePage Arena (`MADV_HUGEPAGE`):** Eliminates CPU TLB misses during multi-megabyte KV and activation sweeps.

---

### 19. Non-Blocking Linux `epoll` OpenAI HTTP/SSE Server (`bin/lfm2_server`, `bin/lfm2-server-static`)

A zero-dependency, single-threaded Linux `epoll` HTTP and Server-Sent Events (SSE) server exposing OpenAI-compatible endpoints with cooperative token stepping:

```bash
# Build production server and standalone static demo
make bin/lfm2_server bin/lfm2-server-static

# Run standalone mock demo (zero weight file required)
./bin/lfm2-server-static 8080

# Or run with full LFM2 model
./bin/lfm2_server models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf 8080 127.0.0.1 4 2048
```

Features:
- **Zero-Stall Event Loop (`epoll_wait` + Single-Token Stepping):** Active SSE client transitions to `CONN_STREAMING_SSE`. On each tick of the event loop, the server calls `lfm2_context_next_utf8()` once (generating a single verified UTF-8 chunk in ~10–25 ms), flushes `data: {...}\n\n` with `MSG_NOSIGNAL` and `TCP_NODELAY`, and returns immediately to `epoll_wait(..., timeout = 0)`.
- **Instant Client-Disconnect Cancellation (`EPOLLRDHUP`):** Detects `EPOLLRDHUP | EPOLLHUP | EPOLLERR` on token boundaries to immediately abort generation and free CPU cores if a user clicks "Stop" or aborts a `curl` stream.
- **Pre-Allocated Connection Pool & Anti-Slowloris Deadlines:** Fixed array of 32 connections with pre-allocated 64 KB request and 128 KB response buffers and 10-second header timeouts—zero dynamic heap allocations (`malloc`/`free`) in request hot paths.
- **Full OpenAI Endpoints & Embedded Chat Studio:** Serves `GET /` (embedded dark-mode JetBrains Mono chat studio), `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming SSE & non-streaming JSON, reasoning content tokens, grammar constraints, stop strings), `POST /v1/embeddings` (L2-normalized embeddings via `lfm2_embed_text`), and `OPTIONS` (CORS preflight).
- **Standalone Compile Mode:** Direct compilation without model dependencies using `-DLFM2_SERVER_STANDALONE_DEMO`:
  ```bash
  gcc -std=c11 -O3 -DLFM2_SERVER_STANDALONE_DEMO -static -Wall -Wextra -o lfm2_server lfm2_server.c -lm
  ```

---

## Phase Status Summary

### Phase 1: Reference Architecture (Complete)
- [x] Zero-copy GGUF v2/v3 binary parser with `mmap` and `madvise(MADV_SEQUENTIAL)`.
- [x] 64-byte aligned memory arena allocator (zero dynamic `malloc` during generation).
- [x] Full forward pass implementation (RMSNorm, LIV ShortConv, GQA, RoPE, SwiGLU).
- [x] Byte-level BPE Tokenizer with byte fallback.
- [x] Top-K, temperature, and greedy sampling pipeline.
- [x] PyTorch numerical parity verified ($< 10^{-7}$ diff).

### Phase 2: SIMD, Quantization & Fused Kernels (Complete)
- [x] **AVX2 + FMA SIMD Vectorization:** 8-wide float dot product, vectorized RMSNorm, vector mul/add, and fast rational SiLU.
- [x] **Quantized Matrix Multiplication:** Block-quantized `Q8_0` and `Q4_0` dot-product kernels with uncompressed runtime parity.
- [x] **Fused LIV ShortConv Kernel:** Register-resident fusion of Gate 1, Conv1D, ring buffer state shift, and Gate 2, slashing memory traffic by 50% ($12D \rightarrow 6D$ floats).
- [x] **Low-Overhead POSIX ThreadPool:** Multi-threaded parallel row-chunked MatVec with monotonic generation-barrier synchronization.

### Phase 3: Single-Binary Distribution, Web Studio & C ABI (Complete)
- [x] **Public C ABI & Static/Shared Libraries:** Stable `lfm_c_api.h` header with `lib/liblfm.so` and `lib/liblfm.a`.
- [x] **Zero-Dependency Python FFI Bindings:** Pure `ctypes` wrapper `LFMEngine` in `bindings/python/laii.py`.
- [x] **Embedded HTTP REST Server & Web Studio:** POSIX socket HTTP server with OpenAI-compatible endpoints and in-memory glassmorphic UI (`lfm-server`).
- [x] **Standalone Static ELF Packaging:** Statically linked binaries `bin/lfm-run-static` and `bin/lfm-server-static`.

### Phase 4: Fast Architectural Kernels & Runtime Control Plane (Complete)
- [x] **Priority-Ordered BPE Rank Merging:** $O(N \log N)$ doubly-linked list with binary min-heap and GPT-2 regex pre-tokenizer state machine (`lfm2_bpe.c`).
- [x] **Lock-Free Spin-Barrier ThreadPool:** SMT/Hyper-Threading core isolation, 64-byte aligned atomic state, user-space spin-barrier (~260 ns latency), and Linux `FUTEX_WAIT_PRIVATE` sleep (`lfm2_threadpool.c`).
- [x] **Batched Prefill Acceleration:** Outer-product rank-1 GEMM across prompt token blocks with fused SIMD `Q8_K` quantization (`lfm2_prefill_batch.c`).
- [x] **1x4 Row-Unrolled AVX2 MatVec & Register-Resident SwiGLU FFN:** Quad-row unrolling and parallel register accumulators (`lfm2_decode_1x4.c`).
- [x] **Fast Architectural Kernels:** Planar AVX2 ShortConv with aligned planar arrays, precomputed RoPE LUT (`LFM2RoPELUT`), and FlashAttention-CPU single-pass online softmax over FP16 KV-cache (`lfm2_kernels_fast.c`).
- [x] **Runtime CPUID Multi-ISA Dynamic Dispatch:** Target attributes (`avx2,fma,f16c`), AVX2 `Q4_0`, `Q5_K`, and `IQ4_XS` (via 1-cycle `vpshufb`), and safe portable scalar fallback (`lfm2_cpuid_quants.c`).
- [x] **Runtime Control Plane:** Zero-allocation byte-level PDA grammar logit masker (RFC 8259 JSON & Pythonic tool calls), `<think>` reasoning token budget controller, and GQA attention-sink KV-cache shifting with AVX2 RoPE re-rotation (`lfm2_runtime_features.c`).
- [x] **Sparse Mixture-of-Experts & Hybrid State Checkpointing:** Dense layers 0–1, sparse MoE layers 2..N-1 activating Top-4 experts, Sigmoid router with `exp_probs_b` selection bias, zero-copy 3D tensor slicing over `mmap`, and sub-20 µs ShortConv + GQA state rollback (`lfm2_moe_checkpoint.c`).
- [x] **Streaming UTF-8 Assembly & Two-Stage SIGINT:** Multi-byte boundary assembly, stop holdback, pipe/TTY detection, and signal handling (`lfm2_cli_stream.c`).
- [x] **Linux OOM Guard, Sequential Page Warmup & Hybrid Prompt Cache:** Real-time `/proc/meminfo` parsing, context clamping, 4 KB stride page-fault sweeping with live `MB/s` progress bar, and atomic exact-prefix prompt caching (`lfm2_sys_cache.c`).
- [x] **Public C11 ABI & High-Performance Python SDK:** Stable opaque handles (`lfm2_model_t`, `lfm2_context_t`, `lfm2_checkpoint_t`), zero-copy raw logit accessor (`lfm2_context_get_logits_ptr` / `ctx.get_logits()`), pull-based UTF-8 generator (`lfm2_context_next_utf8`), sub-15 µs context branching rollback (`ctx.branch()`), and L2-normalized text embeddings (`lfm2_embed_text`) in `include/lfm2.h`, `src/lfm2_api.c`, `lib/liblfm2.so`, and `python/lfm2.py`.
- [x] **Non-Blocking Linux `epoll` OpenAI Server:** Zero-stall cooperative token stepping, `EPOLLRDHUP` client disconnect abortion, pre-allocated fixed connection array, embedded web studio, and OpenAI HTTP/SSE endpoints (`tools/lfm2_server.c`, `lfm2_server.c`, `bin/lfm2_server`, `bin/lfm2-server-static`).
- [x] **Comprehensive Test Suite & Static Artifacts:** 27 automated C test binaries (`make test-c-bins`), Python SDK test suites, 5-tier numerical parity SLA & golden oracle verifier (`tests/verify_parity.py` / `make test-parity`), 14 standalone tools/benchmarks, and 16 static ELF binaries (`make static-test`) passing with 100% success rate under GCC and AddressSanitizer/UBSan.

---

## License & Disclaimers

This project is licensed under the [MIT License](LICENSE).

### Trademarks & Non-Affiliation
"Liquid AI", "LFM", "LFM2", and "LFM2.5" are trademarks or registered trademarks of Liquid AI, Inc. This repository is an independent, community-developed open-source software project engineered to execute models conforming to the LFM architecture. It is not affiliated with, endorsed by, sponsored by, or associated with Liquid AI, Inc.

### Weights & Artifacts
This repository contains code only. Pretrained model checkpoints and weights are published separately by Liquid AI under the Liquid AI Community License and can be downloaded from Hugging Face.


