# LFM Engine Engineering Roadmap

This document outlines the three-phase engineering trajectory for building a specialized, zero-dependency inference engine for Liquid AI models on Linux.

---

## Phase 1: Mathematical Parity & Reference C Engine (Completed)

* **Goal:** A clean, zero-dependency C11 reference implementation running full forward pass (FP32/FP16) verified with exact numerical parity against Hugging Face PyTorch references.
* **Key Deliverables:**
  - [x] **GGUF v3 Binary Parser:** Zero-copy `mmap` loader reading header, metadata key-value tables, and tensor descriptor directory.
  - [x] **Memory Arena Allocator:** Single 64-byte cache-line aligned arena preventing all dynamic `malloc` during token generation.
  - [x] **Forward Pass Graph:**
    - RMSNorm with configurable epsilon.
    - Double-gated Linear Input-Varying (LIV) ShortConv ($k=3$) with 2-token ring-buffer states.
    - Causal 1D convolution prefill routine.
    - Grouped Query Attention (GQA) with QK-Norm, RoPE, and KV-cache.
    - SwiGLU FFN with SiLU activation.
  - [x] **Byte-level BPE Tokenizer:** Decode/encode pipeline handling vocab lookup, special tokens (`<|im_start|>`, `<|im_end|>`), and UTF-8 byte fallbacks.
  - [x] **Sampling Pipeline:** Greedy (argmax), temperature scaling, and top-k filtering.
  - [x] **Comprehensive 7-Part Test Suite:** Unit tests, PyTorch numerical parity tests ($< 10^{-7}$ diff), end-to-end forward pass test, and AddressSanitizer/UBSan memory verification (see [TEST_SUITE.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/TEST_SUITE.md)).

---

## Phase 2: High-Performance SIMD & Quantization (Completed)

* **Goal:** Maximum memory-bandwidth utilization on modern CPUs with AVX2/FMA vectorization, block quantization (`Q8_0`, `Q4_0`), fused LIV ShortConv kernels, and multi-core thread scaling.
* **Key Deliverables:**
  - [x] **AVX2 + FMA SIMD Vectorization (`lfm_simd.h`, `lfm_simd.c`):**
    - 8-wide float dot product (`vec_dot_f32`) with horizontal register reduction.
    - Vectorized RMSNorm (`vec_rmsnorm`), elementwise multiply (`vec_mul`), and add (`vec_add`).
    - Vectorized rational approximation of SiLU activation (`vec_silu`).
  - [x] **Quantized Tensor Computation (`Q8_0` & `Q4_0`):**
    - `Q8_0` block dot product with 32x int8 weights + fp16 scale (`vec_dot_q8_0`).
    - `Q4_0` block dot product with packed 4-bit nibbles and offset centering (`vec_dot_q4_0`).
    - Unified matrix-vector dispatch (`lfm_matvec_simd`) supporting FP32, Q8_0, and Q4_0.
  - [x] **Fused LIV ShortConv Kernel (`lfm_fused.h`, `lfm_fused.c`):**
    - Single-pass register-resident fusion of Gate 1 ($u = B \odot x$), Conv1D ($k=3$), ring buffer state shift ($s_0 \leftarrow s_1, s_1 \leftarrow u$), and Gate 2 ($y = C \odot \text{conv\_out}$).
    - Slashes intermediate DRAM/L3 memory traffic by 50% ($12D \rightarrow 6D$ floats).
  - [x] **Low-Overhead POSIX ThreadPool (`lfm_threadpool.h`, `lfm_threadpool.c`):**
    - Zero-allocation persistent worker threads with generation-based condition variable barrier.
    - Multi-threaded row-chunked parallel MatVec (`lfm_matvec_parallel`) scaling linear projections across all CPU cores.
    - CLI options `-j / --threads` for user-configurable core parallelism.
  - [x] **Expanded 11-Part Automated Test Suite:**
    - Added `test_simd_avx2`, `test_quants`, `test_fused_conv`, and `test_threads` (11/11 tests passing, verified with ASan/UBSan).
    - Detailed in [PHASE2_SPECIFICATION.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/PHASE2_SPECIFICATION.md) and [TEST_SUITE.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/TEST_SUITE.md).

---

## Phase 3: Single-Binary Distribution & Packaging (Completed)

* **Goal:** Single static Linux executable running offline with zero dependencies, embedded web UI, C ABI shared/static libraries, and Python ctypes bindings.
* **Key Deliverables:**
  - [x] **Public C ABI & Static/Shared Libraries (`lfm_c_api.h`, `liblfm.a`, `liblfm.so`):**
    - Clean public C header with opaque engine handle, streaming callback generation, tokenization, and metadata getters.
    - Exported shared object (`lib/liblfm.so`) and static archive (`lib/liblfm.a`) compiled with `-fPIC`.
  - [x] **Zero-Dependency Python FFI Bindings (`bindings/python/laii.py`):**
    - Pure `ctypes` wrapper `LFMEngine` supporting context managers, generator-based token streaming, and vocabulary access without PyTorch/CUDA dependencies.
    - Automated unit test suite in `tests/test_python_bindings.py`.
  - [x] **Embedded HTTP Server & Web Studio (`lfm_server.h`, `lfm_server.c`, `lfm_webui.h`):**
    - Zero-dependency POSIX socket HTTP server with CORS support and graceful SIGINT/SIGTERM shutdown.
    - OpenAI-compatible REST API endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
    - Single-page dark glassmorphic Web Studio embedded directly in binary memory as `LFM_EMBEDDED_INDEX_HTML`.
    - Dedicated server binary `lfm-server` and CLI flag `lfm-run -s / --server`.
  - [x] **Standalone Static ELF Packaging (`make static`):**
    - Fully static ELF executables `bin/lfm-run-static` and `bin/lfm-server-static` (`statically linked`) runnable on any x86_64 Linux kernel with zero external dependencies.
  - [x] **Standalone Zero-Copy LFM2 Tools (`bin/lfm2_loader`, `bin/lfm2_forward`, `bin/lfm2_chat`):**
    - Zero-dependency C11 GGUF v3 inspector and forward pass engine supporting real-world Liquid AI checkpoints.
    - Native support for mixed quantization (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q6_K`) and OpenMP multi-core parallelization.
    - Interactive terminal chat REPL (`bin/lfm2_chat`) matching Liquid AI's post-training specifications: logit-space Min-P pre-filtering, deduplicated repetition penalty, and stateful multi-turn ChatML continuation.
  - [x] **Comprehensive 17-Part Automated Test Suite:**
    - Expanded test framework covering all 17 test suites with 100% pass rate under normal build and AddressSanitizer/UBSan runs.
    - Detailed in [PHASE3_SPECIFICATION.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/PHASE3_SPECIFICATION.md) and [TEST_SUITE.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/TEST_SUITE.md).

---

## Phase 4: Advanced Architectural Kernels & Runtime Control Plane (Completed)

* **Goal:** Maximum end-to-end decode efficiency, unified CPUID multi-ISA portability without SIGILL crashes, sub-50 µs constrained grammar decoding, and attention-sink infinite context management.
* **Key Deliverables:**
  - [x] **Priority-Ordered GGUF BPE Rank Merging (`tools/lfm2_bpe.c`):**
    - $O(1)$ FNV-1a string and merge-pair hash tables for 65,536 tokens.
    - $O(N \log N)$ doubly-linked list with binary min-heap and lazy version invalidation.
    - GPT-2 regex pre-tokenizer state machine and special boundary tag pre-splitting.
  - [x] **Lock-Free Physical Topology Spin-Barrier ThreadPool (`tools/lfm2_threadpool.c`):**
    - SMT/hyperthreading core isolation with `/sys/devices/system/cpu/` pinning.
    - 64-byte cache-line aligned atomic state layout with PAUSE busy-wait (~260 ns dispatch latency).
    - Dynamic work-stealing and Linux `FUTEX_WAIT_PRIVATE` low-power sleep transitions.
  - [x] **Batched Prefill Acceleration (`tools/lfm2_prefill_batch.c`):**
    - Rank-1 accumulator GEMM across parallel token blocks during prompt prefill.
    - Fused SIMD `Q8_K` activation pre-quantization.
  - [x] **1x4 Row-Unrolled AVX2 MatVec & Register-Resident SwiGLU FFN (`tools/lfm2_decode_1x4.c`):**
    - 4-row unrolled GEMV amortizing activation vector loads across consecutive rows.
    - Register-resident fused SwiGLU FFN computing $w_1$ (gate) and $w_3$ (up) simultaneously in vector registers.
  - [x] **Fast SIMD Kernels (`tools/lfm2_kernels_fast.c`):**
    - Planar AVX2 ShortConv 3-tap kernel with transposed weights and direct L1-hot quantization.
    - Precomputed RoPE LUT (`LFM2RoPELUT`) + fused per-head QK-norm.
    - Single-pass online softmax (FlashAttention-CPU) over FP16 KV-cache (`F16C`).
  - [x] **Runtime CPUID Multi-ISA Dispatch & Extended Quantization (`tools/lfm2_cpuid_quants.c`):**
    - GCC/Clang target attributes (`avx2,fma,f16c`) enabling single-binary portability with zero SIGILL risk.
    - AVX2 inner loops for `Q4_0`, `Q5_K`, and `IQ4_XS` (1-cycle non-linear grid lookup via `vpshufb`).
    - Dynamic runtime selection with portable scalar fallback and `LFM2_FORCE_ISA=scalar` testing hook.
  - [x] **Runtime Control Plane (`tools/lfm2_runtime_features.c`):**
    - Zero-allocation byte-level PDA grammar logit masker for RFC 8259 JSON and LFM2 Pythonic tool calls (`[func(k=v)]`).
    - `<think>` reasoning token budget controller with deterministic forced closing sequence emission.
    - GQA attention-sink KV-cache shifting with AVX2 RoPE re-rotation ($-\Delta \cdot \theta_i$).
  - [x] **Sparse Mixture-of-Experts & Hybrid State Checkpointing (`tools/lfm2_moe_checkpoint.c`):**
    - Sparse MoE (`lfm2moe`) support for `LFM2-8B-A1B` and `LFM2.5-24B-A2B` with hybrid leading dense layers (0–1) and 32/64-expert layers activating Top-4 experts per token.
    - Sigmoid router with learned per-expert load-balancing selection bias (`exp_probs_b` / `exp_probs_b.bias`) and unbiased normalized Top-4 probability mixing.
    - $O(1)$ zero-copy 3D expert tensor slicing over `mmap` buffers (`[ne0, ne1, n_exp]`), loading only active expert weights (12.5% of layer parameters).
    - Fused 1x4 MoE down-projection with thread-pool row chunking accumulating directly into the residual stream.
    - Hybrid state checkpointing (`LFM2Checkpoint`) capturing ShortConv rolling buffers (~160–480 KB) and GQA `pos` in $< 20\ \mu\text{s}$ for instant speculative draft rollback and tree-search backtracking.
  - [x] **Streaming UTF-8 Boundary Buffer, Multi-Token Stop-String Filter & Two-Stage SIGINT (`tools/lfm2_cli_stream.c`):**
    - Sliding-window assembly of 2-, 3-, and 4-byte UTF-8 codepoints across token boundaries (zero mojibake replacement characters ``).
    - Multi-token stop-string matcher (`--stop`) with proper prefix holdback and instant divergence flush.
    - Two-stage `sigaction` signal handler (`Ctrl+C` aborts active generation turn, while second `Ctrl+C` at idle prompt gracefully exits).
    - UNIX pipe & TTY detector (`LFM2TerminalEnv`) routing telemetry/banners to `stderr` when redirected so `stdout` remains pure model text.
  - [x] **Linux `/proc/meminfo` OOM Guard, Page Warmup & Hybrid Prompt Cache (`tools/lfm2_sys_cache.c`):**
    - Direct `/proc/meminfo` parser evaluating memory headroom (`MemAvailable`, `Cached`, `SwapFree`) and calculating hybrid model footprint (GGUF `mmap`, ShortConv buffers, 32 MB arena, and FP16 KV-cache).
    - Dynamic context window clamper preventing Linux OOM killer termination on memory-constrained systems (8 GB RAM).
    - 2 MB HugePage arena allocator (`MADV_HUGEPAGE`) eliminating CPU TLB misses during multi-megabyte activation sweeps.
    - Sequential 4 KB stride page-fault warmup (`MADV_WILLNEED`) with real-time `MB/s` console progress bar eliminating mechanical drive seek thrashing during Token 0.
    - 64-bit FNV-1a model hyperparameter & routing table fingerprinting with atomic file writes (`.tmp.<pid>` + `rename()`).
    - Exact full-prefix prompt cache (`--prompt-cache`, `.lfmcache` v1) enforcing LFM2 recurrent ShortConv state invariance ($s_0, s_1$), skipping matched prefix prefill in $< 1\ \text{ms}$ with full checksum tamper rejection.
  - [x] **Public C11 ABI & High-Performance Python SDK (`include/lfm2.h`, `src/lfm2_api.c`, `lib/liblfm2.so`, `python/lfm2.py`):**
    - Zero-dependency public C11 ABI with opaque handles (`lfm2_model_t`, `lfm2_context_t`, `lfm2_checkpoint_t`) and fixed-width POD structs compiled with `-fvisibility=hidden`.
    - Pull-based single-threaded UTF-8 chunk generation (`lfm2_context_next_utf8`) eliminating background worker thread / queue overhead for Python generators and avoiding mid-token mojibake or stop-prefix leakage.
    - Zero-overhead hybrid context branching context manager (`ctx.branch()`) restoring ShortConv recurrent states and GQA `pos` in $< 15\ \mu\text{s}$ for tree search and speculative decoding.
    - High-throughput L2-normalized text embedding generation (`lfm2_embed_text`) supporting both causal `LAST_TOKEN` and sequence `MEAN` pooling.
    - Zero-copy raw logit vector distribution accessor (`lfm2_context_get_logits_ptr` / `ctx.get_logits()`) and 5-tier numerical parity & golden oracle verification suite (`tests/verify_parity.py` / `make test-parity`).
  - [x] **Non-Blocking Linux `epoll` OpenAI HTTP/SSE Server (`tools/lfm2_server.c`, `lfm2_server.c`, `bin/lfm2_server`, `bin/lfm2-server-static`):**
    - Zero-stall cooperative event loop (`epoll_wait` + single-token stepping with `lfm2_context_next_utf8()` and timeout = 0 during active SSE generation).
    - Instant client-disconnect cancellation (`EPOLLRDHUP | EPOLLHUP | EPOLLERR`) aborting generation on the very next token boundary to free CPU cores.
    - Pre-allocated 32-connection pool with 64 KB request and 128 KB response buffers and anti-Slowloris monotonic deadlines (zero dynamic allocations).
    - Embedded zero-dependency HTML/CSS/JS chat studio (`GET /`) and full OpenAI REST API v1 (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `OPTIONS`).
    - Standalone compilation support via `-DLFM2_SERVER_STANDALONE_DEMO` requiring zero model checkpoints.
  - [x] **Expanded 30-Part Automated Test Suite & 18 Static ELF Binaries:**
    - 30 automated C test binaries (`make test-c-bins`) + Python SDK automated test (7/7 tests) + 16 standalone tools + 18 statically linked ELF binaries (`make static-test`).
    - 100% pass rate under normal compilation, AddressSanitizer/UBSan runs, and ThreadSanitizer (`make test-tsan`).
    - Detailed in [PHASE4_SPECIFICATION.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/PHASE4_SPECIFICATION.md) and [TEST_SUITE.md](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/TEST_SUITE.md).

---

## Phase 5: Mobile Edge & Systems Hardening (v1.0.0 Certified)

* **Goal:** Cross-platform Android mobile SoC readiness, ThreadSanitizer concurrency certification, musl static distribution, big.LITTLE thermal optimization, and zero-weight speculative decoding.
* **Key Deliverables:**
  - [x] **Zero-Weight Prompt-Lookup Speculative Decoding (`tools/lfm2_speculative_pld.c`, `include/lfm2_speculative_pld.h`):**
    - $O(1)$ hierarchical 3-gram and 2-gram MurmurHash3 lookup tables ($< 1\ \mu\text{s}$ drafting).
    - Amortized memory bandwidth verification via batched GEMM ($B \le 8$), evaluating candidate drafts in 1 DRAM weight sweep.
    - Per-slot ShortConv rolling trace (`conv_trace[b][l]`) enabling $O(1)$ causal rollback in $8\ \mu\text{s}$ without recomputing previous tokens.
    - Automated unit test in `tests/test_speculative_pld.c` certifying 100% bit-exact sequence parity against baseline decode.
  - [x] **Native ARM64 NEON + DotProd (`vdotq_s32`) + FP16 SIMD Suite (`tools/lfm2_simd_arm64.c`, `include/lfm2_simd_arm64.h`):**
    - 1x1 and 1x4 row-unrolled microkernels for `Q4_K`, `Q6_K`, `Q5_K`, `IQ4_XS`, `Q4_0`, and `Q8_0` with zero register spills (32 ARM64 registers).
    - Single-cycle signed `SDOT` (`vdotq_s32`) integer dot products for high-throughput mobile inference (25–45 tok/s).
    - 1-cycle `TBL` lookup (`vqtbl1q_s8`) for `IQ4_XS` non-linear grid translation.
    - `getauxval(AT_HWCAP)` dynamic CPU capability detection (`HWCAP_ASIMDDP`, `HWCAP_FPHP`) preventing `SIGILL` on older devices.
    - Automated unit test suite in `tests/test_simd_arm64.c`.
  - [x] **Android Scoped Storage Zero-Copy Loader (`tools/lfm2_loader.c`, `src/lfm2_api.c`, `python/lfm2.py`):**
    - `lfm2_model_load_from_fd(int fd, ...)` duplicating descriptors via `dup(fd)` for zero-copy memory mapping of content URIs without filesystem path access.
    - Python SDK `LFM2Model.from_fd(fd)` for turnkey integration with Android `pyjnius` / SAF.
  - [x] **Android `lmkd` Memory Survival & Dynamic Page Warmup (`include/lfm2.h`, `tools/lfm2_loader.c`):**
    - `lfm2_model_release_page_cache(model)` calling `madvise(MADV_DONTNEED)` for instant RSS drop on app pause/backgrounding.
    - Dynamic 16 KB page size support via `sysconf(_SC_PAGESIZE)` for Android 15+ kernel compatibility.
  - [x] **Unified Cross-Platform & Asymmetric `big.LITTLE` Thread Pool (`tools/lfm2_threadpool.c`):**
    - Sysfs EAS capacity (`/sys/devices/system/cpu/cpuN/cpu_capacity`) and clock frequency (`cpuinfo_max_freq`) scanner.
    - Automatic core tier classification (`PRIME`, `PERF`, `LITTLE`).
    - PC $N_{\text{phys}} - 1$ core rule vs Mobile $N_{\text{big}} - 1$ core rule (leaving 1 Big core for Kivy/OpenGL and Little cores for OS).
    - Thermal safety ceiling $N_{\text{total}} - 2$ (max 6 cores on 8-core SoCs) eliminating Vdroop brownouts and thermal shutdowns.
    - Worker 0 pinned to fastest Prime core accelerating single-threaded sequential ops.
    - ARM64 `FPCR.FZ` and `FPCR.FZ16` flush-to-zero in [`include/lfm2_simd_avx2.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2_simd_avx2.h).
    - Adaptive spin loops (25,000 on x86 vs 4,000 on ARM64) and `lfm2_cpu_pause()` (`yield` on ARM64).
  - [x] **ThreadSanitizer & Musl Static Build Pipeline (`Makefile`):**
    - `make test-tsan` compiling with `-fsanitize=thread` and executed under `setarch x86_64 -R` (0 data races verified).
    - `make static-musl` target for distro-agnostic static ELF binaries via `musl-gcc`.
  - [x] **Mobile Integration Documentation (`docs/ANDROID_KIVY_INTEGRATION.md`):**
    - Turnkey developer guide for NDK compilation, `p4a` recipe, threading, Scoped Storage, and lifecycle management without repo bloat.



