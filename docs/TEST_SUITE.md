# Test Suite Specification & Quality Assurance

This document details the test framework backing the Liquid AI (LFM2) C inference engine. To maintain production reliability and eliminate regressions, **every phase of development is backed by an automated test suite**.

---

## 1. Test Suite Architecture

```
                                             [ Full Test Suite: make test ]
                                                           |
     +---------------+--------------+--------------+-------+-------+---------------+-----------------+---------------+
     |               |              |                      |       |               |                 |               |
     v               v              v                      v       v               v                 v               v
[ Unit Tests ]  [ Math Tests ] [ SIMD Tests ]        [ Quantization ] [ Parity Tests ] [ Integration ]  [ C ABI & FFI ] [ REST & Static ]
- test_arena    - test_math    - test_simd_avx2      - test_quants    - test_conv1d    - test_forward    - test_c_api    - test-http
- test_gguf     (RMSNorm,      - test_fused_conv     (Q8_0, Q4_0,     - test_attention (toy GGUF model)  - test-python   - static-test
- test_tokenizer SiLU, RoPE)   - test_threads         Q5_K, IQ4_XS)   - test_optimizations
- test_bpe                     - test_kernels_fast   - test_cpuid_quants
                               - test_decode_1x4     - test_prefill_batch
                               - test_runtime_features
                               - test_moe_checkpoint
                                                           |
                                                           v
                                             [ Memory Safety: make test-asan ]
                                              - AddressSanitizer (ASan)
                                              - UndefinedBehaviorSanitizer (UBSan)
                                              - Zero memory leaks across C & HTTP runs
```

---

## 2. Test Catalog

### 2.1 Unit Tests

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_arena` | `arena.c` | • Strict 64-byte cache-line alignment across 10 distinct allocation sizes<br>• Arena pointer offset reset<br>• Out-of-memory boundary checks |
| `bin/test_math` | `lfm_forward.c` | • RMSNorm numerical stability & epsilon scaling<br>• SiLU non-linear activation accuracy<br>• RoPE norm preservation (orthogonal rotation: $\|RoPE(x)\| = \|x\|$)<br>• FP32 Matrix-Vector dot product correctness |
| `bin/test_gguf` | `gguf.c` | • GGUF v3 magic byte and version validation<br>• Metadata dictionary parsing (`lfm2.*` keys)<br>• Tensor directory reading & 32-byte data alignment |
| `bin/test_tokenizer` | `lfm_tokenizer.c` | • Token string vocabulary extraction from GGUF<br>• Byte-level BPE longest-match prompt encoding<br>• `<0xHH>` hex-to-byte fallback decoding |
| `bin/test_bpe` | `tools/lfm2_bpe.c` | • Exact priority-ordered GGUF BPE rank merging<br>• GPT-2 Byte-to-Unicode mapping & round-trip detokenization<br>• GPT-2 pre-tokenizer regex state machine<br>• Discrete special token preservation (`<|startoftext|>`, `<|im_start|>`, `<|im_end|>`) |

### 2.2 Numerical Parity Tests (PyTorch Golden Standards)

Generated via `tools/generate_reference_tensors.py`:

| Test Binary | Component | Reference Oracle | Tolerance Threshold | Measured Max Difference |
| :--- | :--- | :--- | :--- | :--- |
| `bin/test_conv1d` | LIV Gated ShortConv | PyTorch `F.conv1d` + linear projections | $< 10^{-4}$ | **$2.98 \times 10^{-8}$** (near float epsilon) |
| `bin/test_attention` | GQA + RoPE + QK-Norm | PyTorch `torch.matmul` + Softmax | $< 10^{-4}$ | **$9.69 \times 10^{-8}$** |

### 2.3 Integration Tests

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_forward` | `lfm_engine.c` | • Full 2-layer hybrid model forward pass on `lfm2_toy.gguf`<br>• Prompt ingestion (prefill) & KV cache recording<br>• Autoregressive token generation & callback streaming |

### 2.4 AVX2 SIMD & Quantization Tests (Phase 2)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_simd_avx2` | `lfm_simd.c` | • Vector dot product (`vec_dot_f32`) across 12 distinct alignment boundaries and vector sizes ($N \in [1, 1024]$)<br>• Vectorized RMSNorm (`vec_rmsnorm`) precision vs scalar<br>• 8-wide vector multiply (`vec_mul`) and accumulate (`vec_add`) |
| `bin/test_quants` | `lfm_simd.c` | • `Q8_0` SIMD dot product parity vs unquantized reference ($< 10^{-6}$ diff)<br>• `Q4_0` SIMD dot product with 4-bit nibble unpacking and offset centering vs reference ($< 10^{-6}$ diff) |
| `bin/test_simd_kquants` | `lfm2_simd_avx2.h`, `lfm_simd.c` | • Single-pass activation vector $x \rightarrow Q8\_K$ quantization with sub-block `bsums[16]` sums<br>• `_mm256_maddubs_epi16` integer SIMD dot products for `Q4_K` and `Q6_K` with zero-point cancellation<br>• Parity vs scalar reference ($< 0.15\%$ relative diff for `Q4_K`, $< 0.25\%$ for `Q6_K`)<br>• High-throughput OpenMP parallel MatVec benchmarking |
| `bin/test_fused_conv` | `lfm_fused.c` | • Register-resident fused LIV kernel (`Gate 1 -> Conv1d -> State Update -> Gate 2`) numerical equivalence vs unfused reference ($< 10^{-6}$ diff on output $y$, exact $0.0$ diff on states $s_0, s_1$) |

### 2.5 Concurrency & ThreadPool Tests (Phase 2)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_threads` | `lfm_threadpool.c` | • Worker thread initialization and clean shutdown<br>• Concurrency stress test: 50 consecutive parallel MatVec dispatches with generation-counter barrier synchronization<br>• Zero output divergence ($\Delta < 10^{-5}$) between multi-threaded and serial execution |
| `bin/test_lfm2_threadpool` | `lfm2_threadpool.c` | • Physical CPU topology discovery (`/sys/devices/system/cpu/`) and SMT core isolation<br>• Zero-lock dynamic atomic work-stealing correctness across 16,384 chunks<br>• Bit-exact numerical parity ($0.00$ diff) between serial and MT for `Q4_K` and `Q6_K` dot products<br>• Low-power `FUTEX_WAIT_PRIVATE` sleep transition during idle and instant wake-up recovery<br>• Full 8-step decode pass through hybrid ShortConv + GQA model state |

### 2.6 Public C ABI & FFI Tests (Phase 3)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_c_api` | `lfm_c_api.c` | • Engine lifecycle: `lfm_create` and `lfm_destroy`<br>• Model metadata inspection (`lfm_get_hidden_dim`, `lfm_get_num_layers`, `lfm_get_vocab_size`, `lfm_get_architecture`)<br>• Synchronous text generation into caller-allocated buffer<br>• Streaming generation via callback `lfm_token_callback_t`<br>• Tokenization and detokenization routines |

### 2.7 Python ctypes Bindings Tests (Phase 3)

| Test Suite | Target Component | Verifications |
| :--- | :--- | :--- |
| `tests/test_python_bindings.py` | `bindings/python/laii.py` | • Loading `liblfm.so` via `ctypes` without PyTorch/CUDA runtime<br>• Pythonic context manager lifecycle (`with LFMEngine(...) as engine:`)<br>• Text tokenization and single-token detokenization<br>• Synchronous string generation (`generate`)<br>• Real-time token generator streaming (`generate_stream`) |

### 2.8 Embedded HTTP REST Server & Web Studio Tests (Phase 3)

| Test Suite | Target Component | Verifications |
| :--- | :--- | :--- |
| `tests/test_http_server.sh` | `lfm_server.c`, `lfm_webui.h` | • Spawns background HTTP server on port 18080<br>• `GET /health` and `GET /v1/models` JSON metadata verification<br>• `GET /` serves complete embedded glassmorphic Web Studio (17KB+ HTML)<br>• `POST /v1/chat/completions` OpenAI-compatible completion verification<br>• Clean SIGTERM termination with zero socket leaks |

### 2.9 Advanced Architectural Optimizations Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_optimizations` | `tools/lfm2_forward.c` | • Bit-exact parity between scalar and fused ShortConv + SwiGLU kernels<br>• Verification across arbitrary alignment boundaries |
| `bin/test_prefill_batch` | `tools/lfm2_prefill_batch.c` | • Multi-token parallel activation pre-quantization into `block_q8_K`<br>• Rank-1 outer-product accumulator GEMM speedup and bit-exact parity vs single-token decode |
| `bin/test_decode_1x4` | `tools/lfm2_decode_1x4.c` | • 1x4 row-unrolled AVX2 MatVec amortizing activation loads across 4 weight rows<br>• Register-resident fused SwiGLU FFN computing $w_1$ and $w_3$ simultaneously in registers |
| `bin/test_kernels_fast` | `tools/lfm2_kernels_fast.c` | • Planar AVX2 ShortConv 3-tap kernel with transposed weights and direct L1-hot quantization<br>• Precomputed RoPE lookup table (`LFM2RoPELUT`) + fused per-head QK-norm<br>• FlashAttention-CPU single-pass online softmax over FP16 KV-cache (`F16C`) |

### 2.10 Multi-ISA CPUID & Extended Quantization Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_cpuid_quants` | `tools/lfm2_cpuid_quants.c` | • Single-binary CPUID dynamic dispatch with GCC target attributes (`avx2,fma,f16c`)<br>• AVX2 1x1 & 1x4 kernels for `Q4_0`, `Q5_K`, and `IQ4_XS`<br>• `IQ4_XS` non-linear importance grid lookup via 1-cycle `vpshufb` (`_mm256_shuffle_epi8`)<br>• Safe portable scalar fallback with zero `SIGILL` risk and `LFM2_FORCE_ISA=scalar` testing hook |

### 2.11 Runtime Control Plane Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_runtime_features` | `tools/lfm2_runtime_features.c` | • Vocabulary raw-byte slice pre-decoding (`LFM2DecodedVocab`)<br>• RFC 8259 JSON object Pushdown Automaton (PDA) grammar logit masking<br>• LFM2 Pythonic tool-call grammar PDA enforcement (`[func(k=v)]`)<br>• `<think>` reasoning token budget lifecycle and deterministic forced closing sequence emission<br>• GQA attention-sink KV-cache shifting with AVX2 RoPE re-rotation ($-\Delta \cdot \theta_i$)<br>• ShortConv state invariance during attention-sink shifting (0 memory movement or rotation)<br>• Bit-exact numerical parity between AVX2 and scalar shift fallback |

### 2.12 Sparse MoE & Hybrid State Checkpoint Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_moe_checkpoint` | `tools/lfm2_moe_checkpoint.c` | • MoE layer binding and dense layer (0–1) fallback detection<br>• Zero-copy 3D tensor slicing (`[ne0, ne1, n_exp]`) and pointer arithmetic over `mmap`<br>• Sigmoid router + `exp_probs_b` selection bias ranking parity & unbiased weight normalization ($\sum w_k = 1.0$)<br>• Full sparse MoE FFN forward pass execution with residual accumulation<br>• Hybrid state checkpoint lifecycle (save, mutate, restore bit-exact parity)<br>• Multi-step speculative draft rejection and instant rollback verification<br>• Multi-threaded vs single-threaded MoE down-projection parity (max diff: `0.00000000`) |

### 2.13 Streaming UTF-8 Boundary & CLI Signal Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_cli_stream` | `tools/lfm2_cli_stream.c` | • Multi-byte UTF-8 boundary chunking (2-, 3-, 4-byte codepoint assembly without replacement chars)<br>• Multi-token stop string detection (`--stop`) with prefix holdback and rollback flush<br>• Two-stage non-restarting `SIGINT` signal handler (first aborts generation turn, second exits REPL)<br>• Pipe vs TTY stdout/stderr telemetry routing detector |

### 2.14 System Runtime, OOM Guard & Prompt Cache Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_sys_cache` | `tools/lfm2_sys_cache.c` | • Direct `/proc/meminfo` parser (`MemAvailable`, `Cached`, `SwapFree`)<br>• Dynamic hybrid memory footprint planner and safe context clamper (OOM killer prevention)<br>• 2 MB HugePage arena allocator (`MADV_HUGEPAGE`) with zero TLB miss alignment<br>• Sequential 4 KB stride page-fault warmup (`MADV_WILLNEED`) with live `MB/s` console progress bar<br>• 64-bit FNV-1a model hyperparameter & routing table fingerprinting<br>• Exact full-prefix prompt cache (`--prompt-cache`, `.lfmcache` v1) save and $< 1\ \text{ms}$ restoration<br>• Prefix mismatch rejection (enforcing ShortConv rolling state invariance)<br>• Payload checksum corruption / disk tamper detection |

### 2.15 Public C11 ABI & Modern Python SDK Tests (Phase 4)

| Test Binary / Suite | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_lfm2_api` | `src/lfm2_api.c`, `include/lfm2.h`, `lib/liblfm2.so` | • Default context and sampling parameter population<br>• Model introspection and architecture discovery<br>• Tokenization and detokenization routines<br>• Pull-based UTF-8 chunk generation (`lfm2_context_next_utf8`) without background threads<br>• Hybrid state checkpointing (`lfm2_checkpoint_create` & `lfm2_checkpoint_restore`) with bit-exact rollback<br>• Disk-backed prompt cache saving and loading (`lfm2_prompt_cache_save` / `lfm2_prompt_cache_load`)<br>• L2-normalized text embedding extraction (`lfm2_embed_text`) for both `LAST_TOKEN` and `MEAN` pooling<br>• Zero-copy raw logit vector pointer accessor (`lfm2_context_get_logits_ptr`) NULL-safe verification across all vocab tokens |
| `tests/test_python_lfm2.py` | `python/lfm2.py` | • Pure Python `ctypes` bindings loading `liblfm2.so` with zero internal symbol leakage<br>• Model info inspection (`arch`, `n_layers`, `n_shortconv_layers`, `n_gqa_layers`)<br>• Single-threaded generator streaming (`for chunk in ctx.stream_chat(...)`) without mojibake<br>• Automatic context branching rollback context manager (`with ctx.branch():`)<br>• L2-normalized embedding extraction with unit vector verification ($\|\mathbf{e}\| = 1.000000$)<br>• Grammar constraint invocation (`GrammarType.JSON_OBJECT`)<br>• Direct tokenization, prefill, and zero-copy logit extraction (`ctx.get_logits()` and `ctx.get_logits_ptr()`) |
| `tests/verify_parity.py` (`make test-parity`) | `tests/verify_parity.py` | • 5-Tier numerical parity & contract verification suite:<br>&nbsp;&nbsp;1. BPE Tokenizer special tags & ChatML contract<br>&nbsp;&nbsp;2. Batched GEMM (B=8) vs Sequential Decode (B=1) logit parity<br>&nbsp;&nbsp;3. Hybrid Checkpoint Rollback bit-exactness (0.0 ULP on 65,536 logits)<br>&nbsp;&nbsp;4. L2 Embedding Norm invariant ($\|\mathbf{e}\| = 1.0 \pm 10^{-3}$)<br>&nbsp;&nbsp;5. External Golden Oracle comparison against `.lfm_golden` binary fixture or live Hugging Face FP32 model with SLA thresholds (Cosine $\ge 0.9930$, KL $\le 0.0650$, Top-5 overlap $\ge 4/5$) |

### 2.16 Production Hardening & Numerical Boundary Tests (Phase 4)

| Test Binary | Target Component | Verifications |
| :--- | :--- | :--- |
| `bin/test_hardening` | `include/lfm2_simd_avx2.h`, `tools/lfm2_threadpool.c`, `src/lfm2_api.c` | • FTZ and DAZ mode active in MXCSR register preventing microcode assist penalties (~100+ cycle stalls)<br>• Worker threadpool inheritance of FTZ/DAZ across all physical CPU workers<br>• Non-256 multiple column dimensions ($N_{\text{cols}} \pmod{256} \ne 0$) zero-padded activation quantization (`100`, `128`, `300`, `2688`, `5632`)<br>• Dot products (`Q4_K`, 1x4 microkernel, `Q6_K`, `Q8_0`) and unified matrix-vector multiplier boundary safety with zero memory overreads<br>• C-API context creation, FTZ status, and finite zero-copy logit exposure |

### 2.16 Standalone LFM2 Tools & Benchmarks Verification

| Test Target | Target Component | Verifications |
| :--- | :--- | :--- |
| `make test-c-bins` | `bin/lfm2_loader` | Zero-copy GGUF v3 parser and metadata inspector on hybrid LFM2 checkpoints |
| `make test-c-bins` | `bin/lfm2_bpe` | Zero-copy GGUF BPE tokenizer and merge inspector on GGUF checkpoints |
| `make test-c-bins` | `bin/lfm2_forward` | 8-step autoregressive decode pass on hybrid GGUF checkpoint with ShortConv and GQA |
| `make test-c-bins` | `bin/lfm2_chat` | Interactive Chat REPL lifecycle test: logit-space Min-P filtering, sliding-window repetition penalty, and clean exit |
| `make test-c-bins` | `bin/lfm2_threadpool` | Standalone physical-core spin-barrier benchmark: ~260 ns round-trip latency, 24+ GB/s AVX2 bandwidth, static verification |
| `make test-c-bins` | `bin/lfm2_prefill_bench` | Standalone batched prefill benchmark: rank-1 outer product GEMM speedup and bit-exact parity |
| `make test-c-bins` | `bin/lfm2_decode_1x4_bench` | Standalone 1x4 row-unrolled and register-resident SwiGLU FFN benchmark |
| `make test-c-bins` | `bin/lfm2_kernels_fast_bench` | Standalone fast kernels benchmark: planar ShortConv, RoPE LUT, and FP16 online softmax |
| `make test-c-bins` | `bin/lfm2_cpuid_quants` | Standalone CPUID dispatch & quantization benchmark for `Q4_0`, `Q5_K`, and `IQ4_XS` |
| `make test-c-bins` | `bin/lfm2_runtime_features` | Standalone runtime control verification: grammar PDA masking, think budget, and KV sink shifting |
| `make test-c-bins` | `bin/lfm2_moe_checkpoint` | Standalone sparse MoE & hybrid state checkpoint benchmark: 3D tensor slicing, Top-4 Sigmoid routing, and $< 20\ \mu\text{s}$ rollback |
| `make test-c-bins` | `bin/lfm2_cli_stream` | Standalone streaming UTF-8 boundary assembly, stop holdback, and two-stage SIGINT guard |
| `make test-c-bins` | `bin/lfm2_sys_cache` | Standalone Linux OOM guard, 4KB page warmup, and disk-backed hybrid prompt cache verifier |

---

## 3. Memory Safety & Sanitizers

The test suite includes a dedicated target for memory sanitization:
```bash
make test-asan
```

This compiles all engine modules and test binaries with:
`-fsanitize=address,undefined -fno-omit-frame-pointer`

It verifies:
1. **Zero memory leaks:** All allocated buffers, arenas, strings, and mmap regions are completely freed.
2. **Zero buffer overflows:** Out-of-bounds reads or writes in vector loops are caught immediately.
3. **Zero undefined behavior:** Misaligned pointers, signed integer overflows, or invalid bit shifts will trigger immediate assertion failure.

---

## 4. Phase-by-Phase Testing Plan

### Phase 1: Reference Engine (Status: 7/7 PASSED)
- [x] Memory Arena Allocator Test (`test_arena`)
- [x] Mathematical Primitives Test (`test_math`)
- [x] GGUF Binary Parser Test (`test_gguf`)
- [x] BPE Tokenizer Test (`test_tokenizer`)
- [x] LIV ShortConv PyTorch Parity Test (`test_conv1d`)
- [x] GQA + QK-Norm + RoPE PyTorch Parity Test (`test_attention`)
- [x] End-to-end forward pass test (`test_forward`)
- [x] AddressSanitizer & UndefinedBehaviorSanitizer zero-defect verification

### Phase 2: High-Performance SIMD & Quantization (Status: 11/11 PASSED)
- [x] **`test_simd_avx2`:** AVX2 vector routines validated against scalar float loops across 12 boundary lengths.
- [x] **`test_quants`:** Dequantization and dot-product parity tests for `Q8_0` and `Q4_0` formats against unquantized FP32 ($< 10^{-6}$).
- [x] **`test_simd_kquants`:** AVX2 `Q8_K` pre-quantization, `bsums[16]` zero-point cancellation, and integer SIMD dot products for `Q4_K` and `Q6_K`.
- [x] **`test_fused_conv`:** Register-resident fused LIV kernel matching unfused reference ($< 10^{-6}$ output, bit-exact states).
- [x] **`test_threads`:** Multi-threaded row-chunked parallel MatVec stress test with generation-counter barrier synchronization.
- [x] **Sanitizer Verification:** All 11 tests pass under AddressSanitizer and UndefinedBehaviorSanitizer with zero memory leaks.

### Phase 3: Packaging, Web Studio, Language Bindings & Exact BPE Engine (Status: Complete)
- [x] **`test_c_api`:** Public C ABI lifecycle, metadata getters, buffer generation, and streaming callbacks.
- [x] **`test_bpe`:** Exact priority-ordered GGUF BPE rank merging, GPT-2 byte mapping, pre-tokenization regex, special tag preservation, and min-heap version invalidation.
- [x] **`test_lfm2_loader`:** Zero-copy C11 GGUF v3 inspector verifying tensor offsets, types, and array metadata.
- [x] **`test_lfm2_bpe`:** Standalone C11 exact BPE merge engine inspector on GGUF checkpoints.
- [x] **`test_lfm2_forward`:** Standalone C11 forward pass engine verifying hybrid ShortConv + GQA decode loops.
- [x] **`test_lfm2_chat`:** Interactive terminal chat REPL verifying logit-space Min-P sampling and multi-turn state preservation.
- [x] **`test_lfm2_threadpool`:** C11 lock-free spin-barrier thread pool with physical core topology discovery, futex idle sleep, and MT MatMul bit-exact parity.
- [x] **`test-python`:** Python ctypes bindings test (`LFMEngine`) verifying zero-dependency FFI.
- [x] **`test-http`:** Automated integration test verifying embedded Web Studio (`GET /`), metadata (`GET /health`), and chat completions (`POST /v1/chat/completions`).
- [x] **`static-test`:** Standalone static ELF executable packaging verification (`statically linked` for all 8 static targets).

### Phase 4: Advanced Architectural Kernels & Runtime Control Plane (Status: 26/26 Automated C Bins + 14 Standalone Tools PASSED)
- [x] **`test_optimizations`:** Bit-exact parity between scalar and fused ShortConv + SwiGLU kernels.
- [x] **`test_prefill_batch`:** Multi-token parallel activation pre-quantization into `block_q8_K` and batched rank-1 outer product GEMM.
- [x] **`test_decode_1x4`:** 1x4 row-unrolled AVX2 MatVec amortizing activation loads across 4 weight rows and register-resident fused SwiGLU FFN.
- [x] **`test_kernels_fast`:** Planar ShortConv 3-tap kernel with direct L1-hot quantization, precomputed RoPE LUT (`LFM2RoPELUT`), and FlashAttention-CPU single-pass online softmax over FP16 KV-cache.
- [x] **`test_cpuid_quants`:** Multi-ISA dynamic CPUID dispatch with GCC target attributes, AVX2 `Q4_0`, `Q5_K`, and `IQ4_XS` (via 1-cycle `vpshufb`), and safe portable scalar fallback without `SIGILL` crashes.
- [x] **`test_runtime_features`:** Zero-allocation byte-level PDA grammar logit masking (RFC 8259 JSON & Pythonic tool calls), `<think>` reasoning token budget controller with forced closing sequence, and GQA attention-sink KV-cache shifting with AVX2 RoPE re-rotation.
- [x] **`test_moe_checkpoint`:** Sparse MoE layer binding, 3D tensor slicing over `mmap`, Sigmoid + bias routing parity, unbiased Top-4 weight normalization, fused 1x4 MoE down projection, and hybrid state checkpointing/rollback lifecycle.
- [x] **`test_cli_stream`:** Multi-byte UTF-8 boundary assembly without replacement characters, rolling stop-string holdback, and two-stage SIGINT handler.
- [x] **`test_sys_cache`:** Linux `/proc/meminfo` OOM guard, 2 MB HugePage arena (`MADV_HUGEPAGE`), 4 KB stride sequential page-fault warmup, and disk-backed hybrid prompt cache (`--prompt-cache`, `.lfmcache` v1) with exact-prefix enforcement and checksum tamper detection.
- [x] **`test_lfm2_api` & `test_python_lfm2`:** Public C11 ABI unit tests and modern Python SDK tests covering single-threaded pull streaming generator, hybrid context branching rollback, disk prompt caching, and L2-normalized embeddings.
- [x] **`test_lfm2_server`:** Zero-dependency Linux `epoll` HTTP and Server-Sent Events (SSE) test suite validating `GET /health`, `GET /`, `GET /v1/models`, `OPTIONS` preflight, streaming and non-streaming chat completions, L2 embeddings, and instant client disconnect cancellation (`EPOLLRDHUP`).
- [x] **14 Standalone Tools & Benchmarks:** `bin/lfm2_loader`, `bin/lfm2_bpe`, `bin/lfm2_forward`, `bin/lfm2_chat`, `bin/lfm2_threadpool`, `bin/lfm2_prefill_bench`, `bin/lfm2_decode_1x4_bench`, `bin/lfm2_kernels_fast_bench`, `bin/lfm2_cpuid_quants`, `bin/lfm2_runtime_features`, `bin/lfm2_moe_checkpoint`, `bin/lfm2_cli_stream`, `bin/lfm2_sys_cache`, `bin/lfm2_server`.
- [x] **16 Standalone Statically Linked ELF Binaries:** `bin/lfm-run-static`, `bin/lfm-server-static`, `bin/lfm2-loader-static`, `bin/lfm2-bpe-static`, `bin/lfm2-threadpool-static`, `bin/lfm2-forward-static`, `bin/lfm2-chat-static`, `bin/lfm2-prefill-static`, `bin/lfm2-decode-1x4-static`, `bin/lfm2-kernels-fast-static`, `bin/lfm2-cpuid-quants-static`, `bin/lfm2-runtime-features-static`, `bin/lfm2-moe-checkpoint-static`, `bin/lfm2-cli-stream-static`, `bin/lfm2-sys-cache-static`, `bin/lfm2-server-static`.
- [x] **Complete Test Suite:** 100% pass rate under normal build and AddressSanitizer/UBSan runs.


