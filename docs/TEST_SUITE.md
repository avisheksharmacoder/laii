# Test Suite Specification & Quality Assurance

> For the full documentation index, see [docs/README.md](README.md).

This document details the automated test framework, numerical parity contract, and memory/concurrency sanitizer gates for the **LFM2.c (`laii`)** inference engine.

---

## 1. Quick Commands

```bash
make test                         # Run 32 C test binaries, 16 standalone tools, HTTP/Python checks, and static-test
make test-c-bins                  # Run the 32 automated C test binaries, 16 standalone tools, and test-http
make test-python                  # Run Python bindings, Python LFM2 SDK, and verify_parity.py
make test-jni                     # Compile and run Java JNI harness (tests/TestJNI.java)
make test-kotlin-app              # Compile and run Kotlin Android ViewModel harness (tests/AndroidAppRunner.kt)
make parity MODEL=<path-to.gguf>  # Run 5-tier numerical & contract parity suite against a real GGUF checkpoint
make asan                         # Rebuild with AddressSanitizer + UBSan and run C test suite
make tsan                         # Rebuild threadpool & CLI with ThreadSanitizer and run race-detection checks
make static-test                  # Verify statically linked ELF binaries and live lfm2-server-static endpoints
```

---

## 2. Test Suite Architecture

```text
                                 [ Full Suite: make test ]
                                             |
   +--------------------+--------------------+--------------------+--------------------+
   |                    |                    |                    |                    |
   v                    v                    v                    v                    v
[ Core & Parity ]  [ SIMD, Kernels & MoE ] [ ABI, Server & Mob ] [ Bindings & Android ] [ Packaging & CI ]
- test_arena       - test_simd_avx2      - test_c_api          - test_python_bindings - static-test
- test_math        - test_quants         - test_lfm2_api       - test_python_lfm2     - static-musl
- test_gguf        - test_simd_kquants   - test_lfm2_server    - verify_parity.py     - test-asan (ASan/UBSan)
- test_tokenizer   - test_fused_conv     - test_hardening      - test-http            - test-tsan (TSan)
- test_bpe         - test_optimizations  - test_simd_arm64     - test-jni
- test_conv1d      - test_prefill_batch  - test_android_io     - test-kotlin-app
- test_attention   - test_decode_1x4     - test_speculative_pld
- test_forward     - test_kernels_fast
                   - test_cpuid_quants
                   - test_runtime_features
                   - test_moe_checkpoint
                   - test_cli_stream
                   - test_sys_cache
                   - test_threads / test_lfm2_threadpool
```

---

## 3. Automated C Test Binaries (`make test-c-bins`)

`TEST_BINS` in [`Makefile`](../Makefile) comprises **32 C test binaries** executed sequentially by `make test-c-bins`:

### 3.1 Core Engine, Tokenizer & PyTorch Numerical Parity

| Test Binary | Component | Key Verifications |
| :--- | :--- | :--- |
| `bin/test_arena` | `src/arena.c` | Strict 64-byte cache-line alignment across 10 allocation sizes, pointer reset, OOM boundary guards |
| `bin/test_math` | `src/lfm_forward.c` | RMSNorm stability, SiLU accuracy, RoPE norm preservation, FP32 MatVec correctness |
| `bin/test_gguf` | `src/gguf.c` | GGUF v3 header validation, `lfm2.*` metadata parsing, 32-byte aligned tensor directory |
| `bin/test_tokenizer` | `src/lfm_tokenizer.c` | Vocabulary loading, byte-level BPE encoding, `<0xHH>` fallback decoding |
| `bin/test_bpe` | `tools/lfm2_bpe.c` | Priority-ordered GGUF BPE rank merging, GPT-2 byte-to-Unicode round-trip, pre-tokenizer regex, special tags (`<|startoftext|>`, `<|im_start|>`, `<|im_end|>`) |
| `bin/test_conv1d` | `src/lfm_forward.c` | LIV Gated ShortConv vs. PyTorch `F.conv1d` reference (`tools/generate_reference_tensors.py`; max diff $< 10^{-4}$, measured $2.98 \times 10^{-8}$) |
| `bin/test_attention` | `src/lfm_forward.c` | GQA + RoPE + QK-Norm vs. PyTorch reference (max diff $< 10^{-4}$, measured $9.69 \times 10^{-8}$) |
| `bin/test_forward` | `src/lfm_engine.c` | Full 2-layer hybrid forward pass on `tests/lfm2_toy.gguf` (prefill, KV-cache, token streaming) |

### 3.2 Gen-1 SIMD, Fused Kernels, Threads & Legacy Interfaces

| Test Binary | Component | Key Verifications |
| :--- | :--- | :--- |
| `bin/test_simd_avx2` | `src/lfm_simd.c` | AVX2 `vec_dot_f32` across 12 alignment boundaries ($N \in [1, 1024]$), `vec_rmsnorm`, `vec_mul`, `vec_add` |
| `bin/test_quants` | `src/lfm_simd.c` | `Q8_0` and `Q4_0` dot product parity vs. FP32 reference ($< 10^{-6}$ diff) |
| `bin/test_simd_kquants` | `include/lfm2_simd_avx2.h` | `Q8_K` activation pre-quantization (`bsums[16]`), AVX2 `Q4_K` and `Q6_K` integer SIMD dot products, OpenMP MatVec scaling |
| `bin/test_fused_conv` | `src/lfm_fused.c` | Fused LIV kernel equivalence vs. unfused reference ($< 10^{-6}$ diff output, bit-exact rolling state updates) |
| `bin/test_threads` | `src/lfm_threadpool.c` | Generation-counter barrier stress test across 50 parallel MatVec dispatches ($\Delta < 10^{-5}$) |
| `bin/test_c_api` | `src/lfm_c_api.c` | Gen-1 C ABI lifecycle, metadata getters, sync generation, callback streaming, tokenize/detokenize |

### 3.3 LFM2 Production Kernels, Sparse MoE & Runtime Subsystems

| Test Binary | Component | Key Verifications |
| :--- | :--- | :--- |
| `bin/test_lfm2_threadpool` | `tools/lfm2_threadpool.c` | Sysfs physical core discovery, EAS `cpu_capacity`/`cpuinfo_max_freq` scanner, mobile `big.LITTLE` policy ($N_{\text{big}}-1$, ceiling $N_{\text{total}}-2$, Worker 0 pinned to Prime), `FUTEX_WAIT_PRIVATE` idle sleep, work-stealing, bit-exact MT parity |
| `bin/test_optimizations` | `tools/lfm2_forward.c` | Bit-exact parity between scalar and fused ShortConv + SwiGLU kernels |
| `bin/test_prefill_batch` | `tools/lfm2_prefill_batch.c` | Multi-token parallel `block_q8_K` quantization and rank-1 outer-product GEMM parity vs. sequential decode |
| `bin/test_decode_1x4` | `tools/lfm2_decode_1x4.c` | 1x4 row-unrolled AVX2 MatVec and register-resident fused SwiGLU FFN |
| `bin/test_kernels_fast` | `tools/lfm2_kernels_fast.c` | Planar 3-tap ShortConv kernel, precomputed `LFM2RoPELUT`, fused QK-norm, and single-pass FP16 online softmax |
| `bin/test_cpuid_quants` | `tools/lfm2_cpuid_quants.c` | Single-binary CPUID dispatch, `Q4_0`/`Q5_K`/`IQ4_XS` (`vpshufb`) kernels, and `LFM2_FORCE_ISA=scalar` fallback |
| `bin/test_runtime_features` | `tools/lfm2_runtime_features.c` | `LFM2DecodedVocab` PDA JSON & Pythonic tool-call grammar masker, `<think>` budget controller, GQA attention-sink KV shifting + AVX2 RoPE counter-rotation |
| `bin/test_moe_checkpoint` | `tools/lfm2_moe_checkpoint.c` | `lfm2moe` 3D `mmap` tensor slicing, Sigmoid + `exp_probs_b` Top-4 routing parity, parallel MoE FFN, and $< 20\ \mu\text{s}$ `LFM2Checkpoint` rollback |
| `bin/test_cli_stream` | `tools/lfm2_cli_stream.c` | Multi-byte UTF-8 boundary reassembly, rolling `--stop` prefix holdback, two-stage `SIGINT`, TTY vs. pipe detection |
| `bin/test_sys_cache` | `tools/lfm2_sys_cache.c` | `/proc/meminfo` OOM context clamper, `MADV_HUGEPAGE` arena, page-stride warmup, `.lfmcache` v1 save/load, exact-prefix enforcement, FNV-1a tamper rejection |

### 3.4 Public C11 ABI, `epoll` Server, Hardening, ARM64 & Speculative PLD

| Test Binary | Component | Key Verifications |
| :--- | :--- | :--- |
| `bin/test_lfm2_api` | `src/lfm2_api.c`, `include/lfm2.h` | Context/sample defaults, pull-based `lfm2_context_next_utf8`, `LFM2Checkpoint` save/restore, `.lfmcache`, L2-normalized `lfm2_embed_text`, `lfm2_context_get_logits_ptr`, Scoped Storage `lfm2_model_load_from_fd` (`dup(fd)`), `lfm2_model_release_page_cache` |
| `bin/test_lfm2_server` | `tools/lfm2_server.c` | Non-blocking `epoll` server: `/health`, `/`, `/v1/models`, `OPTIONS`, SSE & JSON chat completions, embeddings, `EPOLLRDHUP` disconnect abort |
| `bin/test_hardening` | `include/lfm2_simd_avx2.h`, `tools/lfm2_threadpool.c`, `src/lfm2_api.c` | MXCSR FTZ/DAZ verification & worker inheritance, zero-padded non-256 column boundaries (`100`, `128`, `300`, `2688`, `5632`), finite zero-copy logits |
| `bin/test_simd_arm64` | `include/lfm2_simd_arm64.h`, `tools/lfm2_simd_arm64.c` | ARM64 mobile kernel table dispatch, `getauxval(AT_HWCAP)` detection (`HWCAP_ASIMDDP`, `HWCAP_FPHP`), `Q8_K` quant, 1x1 & 1x4 dot products across 6 formats (`Q4_K`, `Q6_K`, `Q5_K`, `IQ4_XS`, `Q4_0`, `Q8_0`), `LFM2_FORCE_ISA=scalar` override |
| `bin/test_android_io` | `include/lfm2.h`, `src/lfm2_api.c` | `sysconf(_SC_PAGESIZE)` Android 15+ 16 KB page support, `/proc/self/statm` RSS profiler, `dup(fd)` Scoped Storage mapping, dual-layer `MADV_DONTNEED` + `POSIX_FADV_DONTNEED` `lmkd` survival, background warmup |
| `bin/test_speculative_pld` | `include/lfm2_speculative_pld.h`, `tools/lfm2_speculative_pld.c` | O(1) 3-gram/2-gram lookup, stop-token bounds, `conv_trace[b][l]` rollback, $B=4$ speculative batch forward, 100% bit-exact parity vs. $M=1$ baseline |
| `bin/test_lfm3` | `tools/lfm3_engine.c`, `tests/test_lfm3.c` | LFM3 tri-hybrid ShortConv7 ring, SWA (`w=256`), Sandwich Norms, `attn_gate`, checkpoint & KV sink shift |
| `bin/test_lfm2_gui` | `include/lfm_ui.h`, `tools/lfm2_gui_worker.h`, `tests/test_lfm2_gui.c` | Headless GUI suite (no X11 display needed): layout slicing, wrapped UTF-8 + markdown code blocks, multi-line `LfmUiTextEdit` clipboard/submit, SSE JSON delta & `<think>` splitter, `.gguf` dir scanner, and worker `pthread` inference + reset against `tests/lfm2_toy.gguf` |

*For live X11 rendering smoke verification of `bin/lfm2-gui`, run `DISPLAY=:0 ./bin/lfm2-gui --smoke-frames 5 -m tests/lfm2_toy.gguf` (see [DESKTOP_GUI.md](DESKTOP_GUI.md)).*

---

## 4. Standalone Tools & Static ELF Verification

### 4.1 16 Standalone Tools (`make test-c-bins`)
`make test-c-bins` also smoke-tests all 16 standalone tools on `tests/lfm2_toy.gguf`:
`bin/lfm2_loader`, `bin/lfm2_bpe`, `bin/lfm2_forward`, `bin/lfm2_chat` (`/exit`), `bin/lfm2_threadpool`, `bin/lfm2_prefill_bench`, `bin/lfm2_decode_1x4_bench`, `bin/lfm2_kernels_fast_bench`, `bin/lfm2_cpuid_quants`, `bin/lfm2_runtime_features`, `bin/lfm2_moe_checkpoint`, `bin/lfm2_cli_stream`, `bin/lfm2_sys_cache`, `bin/lfm2_server` (live `/health` check), `bin/lfm2_simd_arm64`, and `bin/lfm2_speculative_pld`.

### 4.2 Static ELF Binaries (`make static` / `make static-test`)
* `make static` compiles **19 static binaries** (`bin/lfm2-static`, `bin/lfm-run-static`, `bin/lfm-server-static`, and static builds of all 16 standalone tools above).
* Notably, `bin/lfm2-server-static` compiles `tools/lfm2_server.c` together with `src/lfm2_api.c` (`-DLFM2_API_NO_SERVER -DLFM2_DISABLE_JNI`) for full inference support with zero external `.so` dependencies.
* `make static-test` verifies `statically linked` ELF headers via `file`, runs `./bin/lfm-run-static -h`, and boots `bin/lfm2-server-static` against `tests/lfm2_toy.gguf` to verify live `/health` and `/v1/models` responses.

---

## 5. Language Bindings, Android JNI/Kotlin & HTTP Integration

| Make Target | Source / Test | What Is Verified |
| :--- | :--- | :--- |
| `make test-python` | `tests/test_python_bindings.py`, `tests/test_python_lfm2.py`, `tests/verify_parity.py` | Gen-1 `bindings/python/laii.py` (`liblfm.so`), modern `python/lfm2.py` (`liblfm2.so`: streaming, `ctx.branch()`, L2 embeddings, JSON grammar, `from_fd()`, `release_page_cache()`), plus 5-tier parity checks |
| `make test-jni` | `android/src/main/java/ai/liquid/lfm2/*.java`, `tests/TestJNI.java` | Native JNI bridge (`src/lfm2_jni.c` inside `liblfm2.so`) model load, Scoped Storage FD load, metadata, streaming callback, context lifecycle (`MODEL=` optional) |
| `make test-kotlin-app` | `android/src/main/kotlin/ai/liquid/lfm2/*.kt`, `android/.../app/*.kt`, `tests/AndroidAppRunner.kt` | Full Kotlin wrapper (`LFM2Model.kt`), `ChatViewModel.kt`, and simulated app flow on Linux JVM (`MODEL=` optional; interactive REPL via `make run-kotlin-app`) |
| `make test-http` | `tests/test_http_server.sh` | Gen-1 `./lfm-run --server`: `/health`, `/v1/models`, `GET /` HTML UI, `POST /v1/chat/completions`, clean `SIGTERM` shutdown |

---

## 6. Numerical Parity Contract (`make test-parity` / `make parity`)

[`tests/verify_parity.py`](../tests/verify_parity.py) enforces a 5-tier quality contract:
1. **Tier 1:** GGUF BPE special tag tokenization (`<|startoftext|>`, `<|im_start|>`, `<|im_end|>`) & ChatML formatting contract.
2. **Tier 2:** Batched Prefill GEMM vs. sequential single-token decode across all output logits (Cosine $\approx 1.0$, MaxDiff $= 0.0$).
3. **Tier 3:** Hybrid `LFM2Checkpoint` (`ctx.branch()`) rollback bit-exactness across all vocabulary logits ($0.0$ ULP diff).
4. **Tier 4:** L2-normalized embedding unit vector invariant ($\|\mathbf{e}\|_2 = 1.0 \pm 10^{-3}$).
5. **Tier 5:** Golden oracle verification (`.lfm_golden` / Hugging Face FP32 reference) checking Cosine Similarity ($\ge 0.9930$), KL Divergence ($\le 0.0650$), and Top-5 token overlap ($\ge 4/5$).

---

## 7. Memory Safety, Concurrency Sanitizers & Distro Portability

### 7.1 AddressSanitizer & UndefinedBehaviorSanitizer (`make test-asan` / `make asan`)
* Rebuilds engine objects, tools, and all 32 C test binaries with `-fsanitize=address,undefined -fno-omit-frame-pointer` and executes `make test-c-bins`.
* Also validated in CI against both AVX2 and forced scalar fallback (`LFM2_FORCE_ISA=scalar`) under `LD_PRELOAD` with `verify_parity.py`.

### 7.2 ThreadSanitizer (`make test-tsan` / `make tsan`)
* Compiles `bin/lfm2-tsan`, `bin/test_lfm2_threadpool_tsan`, and `bin/test_threads_tsan` with `-fsanitize=thread -O1 -g`.
* Executes binaries under `setarch $(uname -m) -R` when available to prevent Linux high-entropy ASLR shadow-memory collisions on both `x86_64` and `aarch64`.
* Certifies zero data races across lock-free atomic dispatch/work-stealing, futex transitions, and parallel MatVec workers.

### 7.3 Distro-Agnostic `musl` Static Compilation (`make static-musl`)
* Optional target using `musl-gcc -static` (with clean guidance if `musl-tools` is not installed) to build glibc-independent static ELF binaries (`bin/lfm-run-static-musl`).
