# LAII v1.0.0 Pre-Release Checklist

This checklist outlines the four actionable tasks identified from the system hardening audit and roofline analysis before tagging the official `v1.0.0` release.

---

### Task 1: Add `test-tsan` Target to `Makefile` [COMPLETED]
- **Objective:** Formalize ThreadSanitizer verification in the test suite alongside `make test-asan`.
- **Status:** **Completed & Verified.** Added `make test-tsan` compiling `test_lfm2_threadpool` and `test_threads` with `-fsanitize=thread` and executed under `setarch x86_64 -R`. Verified 0 data races.

---

### Task 2: Add Optional `static-musl` Target to `Makefile` [COMPLETED]
- **Objective:** Support building ultra-compact, distro-agnostic static ELF binaries without glibc coupling.
- **Status:** **Completed & Verified.** Added `make static-musl` checking for `musl-gcc` and compiling stripped static binaries with fallback messaging when `musl-tools` is not installed.

---

### Task 3: Mobile Edge, ARM64 NEON & Android/Kivy Integration [COMPLETED]
- **Objective:** Enable Android Scoped Storage loading, Low Memory Killer (`lmkd`) survival, and native ARM64 NEON + DotProd (`vdotq_s32`) SIMD kernels without bloating the repo with Kivy UI code.
- **Status:** **Completed & Verified.**
  - Added native ARM64 NEON + DotProd suite in [`tools/lfm2_simd_arm64.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_simd_arm64.c) and [`include/lfm2_simd_arm64.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2_simd_arm64.h) with 1x4 row unrolling, 1-cycle `TBL` lookup (`vqtbl1q_s8`), and dynamic `getauxval(AT_HWCAP)` CPU capability detection.
  - Added `lfm2_model_load_from_fd(int fd, ...)` with `dup(fd)` in `include/lfm2.h`, `tools/lfm2_loader.c`, and `src/lfm2_api.c`.
  - Added dual-layer `lfm2_model_release_page_cache(model)` calling `madvise(MADV_DONTNEED)` and `posix_fadvise(POSIX_FADV_DONTNEED)` for Android `on_pause()` memory reclamation to survive `lmkd`.
  - Added `lfm2_model_warmup_page_cache(model)` for background sequential pre-faulting on mobile `on_resume()`.
  - Added zero-allocation `/proc/self/statm` RSS memory profiler `lfm2_get_process_rss_bytes()`.
  - Added dynamic 16 KB page warmup via `sysconf(_SC_PAGESIZE)` in `tools/lfm2_loader.c` and `tools/lfm2_sys_cache.c`.
  - Added `LFM2Model.from_fd()`, `model.release_page_cache()`, `model.warmup_page_cache()`, and `LFM2Model.get_process_rss_bytes()` in `python/lfm2.py`.
  - Added Test 8 to `tests/test_lfm2_api.c` (100% pass) and Test 7 to `tests/test_python_lfm2.py` (100% pass).
  - Added automated test in [`tests/test_simd_arm64.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_simd_arm64.c) verifying kernel table dispatch, `Q8_K` quantization, 1x1 dot products across 6 quants, and 1x4 unrolled equivalence (100% pass).
  - Added dedicated test suite in [`tests/test_android_io.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_android_io.c) verifying dynamic page sizing, RSS tracking, and `lmkd` survival hooks (100% pass).
  - Authored comprehensive developer guide in [`docs/ANDROID_KIVY_INTEGRATION.md`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/docs/ANDROID_KIVY_INTEGRATION.md) with lock-free 60 FPS coalesced token streaming and SAF `pfd.detachFd()`.

---

### Task 4: Unified Cross-Platform & Cluster-Aware Thread Pool [COMPLETED]
- **Objective:** Implement unified Linux PC and Mobile SoC `big.LITTLE` thread pool with cluster-aware core reservation.
- **Status:** **Completed & Verified.**
  - Upgraded [`tools/lfm2_threadpool.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_threadpool.c) with `LFM2CpuTopology` sysfs scanner (`max_freq_khz` and Linux EAS `cpu_capacity`).
  - Preserved PC safety invariant: $N_{\text{phys}} - 1$ cores (e.g. 4 cores -> 3 workers, leaving 1 physical core for OS/UI).
  - Implemented mobile SoC policy: default auto selects $N_{\text{big}} - 1$ cores (e.g. 3 Big cores: 1 Prime + 2 Perf, leaving 1 Big core for Kivy/UI and all Little cores for OS).
  - Implemented strict mobile ceiling: `max_safe_threads = N_total - 2` (e.g., 6 cores max on an 8-core SoC, leaving at least 1 Little and 1 Big core free to prevent thermal blackouts and Vdroop brownouts).
  - Placed Worker 0 on Prime core (`cpu_ids[0]`) to accelerate single-threaded sequential ops (RMSNorm, ShortConv, RoPE, sampling).
  - Added ARM64 `FPCR.FZ` (bit 24) and `FPCR.FZ16` (bit 19) hardware flush-to-zero in [`include/lfm2_simd_avx2.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2_simd_avx2.h).
  - Added adaptive spin iterations: 25,000 on x86, 4,000 on mobile/ARM to prevent thermal throttling and save battery.
  - Updated [`tests/test_lfm2_threadpool.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_lfm2_threadpool.c) to verify simulated 8-core SoC (3 big auto, cpu7 Prime on Worker 0, 6 core max safe ceiling). Verified under ThreadSanitizer (`make test-tsan`) with **0 data races**.

---

### Task 5: Zero-Weight Prompt-Lookup Speculative Decoding (PLD) [COMPLETED]
- **Objective:** Overcome DDR4/DDR5 memory bandwidth bottlenecks during desktop decoding without loading secondary draft models.
- **Status:** **Completed & Verified.**
  - Added `include/lfm2_speculative_pld.h` and `tools/lfm2_speculative_pld.c` implementing `LFM2NgramIndex` with O(1) 3-gram and 2-gram MurmurHash3 lookup tables ($< 1\ \mu\text{s}$ drafting).
  - Added `LFM2SpecState` with `conv_trace[LFM2_MAX_BATCH][LFM2_MAX_LAYERS]` snapshotting 16 KB ShortConv rolling states during causal evaluation, enabling $O(1)$ causal rollback in $8\ \mu\text{s}$ without recomputing previous tokens.
  - Implemented `lfm2_forward_speculative_batch` evaluating micro-batches up to $B = 8$ via batched AVX2 GEMM, projecting `output.weight` across all slots in a single DRAM weight sweep.
  - Added `tests/test_speculative_pld.c` (`bin/test_speculative_pld`) verifying hash indexing, stop-token bounds, adaptive windowing, and 100% bit-exact parity vs. $M=1$ autoregressive baseline (5/5 unit tests passed).
  - Registered `bin/lfm2_speculative_pld` benchmark and static target `bin/lfm2-speculative-pld-static` (18 static binaries total).

---

### Task 6: Stage and Commit Changes [COMPLETED]
- **Objective:** Ensure all recent documentation covering batched prefill acceleration, mobile integration, unified threadpool, ARM64 NEON SIMD kernels, and Prompt-Lookup Speculative Decoding is committed to version control.
- **Status:** **Completed & Verified.** All 33 modified and added files staged and certified across the 5 validation pillars.

---

### Task 7: Tag the `v1.0.0` Release
- **Objective:** Tag and certify the official `v1.0.0` production release of LAII.
- **Prerequisites (All Verified & Passing):**
  - [x] All 30 automated C test binaries passing 100% (`make test-c-bins`).
  - [x] Memory safety verified with zero leaks, overflows, or undefined behavior (`make test-asan`).
  - [x] Thread safety verified with zero data races under ThreadSanitizer (`make test-tsan`).
  - [x] Python SDK verified with 7/7 tests passing (`python3 tests/test_python_lfm2.py`).
  - [x] Numerical parity contract verified and certified (`make test-parity`).
  - [x] Non-blocking epoll HTTP/SSE server and Web Studio integration verified (`make test-http`).
  - [x] 18 standalone statically linked ELF executables verified (`make static-test`).
- **Action:**
  - Create annotated Git release tag: `git tag -a v1.0.0 -m "Release v1.0.0: Production-grade C11 inference engine for LFM2 & LFM2.5"`.

