# LFM2.c Documentation Hub

Welcome to the documentation for **LFM2.c (`laii`)** — a zero-dependency C11 inference engine for Liquid AI's hybrid **LFM2**, **LFM2.5**, and **LFM3** model families (`x86_64` AVX2/AVX-VNNI and `aarch64` NEON/DotProd/FP16).

---

## Recommended Reading Order

1. **Quick Start & Usage:** Start with [`../README.md`](../README.md) to build the engine (`make`), launch the native C11 X11/Xft Desktop GUI ([`DESKTOP_GUI.md`](DESKTOP_GUI.md) / `bin/lfm2-gui`), run interactive CLI chat (`bin/lfm2`), start the `epoll` OpenAI HTTP/SSE server (`bin/lfm2-server`), or use the Python `ctypes` SDK (`python/lfm2.py`).
2. **Performance & Footprint:** Read [`BENCHMARKS.md`](BENCHMARKS.md) for side-by-side benchmarks against `llama.cpp`, RSS memory reductions, cold `mmap` load latency, and DRAM bandwidth roofline analysis.
3. **Engine Design & Kernels:** Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for the mathematical specifications, GGUF loader, AVX2/AVX-VNNI & ARM64 NEON/DotProd SIMD kernels, sparse MoE routing, OOM/prompt caching, C11 ABI, `epoll` server internals, and zero-weight Prompt-Lookup Speculative Decoding.
4. **LFM3 vs. LFM2/LFM2.5 Architecture & Porting:** Read [`LFM3_VS_LFM2_GUIDE.md`](LFM3_VS_LFM2_GUIDE.md) for the complete comparison between LFM2, LFM2.5, and LFM3 (tri-hybrid SWA + 7-tap ShortConv + Sandwich Norm + `attn_gate` + partial RoPE), GGUF tensor inventory, codebase gap audit, and zero-regression optimization blueprint. For the step-by-step engineering roadmap, see [`LFM3_IMPLEMENTATION_PLAN.md`](LFM3_IMPLEMENTATION_PLAN.md).
5. **Android Deployment:**
   - Native Android Studio (Kotlin / Java / Jetpack Compose / CMake JNI): [`ANDROID_KOTLIN_INTEGRATION.md`](ANDROID_KOTLIN_INTEGRATION.md)
   - Python / Kivy (`buildozer` / `p4a`): [`ANDROID_KIVY_INTEGRATION.md`](ANDROID_KIVY_INTEGRATION.md)
6. **Verification & QA:** Read [`TEST_SUITE.md`](TEST_SUITE.md) for all 30 C test binaries, Python SDK tests, Java/Kotlin JNI harnesses, 5-tier parity verification, static binary checks, and ASan/TSan concurrency gates.
7. **Engineering History:** Read [`ROADMAP.md`](ROADMAP.md) for the five-phase v1.0.0 development milestones and module lineage.

---

## Documentation Directory

| Document | Audience | Scope |
| :--- | :--- | :--- |
| [`../README.md`](../README.md) | All Users | Building from source, CLI flags, desktop GUI, REST/SSE endpoints, Python & Kotlin quickstarts |
| [`DESKTOP_GUI.md`](DESKTOP_GUI.md) | Desktop Users & UI Developers | Native C11 + X11/Xft desktop app (`bin/lfm2-gui`), single-header `include/lfm_ui.h`, Local & Server SSE modes |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Systems & Kernel Engineers | Hybrid LIV ShortConv + GQA + MoE math, SIMD microkernels, memory/cache architecture, server & mobile runtime |
| [`LFM3_VS_LFM2_GUIDE.md`](LFM3_VS_LFM2_GUIDE.md) | Systems & Kernel Engineers | LFM3 vs. LFM2/LFM2.5 architectural diff, empirical GGUF layout, codebase gap audit & zero-regression optimization blueprint |
| [`LFM3_IMPLEMENTATION_PLAN.md`](LFM3_IMPLEMENTATION_PLAN.md) | Maintainers & Contributors | Phased C11 implementation plan, LFM2 vs. LFM3 runtime context matrix, micro-details, and parity/sanitizer verification gates |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Performance Engineers | Empirical latency, prefill/decode tok/s, RSS comparisons vs. `llama.cpp`, and roofline validation |
| [`ANDROID_KOTLIN_INTEGRATION.md`](ANDROID_KOTLIN_INTEGRATION.md) | Android Native Developers | CMake/NDK setup, Kotlin Coroutines `Flow`, Java SDK, SAF Scoped Storage `fd`, and `lmkd` memory hooks |
| [`ANDROID_KIVY_INTEGRATION.md`](ANDROID_KIVY_INTEGRATION.md) | Kivy / Python Mobile Developers | NDK cross-compile, `p4a` recipe, `buildozer.spec`, 60 FPS async UI streaming, and SAF `detachFd()` |
| [`TEST_SUITE.md`](TEST_SUITE.md) | Contributors & CI Maintainers | Automated C/Python/JNI/Kotlin/HTTP suites, `verify_parity.py`, ASan/UBSan, TSan, and static/musl checks |
| [`ROADMAP.md`](ROADMAP.md) | Maintainers | Phase 1–5 v1.0.0 milestone record and legacy Gen-1 (`src/lfm_*`) vs. LFM2 (`src/lfm2_api.c`, `tools/lfm2_*`) mapping |

---

## Codebase Entry Points at a Glance

- **Public C11 ABI & JNI Bridge:** [`include/lfm2.h`](../include/lfm2.h), [`src/lfm2_api.c`](../src/lfm2_api.c), [`src/lfm2_jni.c`](../src/lfm2_jni.c)
- **Core Kernels, CLI, Desktop GUI & HTTP/SSE Server:** [`tools/`](../tools) (`lfm2_loader.c`, `lfm2_bpe.c`, `lfm2_forward.c`, `lfm2_decode_1x4.c`, `lfm2_prefill_batch.c`, `lfm2_kernels_fast.c`, `lfm2_cpuid_quants.c`, `lfm2_simd_arm64.c`, `lfm3_engine.c`, `lfm2_runtime_features.c`, `lfm2_moe_checkpoint.c`, `lfm2_speculative_pld.c`, `lfm2_sys_cache.c`, `lfm2_threadpool.c`, `lfm2_cli_stream.c`, `lfm2_chat.c`, `lfm2_server.c`, `lfm2_gui.c`, `lfm2_gui_worker.h`) & [`include/lfm_ui.h`](../include/lfm_ui.h), [`include/lfm_defaults.h`](../include/lfm_defaults.h)
- **Python SDK & Android Kotlin/Java SDK:** [`python/lfm2.py`](../python/lfm2.py), [`android/`](../android)
- **Automated Verification:** [`tests/`](../tests) and [`Makefile`](../Makefile)

