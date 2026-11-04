# Codebase Topology & File Responsibilities

This document provides a complete layout of the LAII repository, detailing every directory, source file, header, and test harness.

---

## 🗂️ Directory Map

```text
laii/
├── aidocs/                  # Dedicated LLM Agent Knowledge Hub
│   ├── README.md            # Hub index & quick navigation
│   ├── AGENT_PRIMER.md      # 7 Golden Rules & quick context primer
│   ├── ARCHITECTURAL_INVARIANTS.md # Mathematical formulations & hybrid state rules
│   ├── MEMORY_AND_STATE_MAP.md     # Arena allocators, KV-cache, OOM guard & HugePages
│   ├── CODEBASE_TOPOLOGY.md # File layout, symbols & dual-build boundary
│   ├── SIMD_AND_KERNELS.md  # AVX2/FMA intrinsics, quantization formats & CPUID
│   ├── EXTENSION_PLAYBOOK.md# Developer cookbooks for adding features & tests
│   └── COMMON_PITFALLS.md   # Fatal traps, alignment bugs & sanitizer failures
├── bindings/
│   └── python/
│       └── laii.py          # Zero-dependency Python ctypes FFI wrapper
├── docs/                    # Detailed architectural & phase specifications
│   ├── ARCHITECTURE.md      # Mathematical reference and full cache analysis
│   ├── PHASE1_SPECIFICATION.md
│   ├── PHASE2_SPECIFICATION.md
│   ├── PHASE3_SPECIFICATION.md
│   ├── PHASE4_SPECIFICATION.md
│   ├── ROADMAP.md           # 4-Phase milestones checklist
│   └── TEST_SUITE.md        # Quality assurance protocol & test catalog
├── include/                 # Public & engine internal C headers
│   ├── lfm2.h               # Public C11 ABI (opaque handles, POD structs, 27 API functions)
│   ├── arena.h              # 64-byte aligned linear memory arena
│   ├── gguf.h               # GGUF v2/v3 parser & tensor directory
│   ├── lfm_model.h          # Model config & layer tensor definitions
│   ├── lfm_simd.h           # AVX2/FMA vector dot products & RMSNorm
│   ├── lfm_fused.h          # Register-resident fused LIV ShortConv kernel
│   ├── lfm_threadpool.h     # POSIX thread pool with generation barrier
│   ├── lfm_forward.h        # Hybrid forward pass execution
│   ├── lfm_tokenizer.h      # BPE tokenizer encoding/decoding & byte fallback
│   ├── lfm_engine.h         # High-level engine lifecycle & autoregressive generation
│   ├── lfm_c_api.h          # Public C ABI interface for shared/static libraries
│   ├── lfm_server.h         # Embedded HTTP REST server interface
│   └── lfm_sys_cache.h      # System cache & OOM guard integration header
├── src/                     # Core library implementation
│   ├── lfm2_api.c           # Public C11 ABI implementation (-fvisibility=hidden)
│   ├── arena.c              # Linear arena allocator implementation
│   ├── gguf.c               # mmap GGUF loader & metadata dictionary
│   ├── lfm_model.c          # Tensor name binding & config extractor
│   ├── lfm_simd.c           # SIMD dot products (FP32, Q8_0, Q4_0) & RMSNorm
│   ├── lfm_fused.c          # Fused LIV ShortConv kernel implementation
│   ├── lfm_threadpool.c     # Multi-threaded parallel row-chunked MatVec
│   ├── lfm_forward.c        # Hybrid forward pass implementation
│   ├── lfm_tokenizer.c      # Tokenizer decoding & fallback
│   ├── lfm_engine.c         # Engine prefill, autoregressive loop & prompt cache
│   ├── lfm_c_api.c          # Public C ABI implementation
│   ├── lfm_server.c         # Embedded POSIX HTTP socket REST server
│   ├── lfm_webui.h          # In-memory single-page glassmorphic Web Studio
│   ├── main.c               # CLI executable (`lfm-run`)
│   └── server_main.c        # Dedicated server executable (`lfm-server`)
├── python/                  # High-Performance Python SDK (ctypes)
│   ├── __init__.py          # Python package exports (`LFMModel`, `LFMContext`, etc.)
│   └── lfm2.py              # Single-threaded UTF-8 pull generator & branch context manager
├── lib/
│   ├── liblfm.a             # Static engine library archive
│   ├── liblfm.so            # Shared dynamic library (-fPIC)
│   └── liblfm2.so           # Public C11 ABI dynamic library (zero internal symbol leakage)
├── tests/                   # 27 automated unit and parity test binaries
│   ├── test_arena.c         # 64-byte alignment & arena reset/OOM tests
│   ├── test_math.c          # RMSNorm, SiLU, RoPE & FP32 MatVec tests
│   ├── test_simd_avx2.c     # AVX2 vector dot product & RMSNorm unit tests
│   ├── test_quants.c        # Q8_0 & Q4_0 block quantization dot product tests
│   ├── test_simd_kquants.c  # AVX2 Q8_K pre-quantization & K-quants dot products
│   ├── test_fused_conv.c    # Fused LIV ShortConv numerical equivalence test
│   ├── test_threads.c       # ThreadPool generation-barrier concurrency stress test
│   ├── test_tokenizer.c     # BPE encoding, decoding & byte fallback tests
│   ├── test_bpe.c           # Priority-ordered GGUF BPE rank merging unit tests
│   ├── test_gguf.c          # GGUF parser & tensor offset alignment test
│   ├── test_conv1d.c        # LIV ShortConv numerical parity vs PyTorch
│   ├── test_attention.c     # GQA + QK-Norm + RoPE parity vs PyTorch
│   ├── test_forward.c       # End-to-end toy LFM2 model forward pass test
│   ├── test_c_api.c         # C ABI lifecycle, buffer generation & callback test
│   ├── test_lfm2_threadpool.c # Spin-barrier thread pool & SMT pinning tests
│   ├── test_optimizations.c # Fused ShortConv & SwiGLU optimization parity tests
│   ├── test_prefill_batch.c # Batched GEMM prefill acceleration & parity tests
│   ├── test_decode_1x4.c    # 1x4 row unrolling & fused SwiGLU FFN tests
│   ├── test_kernels_fast.c  # Planar ShortConv, RoPE LUT & online softmax tests
│   ├── test_cpuid_quants.c  # CPUID dispatch & Q4_0/Q5_K/IQ4_XS tests
│   ├── test_runtime_features.c # PDA grammar masking, think budget & KV shift tests
│   ├── test_moe_checkpoint.c# Sparse MoE routing, 3D tensor slicing & rollback tests
│   ├── test_cli_stream.c    # Streaming UTF-8 assembly, stop holdback & SIGINT tests
│   ├── test_sys_cache.c     # OOM guard, HugePage arena, warmup & prompt cache tests
│   ├── test_lfm2_api.c      # Public C11 ABI unit tests (streaming, rollback, embeddings)
│   ├── test_lfm2_server.c   # Non-blocking epoll OpenAI HTTP/SSE server automated test
│   ├── test_hardening.c     # FTZ/DAZ mode, non-256 divisible dimension & bounds hardening tests
│   ├── test_python_bindings.py # Python ctypes bindings automated test
│   ├── test_python_lfm2.py  # Modern Python SDK test (generator streaming & branch rollback)
│   ├── verify_parity.py     # 5-tier numerical parity SLA & golden logit verifier
│   └── test_http_server.sh  # HTTP REST server & Web Studio integration test
├── tools/                   # 14 Standalone zero-dependency C11 tools & benchmarks
│   ├── lfm2_loader.c        # Zero-copy GGUF v3 inspector for LFM2 hybrid models
│   ├── lfm2_bpe.c           # Priority-ordered BPE rank merging engine
│   ├── lfm2_threadpool.c    # Standalone atomic spin-barrier thread pool
│   ├── lfm2_forward.c       # Standalone C11 forward pass engine for LFM2 models
│   ├── lfm2_chat.c          # Interactive terminal chat REPL with Min-P sampling
│   ├── lfm2_prefill_batch.c # Batched GEMM prefill acceleration benchmark
│   ├── lfm2_decode_1x4.c    # 1x4 row unrolling & fused SwiGLU FFN benchmark
│   ├── lfm2_kernels_fast.c  # Planar ShortConv, RoPE LUT & online softmax benchmark
│   ├── lfm2_cpuid_quants.c  # CPUID multi-ISA dispatch & Q4_0/Q5_K/IQ4_XS benchmark
│   ├── lfm2_runtime_features.c # Grammar PDA, think budget & KV sink shifting benchmark
│   ├── lfm2_moe_checkpoint.c# Sparse MoE routing, 3D slicing & rollback benchmark
│   ├── lfm2_cli_stream.c    # Streaming UTF-8 boundary buffer & two-stage SIGINT
│   ├── lfm2_sys_cache.c     # OOM guard, sequential page warmup & prompt cache engine
│   ├── lfm2_server.c        # Non-blocking Linux epoll OpenAI HTTP/SSE server
│   ├── generate_reference_tensors.py # PyTorch golden tensor generator
│   └── create_toy_gguf.py   # Synthetic LFM2 GGUF model builder
├── lfm2_server.c            # Root forwarding stub for tools/lfm2_server.c
└── Makefile                 # Master build system
```

---

## 🏛️ Library vs Standalone Tools: The Boundary

LAII is organized into two distinct subsystems:

### Subsystem A: Core Engine Library (`src/`, `include/`)
* **Role:** Powers `liblfm.so`, `liblfm.a`, `lfm-run`, `lfm-server`, and the Python bindings.
* **Header Convention:** Includes headers from `include/` (e.g. `#include "gguf.h"`).
* **Tensor Dimension Accessor:** Tensors have shape `dims[4]` (e.g., `tensor->dims[0]`).

### Subsystem B: Standalone Tools & Benchmarks (`tools/`)
* **Role:** Self-contained, single-file C11 utilities that can be compiled independently without the rest of the engine.
* **Header Convention:** Implements internal GGUF parsing directly within `lfm2_loader.c`.
* **Tensor Dimension Accessor:** Tensors have shape `ne[4]` (e.g., `tensor->ne[0]`).

### Conditional Compilation Guard:
When sharing code between `src/` and `tools/` (such as [`tools/lfm2_sys_cache.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_sys_cache.c)):
```c
#if defined(LFM_GGUF_H) || defined(LFM_MODEL_H)
    /* Included from src/ or include/: Use core types */
    uint64_t n_cols = tensor->dims[0];
#else
    /* Compiled standalone or from tools/: Use tools/ types */
    uint64_t n_cols = tensor->ne[0];
#endif
```
This preserves single-file portability of tools while enabling deep integration into the core engine without type redeclaration conflicts.

