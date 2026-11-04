# Phase 3 Technical Specification: Single-Binary Distribution, Embedded Web Studio & C ABI

This specification documents the engineering design, architecture, and verification of **Phase 3** of the specialized Liquid AI (LFM2) Linux inference engine.

---

## 1. Executive Summary & Objectives

Phase 1 established the mathematically verified reference forward pass, and Phase 2 achieved high-throughput AVX2 SIMD acceleration, quantization (`Q8_0`, `Q4_0`), fused LIV kernels, and multi-threaded scaling.

Phase 3 transitions the engine into a **production-ready, zero-dependency deployment artifact**:
1. **Public C ABI & Shared/Static Libraries (`liblfm.so`, `liblfm.a`):** Clean header [`include/lfm_c_api.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm_c_api.h) exposing an opaque engine lifecycle, streaming generation, tokenization, and model metadata queries.
2. **Zero-Dependency Python FFI Bindings (`bindings/python/laii.py`):** Idiomatic Python wrapper via standard library `ctypes` without requiring PyTorch, CUDA, or C++ build tools at runtime.
3. **Embedded HTTP REST Server & Web Studio (`src/lfm_server.c`, `src/lfm_webui.h`):** Single-binary HTTP server implementing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/health`) and serving a self-contained, glassmorphic dark-mode Web Studio directly from binary memory.
4. **Standalone Static ELF Binaries (`bin/lfm-run-static`, `bin/lfm-server-static`):** Fully statically linked executables with zero external dynamic dependencies (`statically linked`), runnable offline on any Linux x86_64 system.

---

## 2. Public C ABI & Architecture

The public interface isolates external consumers from internal memory arena structures and SIMD intrinsics:

```c
// include/lfm_c_api.h
typedef struct LFMEngine lfm_engine_t;

typedef struct {
    float temperature;
    uint32_t top_k;
    uint32_t max_new_tokens;
    uint32_t seed;
    bool echo_prompt;
} lfm_sampler_params_t;

typedef void (*lfm_token_callback_t)(uint32_t token_id, const char *piece, void *user_data);

LFM_API lfm_engine_t* lfm_create(const char *model_path, int num_threads);
LFM_API void lfm_destroy(lfm_engine_t *engine);
LFM_API int lfm_generate(lfm_engine_t *engine, const char *prompt, char *out_buf, size_t out_buf_size, const lfm_sampler_params_t *params);
LFM_API int lfm_generate_stream(lfm_engine_t *engine, const char *prompt, const lfm_sampler_params_t *params, lfm_token_callback_t callback, void *user_data);
LFM_API int lfm_tokenize(lfm_engine_t *engine, const char *text, int32_t *tokens_out, int max_tokens);
LFM_API int lfm_detokenize(lfm_engine_t *engine, int32_t token, char *buf_out, size_t buf_size);
```

### Artifact Targets
* **Shared Object (`lib/liblfm.so`):** Position-independent shared library compiled with `-fPIC -shared` for dynamic linking and language FFI.
* **Static Archive (`lib/liblfm.a`):** Standalone archive for embedding directly into host C/C++ applications.

---

## 3. Python FFI Bindings (`bindings/python/laii.py`)

The Python package `laii` provides a zero-dependency integration layer:

```python
from laii import LFMEngine

# Initialize engine with 4 worker threads
with LFMEngine("models/lfm2-1.2b.gguf", num_threads=4) as engine:
    print(f"Loaded {engine.architecture}: {engine.num_layers} layers, {engine.hidden_dim} dim")

    # Synchronous generation
    response = engine.generate("Explain Liquid Foundation Models", max_tokens=128, temperature=0.7)
    print(response)

    # Real-time token streaming
    for piece in engine.generate_stream("Why LIV convolutions on CPU?", max_tokens=64):
        print(piece, end="", flush=True)
```

### Key Python Features
* Automatic discovery of `liblfm.so` across library paths and `LIBLFM_PATH`.
* Full context manager support (`with LFMEngine(...) as engine:`).
* Fast C callback bridging for Python generator-based streaming (`generate_stream`).
* Direct tokenization and detokenization APIs.

---

## 4. Embedded HTTP Server & Web Studio

### 4.1 Server Architecture (`src/lfm_server.c`)
* **Zero External Dependencies:** Built purely on POSIX sockets (`socket`, `bind`, `listen`, `accept`, `write`).
* **Non-Blocking Signal Handling:** Intercepts `SIGINT` and `SIGTERM` gracefully using socket timeouts (`SO_RCVTIMEO`) and cleans up resources without memory leaks.
* **CORS Support:** Automatically emits `Access-Control-Allow-Origin: *` headers for browser cross-origin requests.

### 4.2 OpenAI-Compatible REST Endpoints

| Endpoint | Method | Payload | Response |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Embedded Single-Page Web Studio (`text/html`) |
| `/health` | `GET` | None | `{"status":"ok","model":"lfm2","num_layers":2,...}` |
| `/v1/models` | `GET` | None | Model architecture, layer count, and dimensions |
| `/v1/chat/completions` | `POST` | `{"messages":[{"role":"user","content":"..."}]}` | Standard OpenAI Chat Completion JSON |
| `/v1/completions` | `POST` | `{"prompt":"..."}` | Standard text completion JSON |

### 4.3 Embedded Web Studio (`src/lfm_webui.h`)
The complete HTML, CSS, and JavaScript interface is compiled into read-only binary memory as `LFM_EMBEDDED_INDEX_HTML`:
* **Aesthetics:** Futuristic glassmorphism theme (`#07090e` base, `#00f2fe` neon cyan accents, translucent frosted glass cards).
* **Real-Time Performance:** Displays active token speed (`tokens/sec`) and generation latency counters.
* **Interactive Controls:** Sliders for Temperature ($0.0 - 1.5$), Top-K ($1 - 100$), and Max Tokens ($16 - 256$).
* **Prompt Chips:** Quick-launch questions explaining Liquid AI architecture, ring buffer states, and KV cache reduction.

---

## 5. Standalone Static ELF Packaging

Target `make static` builds fully self-contained ELF executables:
* `bin/lfm-run-static` (CLI runner)
* `bin/lfm-server-static` (Dedicated web & REST server)

```bash
file bin/lfm-run-static
# Output: bin/lfm-run-static: ELF 64-bit LSB executable, x86-64, version 1 (GNU/Linux), statically linked
```

These executables contain the complete engine, tokenizer, mathematical kernels, and embedded web studio with **zero dynamic library dependencies**. They run immediately on any x86_64 Linux machine (Alpine, Ubuntu, RHEL, Arch, or bare-metal containers) without requiring glibc or external runtimes.

---

## 6. Comprehensive 15-Part Automated Test Suite

Phase 3 expands the automated test framework to **15 comprehensive test suites**:

```text
=======================================================
Running Liquid AI Engine Full Automated Test Suite     
=======================================================
[Test] Memory Arena allocator tests                     -> PASSED
[Test] Mathematical Primitives unit tests               -> PASSED
[Test] AVX2 SIMD Vectorization unit tests               -> PASSED
[Test] Quantization (Q8_0 & Q4_0) dot product tests     -> PASSED
[Test] Fused LIV ShortConv kernel numerical equivalence -> PASSED
[Test] Multi-threaded ThreadPool barrier & concurrency  -> PASSED
[Test] GGUF binary parser and mmap loader test          -> PASSED
[Test] BPE Tokenizer unit tests                         -> PASSED
[Test] LIV Gated ShortConv numerical parity (PyTorch)   -> PASSED
[Test] Grouped Query Attention + RoPE parity (PyTorch)  -> PASSED
[Test] End-to-end forward pass on toy LFM2 model        -> PASSED
[Test] C ABI lifecycle, metadata & tokenizer tests      -> PASSED
[Test] Python ctypes bindings unit test                 -> PASSED
[Test] Embedded HTTP REST server & Web Studio tests     -> PASSED
[Test] Standalone Static ELF packaging verification     -> PASSED
=======================================================
ALL 15/15 TESTS PASSED WITH 100% SUCCESS RATE!         
=======================================================
```

Memory sanitization (`make test-asan`) passes cleanly under AddressSanitizer and UndefinedBehaviorSanitizer with zero memory leaks across all C test suites and HTTP server requests.
