# Extension Playbook: Step-by-Step Developer Recipes

This document provides concrete, actionable recipes for implementing new features, layers, quantization formats, and tests in LAII.

---

## 🍳 Recipe A: Adding a New Quantization Format (e.g., `Q2_K` or `IQ3_S`)

1. **Define the Block Struct:**
   In [`include/lfm_simd.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm_simd.h) and [`tools/lfm2_loader.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_loader.c), define the memory layout of the block matching the GGUF specification.
   ```c
   typedef struct {
       // Exact GGUF memory-packed layout
   } block_qX_t;
   ```
2. **Implement Vector Dot Product Kernel:**
   In [`src/lfm_simd.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm_simd.c) or [`tools/lfm2_cpuid_quants.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_cpuid_quants.c), write both the scalar reference and the AVX2 vectorized implementation:
   ```c
   __attribute__((target("avx2,fma,f16c")))
   float vec_dot_qX_q8_K_avx2(const void *vx, const void *vy, int n) { ... }
   ```
3. **Register Dynamic Dispatch in CPUID System:**
   In [`tools/lfm2_cpuid_quants.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_cpuid_quants.c), bind the function pointer inside `lfm2_init_cpu_kernels()`.
4. **Update MatMul Dispatches:**
   In [`src/lfm_forward.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm_forward.c) and [`tools/lfm2_forward.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_forward.c), add `case GGML_TYPE_QX:` to the matrix-vector dispatch switch.
5. **Add Automated Test:**
   Add a parity check in [`tests/test_cpuid_quants.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_cpuid_quants.c) comparing AVX2 vs scalar reference across multiple boundary column counts.
6. **Verify:**
   Run `make test-c-bins` and `make test-asan`.

---

## 🍳 Recipe B: Adding a New Architectural Feature or Layer

1. **Identify State Nature:**
   Determine whether the new layer has **$O(1)$ recurrent state** (like ShortConv) or **$O(T)$ position-indexed state** (like Attention).
   * If recurrent: It must be preallocated in fixed slabs, snapshotted in `LFM2Checkpoint`, and factored into the prompt cache full-prefix invariant.
   * If attention-like: It must be stored in the linear KV cache and accounted for in the context memory planner.
2. **Add Configuration Keys:**
   Update `LFM2Config` and GGUF metadata parsing in [`src/lfm_model.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm_model.c) and [`tools/lfm2_loader.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_loader.c).
3. **Bind Layer Tensors:**
   Extract and bind GGUF tensor pointers in `lfm_load_weights()` and `lfm2_bind_model_weights()`.
4. **Implement Forward Pass:**
   Add layer execution inside the layer loop of [`src/lfm_forward.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm_forward.c) and [`tools/lfm2_forward.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_forward.c).
5. **Add Unit Parity Test:**
   Create a dedicated test in `tests/test_<feature>.c` comparing execution against PyTorch or golden scalar math.

---

## 🍳 Recipe C: Adding a New CLI Flag to `lfm-run` & `lfm2_chat`

1. **Update Help & Arg Parsing:**
   In [`src/main.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/main.c) (and [`tools/lfm2_chat.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_chat.c)):
   * Add the option to the `print_usage()` function.
   * Add the short/long flag handling in `parse_args()`.
2. **Propagate Config into Sampler / Engine:**
   * If the option impacts generation or sampling, add the field to [`LFMSamplerConfig`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm_engine.h#L39).
   * Propagate the field into [`src/lfm_engine.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm_engine.c).
3. **Update Documentation:**
   * Document the flag in [`README.md`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/README.md) under CLI options.

---

## 🍳 Recipe D: Adding a New Test Target to `Makefile`

When creating a new unit test `tests/test_<name>.c`:

1. **Add Target to `TEST_BINS`:**
   In [`Makefile`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/Makefile):
   ```makefile
   TEST_BINS = \
       ... \
       bin/test_<name>
   ```
2. **Add Compilation Rule:**
   ```makefile
   bin/test_<name>: tests/test_<name>.c $(OBJS) | dirs
   	$(CC) $(CFLAGS) -I. $< $(OBJS) -o $@ $(LDFLAGS)
   ```
   *(Or if it is a standalone test using files from `tools/`, compile without `$(OBJS)` and pass `-I.`)*
3. **Add Execution Line to `test-c-bins`:**
   ```makefile
   test-c-bins: $(TEST_BINS) fixtures
   	...
   	@./bin/test_<name>
   	@echo "-------------------------------------------------------"
   ```
4. **Update Static Test Rule (if applicable):**
   If creating a standalone static binary, add `bin/lfm2-<name>-static` to `static` and verify it in `static-test`.

---

## 🍳 Recipe E: Exposing a New Feature via the Public C11 ABI & Python SDK

1. **Declare Signature in Public Header:**
   In [`include/lfm2.h`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/include/lfm2.h), declare the new API function using `LFM2_API` and fixed-width types:
   ```c
   LFM2_API lfm2_status_t lfm2_my_feature(lfm2_context_t *ctx, int32_t param, ...);
   ```
   *Never leak internal structs (`GGUFTensor`, `LFM2ThreadPool`) into this header.*
2. **Implement in Public ABI Source:**
   In [`src/lfm2_api.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/src/lfm2_api.c), implement the entry point. The file compiles with `-fvisibility=hidden`, so only functions decorated with `LFM2_API` are exported into `lib/liblfm2.so`.
3. **Bind in Python SDK:**
   In [`python/lfm2.py`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/python/lfm2.py), declare `argtypes` and `restype` on `_lib.lfm2_my_feature`, and expose an idiomatic Python method on `LFMContext` or `LFMModel`.
4. **Add Verification to Test Suites:**
   Add test assertions to both [`tests/test_lfm2_api.c`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_lfm2_api.c) and [`tests/test_python_lfm2.py`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tests/test_python_lfm2.py).
5. **Verify Full Pipeline:**
   Run `make test` and `make test-asan`.

---

## 🍳 Recipe F: Running and Testing the Non-Blocking Linux `epoll` OpenAI Server

1. **Compile the Server:**
   ```bash
   # Production dynamic binary linking against liblfm2.so
   make bin/lfm2_server

   # Standalone statically linked demo (zero model weights needed)
   make bin/lfm2-server-static
   ```
2. **Launch Server:**
   ```bash
   # Standalone mock demo on port 8080:
   ./bin/lfm2-server-static 8080

   # Or production model server:
   ./bin/lfm2_server models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf 8080 127.0.0.1 4 2048
   ```
3. **Verify Endpoints with `curl`:**
   ```bash
   # Health check:
   curl -s http://127.0.0.1:8080/health

   # Embedded Web Studio:
   curl -s http://127.0.0.1:8080/ | head -n 10

   # Streaming SSE chat completion:
   curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Stream test"}],"stream":true}'

   # Text embeddings:
   curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"Semantic text embedding","encoding_pooling":"mean"}'
   ```
4. **Automated Server Test Suite:**
   Run `./bin/test_lfm2_server` to verify cooperative token stepping, CORS preflight, and instant client disconnect abort (`EPOLLRDHUP`).

---

## 🍳 Recipe G: Running Multi-Tier Numerical Parity & Golden Oracle Verification

To verify that C11 SIMD/quantized kernels maintain exact distribution parity against FP32 references without needing PyTorch on edge devices:

1. **Run 5-Tier Pure-Python Contract Suite:**
   ```bash
   make test-parity
   ```
2. **Generate Compact Binary Golden Fixture (`.lfm_golden`) from Reference GGUF:**
   ```bash
   python3 tests/verify_parity.py -m models/LFM2.5-1.2B-Instruct-Q8_0.gguf --golden-file tests/lfm2_1.2b.lfm_golden --generate-golden
   ```
3. **Verify Quantized Checkpoint Against SLA Tier:**
   ```bash
   python3 tests/verify_parity.py -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf -q Q4_K_M --golden-file tests/lfm2_1.2b.lfm_golden
   ```
   Asserts Cosine Similarity $\ge 0.99300$, KL Divergence $\le 0.0650$, Top-5 Overlap $\ge 4/5$, and bit-exact state checkpoint rollback ($0.0$ ULP).

---

## 📋 The Agent Pre-Submission Checklist

Before presenting your work as complete to the user, execute this sequence:

- [ ] 1. **Clean build check:** `make clean && make all` passes with 0 warnings.
- [ ] 2. **Unit tests check:** `make test-c-bins` passes 100%.
- [ ] 3. **Full test suite check:** `make test` (Python FFI, HTTP server, static binaries) passes 100%.
- [ ] 4. **Memory safety check:** `make test-asan` passes with **zero** AddressSanitizer or UBSan leaks, errors, or alignment complaints.
- [ ] 5. **Documentation updated:** Any new CLI flags, options, or test targets are reflected in `README.md`, `docs/TEST_SUITE.md`, and `docs/ROADMAP.md`.

