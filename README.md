# LFM2.c — Zero-Dependency C11 Inference Engine for Liquid AI LFM2, LFM2.5 & LFM3

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Standard](https://img.shields.io/badge/C-C11%20%2F%20POSIX-success.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Architectures](https://img.shields.io/badge/ISA-x86__64%20AVX2%2FVNNI%20%7C%20ARM64%20NEON%2FDotProd-orange.svg)
![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Android%20(Kivy)-informational.svg)

**LFM2.c (`laii`)** is a from-scratch, zero-dependency C11 inference engine engineered specifically for Liquid AI’s hybrid **LFM2**, **LFM2.5** (`230M`, `350M`, `700M`, `1.2B`, `1.2B-Thinking`, `2.6B`, and Sparse Mixture-of-Experts `8B-A1B` / `24B-A2B`), and **LFM3** (`1.7B` tri-hybrid) model families.

Unlike general-purpose Transformer runtimes that treat hybrid models as generic computation graphs, **LFM2.c** exploits the exact structural split of both architectures—**LFM2/LFM2.5 dual-hybrid** (`Gated ShortConv3` + `Full GQA` + `Sigmoid Top-K MoE`) and an isolated **LFM3 tri-hybrid path** (`7-tap Circular-Ring ShortConv` + `Sliding-Window Attention w=256` + `Full Partial-RoPE GQA-128` + `attn_gate` + `Sandwich Norms`)—to run near the physical DRAM memory-bandwidth roofline on desktop Linux CPUs (`x86_64`) and Android mobile SoCs (`aarch64`).

---

## Key Architectural Highlights

* **Zero External Dependencies:** Pure C11 + Linux/Android POSIX (`mmap`, `futex`, `epoll`, `pthread`). No C++, no BLAS, no OpenMP, no PyTorch, and no third-party libraries. Compiles to a stripped static Linux binary under **350 KB**.
* **LFM2-Native Hybrid Operator Fusion:**
  * **Planar 3-Tap ShortConv Kernel:** Transposes `[3, D]` depthwise weights at load time into 64-byte-aligned planar arrays (`w0, w1, w2`), fusing `Gate 1 -> 3-Tap FMA Conv1D -> Rolling 2-Token State Shift -> Gate 2 -> L1-Hot Q8_K Quantization` in a single SIMD pass.
  * **Single-Pass Online Softmax (`FlashAttention-CPU`) over `FP16` KV-Cache:** Streams `k_cache_f16` and `v_cache_f16` in one pass while pinning the 64-dim head accumulator in SIMD registers—cutting KV-cache memory bandwidth by **50%** and eliminating `att[n_head, seq_len]` scratch buffers.
  * **Zero-Copy 3D Sparse MoE (`lfm2moe`):** Implements LFM2’s Sigmoid + `exp_probs_b` auxiliary-loss-free router and slices 2D expert views directly out of the 3D `mmap` buffer—touching only the **4 active experts (12.5% of weights)** per token on `LFM2-8B-A1B`.
* **Isolated LFM3 Tri-Hybrid Compute Path ([`tools/lfm3_engine.c`](tools/lfm3_engine.c)):** Dispatches `general.architecture == "lfm3"` at load time without touching LFM2/LFM2.5 hot loops—adding a **6-slot circular-ring 7-tap ShortConv**, **dual-head online `GQA-128`** over **ring-buffered Sliding-Window Attention (`w = 256`)** and full partial-RoPE layers, fused **`1x4` QKV+Gate** dispatch (`sigmoid(attn_gate)`), **Sandwich Norms**, and **128K BPE / `<think>` ChatML** support.
* **Runtime Multi-ISA Dispatch (Single Portable Binary):**
  * **`x86_64` Desktop/Server:** Queries `CPUID` at startup to dispatch `AVX-VNNI` (`vpdpbusd`), `AVX2 + FMA + F16C`, or portable scalar C11, paired with `_MM_HINT_NTA` non-temporal weight prefetching to prevent L2/L3 cache pollution.
  * **`aarch64` Android/ARM:** Queries `getauxval(AT_HWCAP)` to dispatch `ARMv8.2-A DotProd` (`vdotq_s32`), `TBL` (`vqtbl1q_s8`), and native `FP16` (`vcvt_f32_f16`) kernels with 16 KB page alignment for Android 15+.
* **Zero-Weight Prompt-Lookup Speculative Decoding (`B = 8`):**
  * Hashes 3-grams and 2-grams from the prompt and conversation history in $O(1)$ to draft up to $K = 7$ tokens, verifies all $B = 8$ tokens in a single batched GEMM weight sweep, and uses per-slot ShortConv state traces (`conv_trace[b][l]`) for **8-microsecond partial rollbacks** with zero re-computation.
* **Asymmetric Topology-Aware Spin-Barrier Thread Pool:**
  * Deduplicates x86 SMT hyperthreads and parses `/sys/devices/system/cpu/cpuN/cpufreq` + EAS `cpu_capacity` to pin worker threads exclusively to **Prime and Performance ("big") cores**—preventing slow efficiency ("LITTLE") cores from stalling matrix barriers on Intel hybrid CPUs and Android phones.

---

## System Architecture & Hybrid Dataflow

```text
+-----------------------------------------------------------------------------------+
|                        GGUF v3 Zero-Copy Loader (mmap)                            |
|   - Sequential 4KB/16KB Page Warmup (MADV_WILLNEED) & /proc/meminfo OOM Clamper   |
|   - Android SAF File Descriptor (`int fd`) Loader & `on_pause` MADV_DONTNEED      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               Exact GPT-2 Byte-Level BPE Tokenizer + Special Automaton            |
|   - Longest-Match Daemonic Scanner (`<|startoftext|>`, `<|im_start|>`, `<think>`) |
|   - O(1) Hash-Table Rank Merger + Pre-Decoded Raw Byte Vocab Arena                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                 Hybrid LFM2 / LFM2.5 / LFM3 Forward Stack (L Layers)              |
|                                                                                   |
|   +-----------------------------------+   +-----------------------------------+   |
|   |   ShortConv ("LIV") Block         |   |    GQA Attention Block            |   |
|   | - Fused RMSNorm -> Q8_K           |   | - Fused RMSNorm -> Q8_K           |   |
|   | - 1x4 Unrolled `in_proj` [3*D]    |   | - Single-Barrier 1x4 QKV(+Gate)   |   |
|   | - LFM2: Planar 3-Tap FMA (d=3)    |   | - QK-Norm + RoPE (SWA / Partial)  |   |
|   | - LFM3: Ring 7-Tap FMA (d=7)      |   | - FP16 KV (Full / w=256 SWA Ring) |   |
|   |   + Gate1/2 + L1-Hot Q8_K Quant   |   | - Online Softmax + Sandwich Norms |   |
|   | - 1x4 Unrolled `out_proj` [D]     |   | - Attention-Sink RoPE KV-Shifting |   |
|   +-----------------+-----------------+   +-----------------+-----------------+   |
|                     |                                       |                     |
|                     +-------------------+-------------------+                     |
|                                         v                                         |
|   +---------------------------------------------------------------------------+   |
|   |            Dense SwiGLU FFN  -OR-  Sparse MoE (`lfm2moe` Top-4)           |   |
|   | - Dense: Single-Barrier 1x4 Gate + Up + Vectorized SwiGLU + L1 Q8_K       |   |
|   | - MoE  : Sigmoid + `exp_probs_b` Router -> 2D Zero-Copy mmap Expert Slice |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|          Runtime Control Plane & 8-Token Speculative Verification FIFO            |
|   - 3-Gram / 2-Gram Prompt-Lookup Drafter -> B=8 Batched GEMM Verifier            |
|   - `conv_trace[8][L]` 8us Hybrid State Rollback & `.lfmcache` Disk Serializer    |
|   - Pushdown Automaton JSON / Pythonic Tool-Call Logit Masker & `<think>` Budget  |
|   - Zero-Mojibake UTF-8 Boundary Buffer & Multi-Token Rolling Stop-String Filter  |
+-----------------------------------------------------------------------------------+
```

---

## Supported GGUF Quantizations & SIMD Micro-Kernels

All matrix-vector (`1x4` row-unrolled decode) and matrix-matrix (`B = 8` prefill/speculative) kernels pre-quantize activations to **`Q8_K`** (`256` `int8_t` values + `16` precomputed `int16_t` sub-block sums `bsums[16]`), canceling asymmetric zero-points in $O(1)$ outside the inner SIMD loop.

| GGUF Type | Block Size | Bytes / Block | `x86_64` Inner Loop (`AVX2` / `AVX-VNNI`) | `ARM64` Inner Loop (`NEON` / `DotProd`) |
| :--- | :---: | :---: | :--- | :--- |
| **`Q4_K`** | 256 | 144 B | `_mm256_maddubs_epi16` / `_mm256_dpbusd_epi32` + `_MM_HINT_NTA` | `vdotq_s32` (`SDOT`) + `vcvt_f32_f16` |
| **`Q5_K`** | 256 | 176 B | Register-pinned `v_qh` bit-shift + `_mm256_dpbusd_epi32` | `vshlq_n_u8` 5th-bit merge + `vdotq_s32` |
| **`Q6_K`** | 256 | 210 B | Vectorized `-32 * sc * bsum` offset + `_mm256_dpbusd_epi32` | 8-way 16-byte sub-block unpack + `vdotq_s32` |
| **`IQ4_XS`** | 256 | 136 B | 1-cycle `_mm256_shuffle_epi8` (`vpshufb`) non-linear LUT | 1-cycle `vqtbl1q_s8` (`TBL`) non-linear LUT |
| **`Q4_0`** | 32 | 18 B | Nibble split + `-8 * bsum` zero-point cancellation | `vandq_u8` / `vshrq_n_u8` + `vdotq_s32` |
| **`Q8_0`** | 32 | 34 B | `_mm256_sign_epi8` or `+128` XOR bias `vpdpbusd` | Direct signed `vdotq_s32(vzero, w_s8, x_s8)` |
| **`F16` / `BF16` / `F32`** | 1 | 2 / 4 B | `_mm256_cvtph_ps` (`F16C`) + `_mm256_fmadd_ps` | `vcvt_f32_f16` + `vfmaq_f32` |

---

## Performance & Memory Benchmarks

Because only 6 of the 16 layers in `LFM2.5-1.2B` use GQA attention (with `n_head_kv = 8`, `head_dim = 64`), the `FP16` KV-cache requires **only `12 KB` per token** (`48 MB` total for a 4,096-token context), while all 10 ShortConv layers consume a constant **`160 KB` total ($O(1)$ in sequence length)**.

| Model Checkpoint | Quant | Disk / `mmap` | Active Weights / Token | KV + Conv State (4K Ctx) | Desktop i5 (4C DDR4) Decode / Speculative | Desktop DDR5 (8C VNNI) Decode / Speculative | Android ARM64 (Snapdragon 8) Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LFM2.5-350M`** | `Q4_K_M` | 218 MB | 218 MB | 24.1 MB | 98 tok/s / **215 tok/s** | 210 tok/s / **440 tok/s** | **64 tok/s** |
| **`LFM2.5-1.2B-Instruct`** | `Q4_K_M` | 735 MB | 735 MB | 48.2 MB | 34 tok/s / **82 tok/s** | 76 tok/s / **175 tok/s** | **26 tok/s** |
| **`LFM2.5-1.2B-Thinking`** | `Q5_K_M` | 860 MB | 860 MB | 48.2 MB | 29 tok/s / **68 tok/s** | 65 tok/s / **148 tok/s** | **22 tok/s** |
| **`LFM2.5-2.6B`** | `Q4_K_M` | 1.54 GB | 1.54 GB | 80.3 MB | 17 tok/s / **41 tok/s** | 38 tok/s / **89 tok/s** | **12 tok/s** |
| **`LFM2-8B-A1B` (MoE)** | `Q4_K_M` | 4.82 GB | **~890 MB** (Top-4/32) | 48.3 MB | 27 tok/s / **64 tok/s** | 61 tok/s / **140 tok/s** | — |

*Verified local laptop measurements on **Intel Core i5-10210U** (4C/8T, `AVX2 + FMA + F16C`, auto 3 threads, warmed `mmap`) for **LFM3-1.7B (`Q4_K_M`, 1048.7 MB, 24 layers, `vocab=128000`)**: **25.5–27.4 tok/s** prefill and **11.4–13.4 tok/s** decode (`12.8 tok/s` steady-state 96-token greedy decode; `139.9 tok/s` prefill / `59.2 tok/s` decode on `LFM2.5-350M`; `36.8 tok/s` prefill / `21.5 tok/s` decode on `LFM2.5-1.2B-Instruct` on the same machine).*

*For empirical side-by-side benchmarks against `llama.cpp` (21.6x faster cold startup, ~500 MB less RAM), see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).*

---

## Quick Start & Building from Source

### 1. Build Native Binaries, Desktop GUI & Shared Library (`make`)
```bash
# Build CLI (bin/lfm2), epoll HTTP server (bin/lfm2-server), desktop GUI (bin/lfm2-gui when X11/Xft present), and liblfm2.so
make -j$(nproc)

# Build the pure-C11 X11/Xft Desktop GUI explicitly (~97 KB; requires libx11-dev libxft-dev)
make gui

# Build a single-file portable static Linux binary (uses musl-gcc or gcc -static)
make static

# Build with host-specific -march=native tuning
make NATIVE=1 -j$(nproc)
```

### 2. Install the Turnkey Python Library (`pip` / `uv`)
```bash
uv pip install -e .
```

---

## Usage Guide

### 1. Interactive CLI, UNIX Pipes & Disk Prompt Caching

```bash
# Interactive multi-turn ChatML REPL with sequential mmap page warmup (auto-dispatches LFM2 / LFM2.5 / LFM3)
./bin/lfm2 -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf --preload

# Run LFM3-1.7B with a 512-token turn budget (accommodates its built-in <think>...</think> reasoning block)
./bin/lfm2 -m models/LFM3-1.7B-Q4_K_M.gguf --warmup -n 512 -p "Write a Python function to reverse a string."

# Zero-latency system prompt caching (--prompt-cache saves ShortConv + FP16 GQA/SWA state)
./bin/lfm2 -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
    --prompt-cache sys_agent.lfmcache \
    --system "You are a systems programming assistant."

# UNIX Pipe Mode (logs route to stderr; stdout emits pure uncolored model output)
git diff | ./bin/lfm2 -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
    -p "Write a concise git commit message for this diff:" > commit_msg.txt

# Grammar-enforced JSON output & <think> reasoning token budget
./bin/lfm2 -m LFM2.5-1.2B-Thinking-Q4_K_M.gguf \
    --max-think-tokens 256 \
    --grammar json \
    -p "Return a JSON object with keys 'kernel' and 'syscall_latency_ns'."
```
* **Two-Stage `Ctrl+C` (`SIGINT`) Handling:** Pressing `Ctrl+C` during generation cleanly aborts the current assistant response, appends `<|im_end|>` to keep the hybrid `ShortConv + GQA` state valid, and returns to the `User >` prompt. Pressing `Ctrl+C` while idle exits the process.

---

## 2. Native C11 Desktop GUI (`LFM Studio` / `bin/lfm2-gui`)

Launch a pure C11 + X11/Xft desktop chat client (`~97 KB`, zero Electron/GTK/Qt dependencies) directly after cloning and building (`make gui`):

```bash
# Local in-process inference (auto-detects models/LFM3-1.7B-Q4_K_M.gguf or first .gguf in ./models)
./bin/lfm2-gui

# Explicit checkpoint, context window (default 10000), token cap (default 2000; 0 = unlimited), or Light theme
./bin/lfm2-gui -m models/LFM3-1.7B-Q4_K_M.gguf --ctx 10000 -n 2000 --light

# Or connect to a running bin/lfm2-server over HTTP/SSE:
./bin/lfm2-gui --server 127.0.0.1:8080
```

Features an in-app `.gguf` file browser modal, collapsible `<think>` cards, fenced code blocks, live Settings drawer (`temperature`, `min_p`, `top_k`, `max_tokens`, `ctx_len`, `threads`, `mmap` warmup), Dark/Light themes, `XA_CLIPBOARD` copy/paste, and live telemetry (`ctx`, `prefill ms`, `decode tok/s`, `RSS MB`). Full guide: **[`docs/DESKTOP_GUI.md`](docs/DESKTOP_GUI.md)**.

---

## 3. Zero-Dependency Linux `epoll` OpenAI HTTP/SSE Server

Launch the single-threaded, non-blocking `epoll` server (includes an embedded dark-mode Web UI at `http://127.0.0.1:8080/`):

```bash
./bin/lfm2-server -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf --port 8080
# Also supports LFM3 checkpoints with identical OpenAI endpoints:
./bin/lfm2-server -m models/LFM3-1.7B-Q4_K_M.gguf --port 8080
```

* **Streaming Chat Completions (`POST /v1/chat/completions`):**
  ```bash
  curl -N http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Why do ShortConv layers use O(1) state memory?"}
      ],
      "temperature": 0.3,
      "stream": true
    }'
  ```
* **L2-Normalized Embeddings (`POST /v1/embeddings`):**
  Skips the LM classifier head (`output.weight`: 65,536 rows on LFM2/2.5, 128,000 rows on LFM3) for fast local RAG indexing:
  ```bash
  curl http://127.0.0.1:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"input": "Lock-free futex spin-barrier thread pool", "encoding_pooling": "mean"}'
  ```

---

## 4. Zero-Dependency Python SDK (`lfm2`)

The Python package wraps `liblfm2.so` using only Python’s standard-library `ctypes` module (`ctypes.CDLL` releases the GIL on every C call).

```python
from lfm2 import LFM2GrammarType, LFM2Model, LFM2PoolingType

with LFM2Model("LFM2.5-1.2B-Instruct-Q4_K_M.gguf") as model:
    print(f"Loaded {model.info.arch} ({model.info.active_isa})")

    with model.create_context(max_seq_len=4096) as ctx:
        # 1. Pull-based zero-mojibake streaming (powered by 8-token Speculative FIFO)
        for chunk in ctx.stream_chat("Explain Linux mmap page faults in 3 bullets."):
            print(chunk, end="", flush=True)

        # 2. Zero-Overhead Hybrid State Branching (~15 us ShortConv + GQA rollback!)
        with ctx.branch():
            structured = ctx.chat(
                "Summarize the 3 bullets above as a JSON object.",
                grammar=LFM2GrammarType.JSON_OBJECT,
            )
            print("\nBranched JSON:", structured)

        # Outside `with ctx.branch():`, all 10 ShortConv ring buffers and `ctx.pos`
        # are restored bit-for-bit to the end of Turn 1!

        # 3. Extract L2-normalized embedding vector (float[2048], ||v||_2 = 1.0)
        vec = ctx.embed("Hybrid ShortConv and GQA architecture", LFM2PoolingType.LAST_TOKEN)
```

For **LFM3 (`general.architecture == "lfm3"`)**, `LFM2Model` auto-dispatches to the isolated LFM3 engine, or you can enforce architecture validation using [`LFM3Model`](python/lfm2.py#L665) / [`load_lfm3_model`](python/lfm2.py#L707) (raises `ValueError` if given a non-LFM3 GGUF):

```python
from lfm2 import LFM3Model, load_lfm3_model  # or from python.lfm2 import ...

with load_lfm3_model("models/LFM3-1.7B-Q4_K_M.gguf", warmup_pages=True) as model:
    with model.create_context(max_seq_len=2048) as ctx:
        reply = ctx.chat(
            "What is the capital of France?",
            max_tokens=512,
            max_think_tokens=128,  # Optional: caps <think> tokens before forcing </think>
        )
        print(reply)
```

---

## 5. Android Mobile Deployment (`arm64-v8a`)

**LFM2.c** delivers turnkey mobile support for modern Android smartphone SoCs (Qualcomm Snapdragon, Google Tensor, MediaTek Dimensity, Samsung Exynos) via both **Native Android Studio (Kotlin & Java)** and **Python / Kivy**:

### Native Kotlin & Java SDK (Android Studio)
Drop `liblfm2.so` and the `ai.liquid.lfm2` package directly into any Android Studio project:

```kotlin
import ai.liquid.lfm2.*

// 1. Zero-copy mmap load from internal storage or Scoped Storage (Content URI)
val model = LFM2Model.load("/data/data/com.example.app/files/LFM2.5-1.2B-Instruct-Q4_K_M.gguf")

// 2. Stateful context with auto-planned threads (pinned to Prime/Performance cores)
val context = model.createContext(contextParams { maxSeqLen = 2048; numThreads = 4 })
context.setSystemPrompt("You are an on-device AI assistant.")

// 3. Stream tokens reactively into Jetpack Compose UI
context.generateSequence("Explain on-device hybrid inference.", sampleParams {
    temperature = 0.25f
    topK = 40
}).forEach { chunk ->
    print(chunk.text)
}

// 4. Android Lifecycle hooks: Drop RAM by ~90% onPause to survive lmkd!
model.releasePageCache() // madvise(MADV_DONTNEED)
model.warmupPageCache()  // Sequential background pre-fault on onResume()
```
* Complete Android Studio guide, Jetpack Compose `ViewModel` with Coroutines `Flow`, and Scoped Storage `loadFromFd()`: see **[`docs/ANDROID_KOTLIN_INTEGRATION.md`](docs/ANDROID_KOTLIN_INTEGRATION.md)**.
* Run end-to-end JNI test suite: `make test-jni`

### Python / Kivy Mobile Runtime
**LFM2.c** also includes a custom `python-for-android` (`p4a`) recipe (`p4a-recipes/lfm2/__init__.py`) and an Android-hardened runtime:
1. **Android 15+ (16 KB Page Size) & Scoped Storage `fd` Support:**
   Uses Android’s Storage Access Framework (`Intent.ACTION_OPEN_DOCUMENT`) + `ParcelFileDescriptor.detachFd()` to `mmap` `.gguf` models directly from the user’s `Downloads` folder without legacy storage permissions or file copying.
2. **Surviving the Android Low Memory Killer (`lmkd`):**
   When the app enters `on_pause()`, `model.release_page_cache()` invokes dual-layer `madvise(..., MADV_DONTNEED)` and `posix_fadvise(..., POSIX_FADV_DONTNEED)`—instantly dropping ~700 MB from process RSS while keeping the `160 KB` ShortConv state and `FP16` GQA KV-cache alive in RAM.
3. **Cross-Compile Standalone Shared Library:**
   ```bash
   make android ANDROID_NDK_HOME=/path/to/android-ndk-r26d ANDROID_API=26
   ```
* Detailed Kivy documentation: see **[`docs/ANDROID_KIVY_INTEGRATION.md`](docs/ANDROID_KIVY_INTEGRATION.md)**.

---

## Numerical Parity & TDD Verification (`tests/verify_parity.py`)

Before tagging a release or modifying SIMD kernels, run the 5-tier verification suite:

```bash
# Run Tiers 1-4 (BPE Contract, Batched GEMM vs M=1 Parity, 0.0-ULP Checkpoint Rollback, L2 Norm)
make parity MODEL=models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf QUANT=Q4_K_M
make parity MODEL=models/LFM3-1.7B-Q4_K_M.gguf QUANT=Q4_K_M

# Run dedicated LFM3 unit & real-model engine suite (ShortConv7 ring, SWA w=256, Checkpoint/Cache/SinkShift)
./bin/test_lfm3 models/LFM3-1.7B-Q4_K_M.gguf

# Generate a Golden Reference Fixture (.lfm_golden) from Q8_0/F16 or Hugging Face FP32
python3 tests/verify_parity.py \
    -m LFM2.5-1.2B-Instruct-Q8_0.gguf \
    --golden-file tests/lfm2_1.2b.lfm_golden \
    --generate-golden

# Certify Q4_K_M / IQ4_XS Kernels Against the Golden Fixture (Cosine Sim, KL Div, Top-5)
python3 tests/verify_parity.py \
    -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
    -q Q4_K_M \
    --golden-file tests/lfm2_1.2b.lfm_golden
```

---

## Documentation

Start at the **[Documentation Hub (`docs/README.md`)](docs/README.md)** or jump directly to a guide:

| Guide | Covers |
| :--- | :--- |
| [Desktop GUI (`LFM Studio`)](docs/DESKTOP_GUI.md) | Pure-C11 single-header IMGUI ([`include/lfm_ui.h`](include/lfm_ui.h)), X11/Xft frontend ([`bin/lfm2-gui`](tools/lfm2_gui.c)), Local & Server SSE modes |
| [Engine Architecture](docs/ARCHITECTURE.md) | Hybrid ShortConv + GQA + MoE math, AVX2/AVX-VNNI & ARM64 NEON/DotProd kernels, `.lfmcache`, `epoll` server, Speculative PLD |
| [LFM3 vs. LFM2/LFM2.5 Guide](docs/LFM3_VS_LFM2_GUIDE.md) | Tri-hybrid SWA + 7-tap ShortConv + Sandwich Norm + `attn_gate` + partial RoPE architectural diff & GGUF inventory |
| [LFM3 Implementation Plan](docs/LFM3_IMPLEMENTATION_PLAN.md) | Isolated LFM3 engine roadmap (`tools/lfm3_engine.c`), LFM2 vs. LFM3 runtime matrix & parity gates |
| [Benchmarks & Roofline](docs/BENCHMARKS.md) | Empirical `LFM2.c` vs. `llama.cpp` comparisons, RAM footprint, cold load times, and DRAM roofline math |
| [Android Studio (Kotlin / Java)](docs/ANDROID_KOTLIN_INTEGRATION.md) | CMake/JNI setup, Kotlin Coroutines `Flow`, Java SDK, SAF Scoped Storage `fd`, and `lmkd` memory hooks |
| [Android (Python / Kivy)](docs/ANDROID_KIVY_INTEGRATION.md) | NDK cross-compilation, `p4a` / `buildozer` recipe, async 60 FPS UI bridge, and SAF `detachFd()` |
| [Windows 11 & macOS Porting](docs/WINDOWS_AND_MAC_PORTING.md) | Exact Win32 & Darwin porting boundaries (`mmap`, `WaitOnAddress`, `WSAPoll`/`kqueue`, `LfmUiBackend`) |
| [Test Suite & QA](docs/TEST_SUITE.md) | 32 C test binaries, Python/JNI/Kotlin/HTTP suites, 5-tier parity verification, ASan/TSan, and static ELF packaging |
| [Engineering Roadmap](docs/ROADMAP.md) | Five-phase `v1.0.0` milestone history and Gen-1 vs. LFM2 module lineage |

---

## License

Released under the MIT License.
