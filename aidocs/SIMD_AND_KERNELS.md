# SIMD Vectorization, Quantization Formats & Kernel Architecture

This document provides a technical guide to low-level vector intrinsics, block quantization memory layouts, and multi-threaded kernel execution in LAII.

---

## 1. SIMD Vectorization Standards (AVX2 + FMA + F16C)

All performance-critical kernels in LAII are vectorized using 256-bit AVX2, FMA (Fused Multiply-Add), and hardware F16C intrinsics.

### 1.1 Instruction Conventions
* **`_mm256_fmadd_ps(a, b, c)`:** Evaluates $a \cdot b + c$ with a single rounding step. Used across all dot products and matrix multiplications.
* **`_mm256_fmsub_ps(a, b, c)`:** Evaluates $a \cdot b - c$. Used for NeoX half-split RoPE rotation.
* **`_mm256_cvtps_ph(a, 0)` & `_mm256_cvtph_ps(a)`:** Hardware IEEE-754 FP32 $\leftrightarrow$ FP16 conversion in vector registers. Used for KV-cache streaming and RoPE rotations.
* **`_mm256_maddubs_epi16(a, b)`:** Multiplies unsigned 8-bit integers by signed 8-bit integers and adds adjacent pairs into signed 16-bit integers. Core workhorse of `Q4_K`, `Q6_K`, `Q4_0`, and `Q5_K` integer dot products.

### 1.2 Multi-ISA CPUID Dispatch Pattern
To support older x86_64 CPUs without `SIGILL` crashes while maximizing performance on modern CPUs, kernels are compiled with GCC target attributes:

```c
__attribute__((target("avx2,fma,f16c")))
float my_dot_product_avx2(const float *a, const float *b, size_t n) {
    // Hardware accelerated implementation
}

float my_dot_product_scalar(const float *a, const float *b, size_t n) {
    // Portable C11 fallback
}

// Function pointer selected at startup
typedef float (*dot_fn_t)(const float *, const float *, size_t);
dot_fn_t my_dot_product = my_dot_product_scalar;

void lfm2_init_cpu_kernels(void) {
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        my_dot_product = my_dot_product_avx2;
    }
}
```

---

## 2. Quantization Block Formats & Memory Layouts

LAII supports standard GGUF quantization formats:

| Format | Block Size | Byte Size | Description & Bit-Packing |
| :--- | :---: | :---: | :--- |
| **`block_q8_K`** | 256 | 292 B | `float d` (4 B) + `int8_t qs[256]` (256 B) + `int16_t bsums[16]` (32 B). Pre-quantization target for activations. |
| **`block_q4_K`** | 256 | 144 B | `fp16 d` + `fp16 dmin` + 12-byte scales + `uint8_t qs[128]` (4-bit nibbles). |
| **`block_q6_K`** | 256 | 210 B | `uint8_t ql[128]` (lower 4 bits) + `uint8_t qh[64]` (upper 2 bits) + `int8_t scales[16]` + `fp16 d`. |
| **`block_q4_0`** | 32 | 18 B | `fp16 d` + `uint8_t qs[16]` (32 4-bit nibbles, zero-point offset $-8$). |
| **`block_q5_K`** | 256 | 176 B | `d`, `dmin`, `scales`, `uint8_t qh[32]`, `uint8_t qs[128]` (5-bit values). |
| **`block_iq4_xs`**| 256 | 136 B | `fp16 d` + `uint16_t scales_h` + `uint8_t scales_l[4]` + `uint8_t qs[128]` (non-linear 16-entry grid). |

### 2.1 The 1-Clock `IQ4_XS` Non-Linear Grid Lookup
In `IQ4_XS`, the 4-bit indices are mapped to non-linear float values via a 16-entry codebook.
* **Standard scalar approach:** 256 array lookups per super-block.
* **LAII AVX2 approach:** Broadcasts the 16-entry byte grid into a `__m256i` register and uses `_mm256_shuffle_epi8` (`vpshufb`) to translate 32 4-bit indices into non-linear weights in **1 CPU clock cycle**.

---

## 3. High-Throughput Kernel Patterns

### 3.1 1x4 Row Unrolling & 4-Row Horizontal Reduction
When evaluating GEMV $y = W x$, loading the activation vector $x$ once per row is memory-inefficient.
* **1x4 Row Unrolling:** Evaluates 4 consecutive weight rows $(r_0, r_1, r_2, r_3)$ in parallel:
  - Activation super-block $x_q$ is held in vector registers.
  - Four dot-product accumulators are updated simultaneously.
* **4-Row Horizontal Reduction (`hsum4_epi32_avx2`):**
  Reduces four 8-lane 32-bit accumulators into a single 4-lane `__m128i` vector in 4 instructions using `_mm256_hadd_epi32`:
  ```c
  static inline __m128i hsum4_epi32_avx2(__m256i r0, __m256i r1, __m256i r2, __m256i r3) {
      __m256i h01   = _mm256_hadd_epi32(r0, r1);
      __m256i h23   = _mm256_hadd_epi32(r2, r3);
      __m256i h0123 = _mm256_hadd_epi32(h01, h23);
      return _mm_add_epi32(_mm256_castsi256_si128(h0123),
                           _mm256_extracti128_si256(h0123, 1));
  }
  ```

### 3.2 Fused RMSNorm + Q8_K Quantizer
* Traditional engines compute RMSNorm into DRAM, then read it back to quantize to 8-bit.
* [`lfm2_rmsnorm_quantize_q8_K_avx2`](file:///home/avisheks/Documents/python-projects/Python-projects/laii/tools/lfm2_decode_1x4.c#L77) normalizes the activation vector and immediately quantizes each 256-float super-block while data is hot in CPU L1 cache, computing scales and `bsums` with zero intermediate DRAM round-trips.

### 3.3 Register-Resident SwiGLU FFN
* Computes Gate ($w_1$) and Up ($w_3$) projections concurrently in vector registers.
* Evaluates a fast rational approximation of $\text{SiLU}(w_1)$, multiplies by $w_3$, and keeps the result in registers, eliminating DRAM round-trips for intermediate FFN buffers.

---

## 4. Lock-Free Spin-Barrier ThreadPool (`tools/lfm2_threadpool.c`)

* **SMT / Physical Core Discovery:** Reads `/sys/devices/system/cpu/cpu*/topology/` to identify physical core IDs and isolates worker threads via `sched_setaffinity`, preventing Hyper-Threading port saturation.
* **Monotonic Barrier:** Uses `_Alignas(64)` cache-padded atomic generation counters (`atomic_uint_fast64_t`) to achieve ~260 ns dispatch latency.
* **Futex Sleep:** Workers busy-wait with `_mm_pause()` for up to 25,000 iterations before putting the thread to sleep via `FUTEX_WAIT_PRIVATE`, eliminating idle CPU power consumption.

