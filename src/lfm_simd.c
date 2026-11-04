#define _POSIX_C_SOURCE 200809L
#include "lfm_simd.h"
#include <math.h>
#include <string.h>

#if LFM_HAS_AVX2
#include <immintrin.h>
#endif

// Fast software conversion from IEEE-754 FP16 to FP32
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF);

    if (exp == 0) {
        if (frac == 0) {
            uint32_t u = sign;
            float f;
            memcpy(&f, &u, 4);
            return f;
        }
        while ((frac & 0x0400) == 0) {
            frac <<= 1;
            exp--;
        }
        exp++;
        frac &= ~0x0400;
        exp = (exp + (127 - 15)) << 23;
        frac <<= 13;
        uint32_t u = sign | exp | frac;
        float f;
        memcpy(&f, &u, 4);
        return f;
    } else if (exp == 0x1F) {
        uint32_t u = sign | 0x7F800000 | (frac << 13);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }

    exp = (exp + (127 - 15)) << 23;
    frac <<= 13;
    uint32_t u = sign | exp | frac;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#if LFM_HAS_AVX2
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    float buf[4];
    _mm_storeu_ps(buf, vsum);
    return buf[0] + buf[1] + buf[2] + buf[3];
}
#endif

float lfm_vec_dot_f32(const float *a, const float *b, size_t n) {
#if LFM_HAS_AVX2
    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }

    __m256 total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = hsum256_ps(total);

    // Remainder loop
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void lfm_vec_mul_f32(float *out, const float *a, const float *b, size_t n) {
#if LFM_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(out + i,      _mm256_mul_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + 8,  _mm256_mul_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8)));
        _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
        _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

void lfm_vec_add_f32(float *out, const float *a, const float *b, size_t n) {
#if LFM_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(out + i,      _mm256_add_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + 8,  _mm256_add_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8)));
        _mm256_storeu_ps(out + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
        _mm256_storeu_ps(out + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

void lfm_vec_rmsnorm_f32(float *out, const float *x, const float *weight, size_t n, float eps) {
#if LFM_HAS_AVX2
    size_t i = 0;
    __m256 sum_sq_vec = _mm256_setzero_ps();

    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        __m256 v2 = _mm256_loadu_ps(x + i + 16);
        __m256 v3 = _mm256_loadu_ps(x + i + 24);

        sum_sq_vec = _mm256_fmadd_ps(v0, v0, sum_sq_vec);
        sum_sq_vec = _mm256_fmadd_ps(v1, v1, sum_sq_vec);
        sum_sq_vec = _mm256_fmadd_ps(v2, v2, sum_sq_vec);
        sum_sq_vec = _mm256_fmadd_ps(v3, v3, sum_sq_vec);
    }

    float sum_sq = hsum256_ps(sum_sq_vec);
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }

    float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)n) + eps);
    __m256 vrsqrt = _mm256_set1_ps(rsqrt_val);

    i = 0;
    if (weight) {
        for (; i + 16 <= n; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(x + i);
            __m256 vw0 = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx0, vrsqrt), vw0));

            __m256 vx1 = _mm256_loadu_ps(x + i + 8);
            __m256 vw1 = _mm256_loadu_ps(weight + i + 8);
            _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(_mm256_mul_ps(vx1, vrsqrt), vw1));
        }
        for (; i < n; ++i) {
            out[i] = x[i] * rsqrt_val * weight[i];
        }
    } else {
        for (; i + 16 <= n; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(vx0, vrsqrt));

            __m256 vx1 = _mm256_loadu_ps(x + i + 8);
            _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(vx1, vrsqrt));
        }
        for (; i < n; ++i) {
            out[i] = x[i] * rsqrt_val;
        }
    }
#else
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)n) + eps);
    if (weight) {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * rsqrt_val * weight[i];
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * rsqrt_val;
    }
#endif
}

void lfm_vec_silu_f32(float *out, const float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float val = x[i];
        out[i] = val / (1.0f + expf(-val));
    }
}

float lfm_vec_dot_q8_0_f32(const void *row_q8, const float *x, size_t n) {
    const BlockQ8_0_SIMD *blocks = (const BlockQ8_0_SIMD*)row_q8;
    size_t nb = n / 32;
    float total = 0.0f;

#if LFM_HAS_AVX2
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        const int8_t *qs = blocks[b].qs;
        const float *xb = x + b * 32;

        // Load 16 bytes, sign extend to 16-bit
        __m128i raw0 = _mm_loadu_si128((const __m128i*)qs);
        __m128i raw1 = _mm_loadu_si128((const __m128i*)(qs + 16));

        __m256i i16_0 = _mm256_cvtepi8_epi16(raw0);
        __m256i i16_1 = _mm256_cvtepi8_epi16(raw1);

        // Convert to floats (4 x 8 floats = 32 floats)
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16_0)));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(i16_0, 1)));
        __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16_1)));
        __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(i16_1, 1)));

        __m256 p0 = _mm256_mul_ps(f0, _mm256_loadu_ps(xb));
        __m256 p1 = _mm256_mul_ps(f1, _mm256_loadu_ps(xb + 8));
        __m256 p2 = _mm256_mul_ps(f2, _mm256_loadu_ps(xb + 16));
        __m256 p3 = _mm256_mul_ps(f3, _mm256_loadu_ps(xb + 24));

        __m256 sum256 = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
        total += hsum256_ps(sum256) * d;
    }
#else
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        const float *xb = x + b * 32;
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            sum += (float)blocks[b].qs[i] * xb[i];
        }
        total += sum * d;
    }
#endif

    return total;
}

float lfm_vec_dot_q4_0_f32(const void *row_q4, const float *x, size_t n) {
    const BlockQ4_0_SIMD *blocks = (const BlockQ4_0_SIMD*)row_q4;
    size_t nb = n / 32;
    float total = 0.0f;

    for (size_t b = 0; b < nb; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        const uint8_t *qs = blocks[b].qs;
        const float *xb = x + b * 32;

        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            int8_t v0 = (int8_t)(qs[i] & 0x0F) - 8;
            int8_t v1 = (int8_t)(qs[i] >> 4) - 8;
            sum += (float)v0 * xb[i] + (float)v1 * xb[i + 16];
        }
        total += sum * d;
    }

    return total;
}

#include "lfm2_simd_avx2.h"

float lfm_vec_dot_q4_K_f32(const void *row_q4_k, const float *x, size_t n) {
    if (!row_q4_k || !x || n == 0) return 0.0f;
#if LFM2_HAS_AVX2
    block_q8_K_t xq[32];
    block_q8_K_t *xq_ptr = (n <= 8192) ? xq : (block_q8_K_t*)malloc((n / 256 + 1) * sizeof(block_q8_K_t));
    if (!xq_ptr) return 0.0f;
    lfm2_quantize_row_q8_K_avx2(x, xq_ptr, (uint32_t)n);
    float res = lfm2_vec_dot_q4_K_q8_K_avx2((const block_q4_K_t*)row_q4_k, xq_ptr, (uint32_t)n);
    if (xq_ptr != xq) free(xq_ptr);
    return res;
#else
    return lfm2_vec_dot_q4_K_fallback((const block_q4_K_t*)row_q4_k, x, (uint32_t)n);
#endif
}

float lfm_vec_dot_q6_K_f32(const void *row_q6_k, const float *x, size_t n) {
    if (!row_q6_k || !x || n == 0) return 0.0f;
#if LFM2_HAS_AVX2
    block_q8_K_t xq[32];
    block_q8_K_t *xq_ptr = (n <= 8192) ? xq : (block_q8_K_t*)malloc((n / 256 + 1) * sizeof(block_q8_K_t));
    if (!xq_ptr) return 0.0f;
    lfm2_quantize_row_q8_K_avx2(x, xq_ptr, (uint32_t)n);
    float res = lfm2_vec_dot_q6_K_q8_K_avx2((const block_q6_K_t*)row_q6_k, xq_ptr, (uint32_t)n);
    if (xq_ptr != xq) free(xq_ptr);
    return res;
#else
    return lfm2_vec_dot_q6_K_fallback((const block_q6_K_t*)row_q6_k, x, (uint32_t)n);
#endif
}

void lfm_matvec_simd(
    float *out,
    const void *weight,
    GGMLType w_type,
    const float *x,
    size_t out_dim,
    size_t in_dim
) {
    if (!out || !weight || !x) return;

    if (w_type == GGML_TYPE_F32) {
        const float *w = (const float*)weight;
        for (size_t r = 0; r < out_dim; ++r) {
            out[r] = lfm_vec_dot_f32(w + r * in_dim, x, in_dim);
        }
    } else if (w_type == GGML_TYPE_Q8_0) {
        size_t blocks_per_row = in_dim / 32;
        const BlockQ8_0_SIMD *blocks = (const BlockQ8_0_SIMD*)weight;
        for (size_t r = 0; r < out_dim; ++r) {
            out[r] = lfm_vec_dot_q8_0_f32(blocks + r * blocks_per_row, x, in_dim);
        }
    } else if (w_type == GGML_TYPE_Q4_0) {
        size_t blocks_per_row = in_dim / 32;
        const BlockQ4_0_SIMD *blocks = (const BlockQ4_0_SIMD*)weight;
        for (size_t r = 0; r < out_dim; ++r) {
            out[r] = lfm_vec_dot_q4_0_f32(blocks + r * blocks_per_row, x, in_dim);
        }
    } else if (w_type == GGML_TYPE_Q4_K) {
        size_t blocks_per_row = in_dim / 256;
        const block_q4_K_t *blocks = (const block_q4_K_t*)weight;
#if LFM2_HAS_AVX2
        block_q8_K_t xq_stack[32];
        block_q8_K_t *xq_ptr = (in_dim <= 8192) ? xq_stack : (block_q8_K_t*)malloc((in_dim / 256 + 1) * sizeof(block_q8_K_t));
        if (xq_ptr) {
            lfm2_quantize_row_q8_K_avx2(x, xq_ptr, (uint32_t)in_dim);
            for (size_t r = 0; r < out_dim; ++r) {
                out[r] = lfm2_vec_dot_q4_K_q8_K_avx2(blocks + r * blocks_per_row, xq_ptr, (uint32_t)in_dim);
            }
            if (xq_ptr != xq_stack) free(xq_ptr);
        }
#else
        for (size_t r = 0; r < out_dim; ++r) {
            out[r] = lfm2_vec_dot_q4_K_fallback(blocks + r * blocks_per_row, x, (uint32_t)in_dim);
        }
#endif
    } else if (w_type == GGML_TYPE_Q6_K) {
        size_t blocks_per_row = in_dim / 256;
        const block_q6_K_t *blocks = (const block_q6_K_t*)weight;
#if LFM2_HAS_AVX2
        block_q8_K_t xq_stack[32];
        block_q8_K_t *xq_ptr = (in_dim <= 8192) ? xq_stack : (block_q8_K_t*)malloc((in_dim / 256 + 1) * sizeof(block_q8_K_t));
        if (xq_ptr) {
            lfm2_quantize_row_q8_K_avx2(x, xq_ptr, (uint32_t)in_dim);
            for (size_t r = 0; r < out_dim; ++r) {
                out[r] = lfm2_vec_dot_q6_K_q8_K_avx2(blocks + r * blocks_per_row, xq_ptr, (uint32_t)in_dim);
            }
            if (xq_ptr != xq_stack) free(xq_ptr);
        }
#else
        for (size_t r = 0; r < out_dim; ++r) {
            out[r] = lfm2_vec_dot_q6_K_fallback(blocks + r * blocks_per_row, x, (uint32_t)in_dim);
        }
#endif
    } else if (w_type == GGML_TYPE_F16) {
        const uint16_t *w = (const uint16_t*)weight;
        for (size_t r = 0; r < out_dim; ++r) {
            const uint16_t *row = w + r * in_dim;
            float sum = 0.0f;
            for (size_t c = 0; c < in_dim; ++c) {
                sum += fp16_to_fp32(row[c]) * x[c];
            }
            out[r] = sum;
        }
    } else {
        memset(out, 0, out_dim * sizeof(float));
    }
}
