#ifndef LFM_SIMD_H
#define LFM_SIMD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "gguf.h"

// Check for AVX2 support at compile time
#if defined(__AVX2__) && defined(__FMA__)
#define LFM_HAS_AVX2 1
#else
#define LFM_HAS_AVX2 0
#endif

// Quantized Block Definitions for Q8_0 and Q4_0
typedef struct {
    uint16_t d;     // FP16 delta scale
    int8_t qs[32];  // 32 quantized int8 values
} BlockQ8_0_SIMD;

typedef struct {
    uint16_t d;     // FP16 delta scale
    uint8_t qs[16]; // 32 4-bit nibbles (low 4 bits = item i, high 4 bits = item i+16)
} BlockQ4_0_SIMD;

// FP32 Vector Primitives
float lfm_vec_dot_f32(const float *a, const float *b, size_t n);
void  lfm_vec_mul_f32(float *out, const float *a, const float *b, size_t n);
void  lfm_vec_add_f32(float *out, const float *a, const float *b, size_t n);
void  lfm_vec_rmsnorm_f32(float *out, const float *x, const float *weight, size_t n, float eps);
void  lfm_vec_silu_f32(float *out, const float *x, size_t n);

// Quantized Block Dot Products
float lfm_vec_dot_q8_0_f32(const void *row_q8, const float *x, size_t n);
float lfm_vec_dot_q4_0_f32(const void *row_q4, const float *x, size_t n);
float lfm_vec_dot_q4_K_f32(const void *row_q4_k, const float *x, size_t n);
float lfm_vec_dot_q6_K_f32(const void *row_q6_k, const float *x, size_t n);

// Unified Matrix-Vector multiplication using hardware-accelerated SIMD
void lfm_matvec_simd(
    float *out,
    const void *weight,
    GGMLType w_type,
    const float *x,
    size_t out_dim,
    size_t in_dim
);

#endif // LFM_SIMD_H
