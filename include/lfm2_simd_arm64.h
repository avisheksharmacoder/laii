#ifndef LFM2_SIMD_ARM64_H
#define LFM2_SIMD_ARM64_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
    #include <arm_neon.h>
    #include <sys/auxv.h>
    #ifndef HWCAP_ASIMD
        #define HWCAP_ASIMD   (1UL << 1)
    #endif
    #ifndef HWCAP_FPHP
        #define HWCAP_FPHP    (1UL << 9)
    #endif
    #ifndef HWCAP_ASIMDDP
        #define HWCAP_ASIMDDP (1UL << 20)
    #endif
    #ifndef HWCAP2_I8MM
        #define HWCAP2_I8MM   (1UL << 13)
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Quantized Block Definitions (GGML Binary-Compatible Layout)
 * ============================================================================ */
#pragma pack(push, 1)

#ifndef LFM2_BLOCK_Q4_0_DEFINED
#define LFM2_BLOCK_Q4_0_DEFINED
typedef struct {
    uint16_t d;           /* FP16 delta scale */
    uint8_t  qs[16];      /* 32 4-bit unsigned nibbles (offset by -8) */
} block_q4_0_t;
#endif

#ifndef LFM2_BLOCK_Q8_0_DEFINED
#define LFM2_BLOCK_Q8_0_DEFINED
typedef struct {
    uint16_t d;           /* FP16 delta scale */
    int8_t   qs[32];      /* 32 signed 8-bit weights */
} block_q8_0_t;
#endif

#ifndef LFM2_BLOCK_Q4_K_DEFINED
#define LFM2_BLOCK_Q4_K_DEFINED
typedef struct {
    uint16_t d;           /* FP16 super-block scale for quantized scales */
    uint16_t dmin;        /* FP16 super-block scale for quantized mins   */
    uint8_t  scales[12];  /* 6-bit packed scales and mins (8 sub-blocks) */
    uint8_t  qs[128];     /* 256 4-bit unsigned weights                  */
} block_q4_K_t;
#endif

#ifndef LFM2_BLOCK_Q5_K_DEFINED
#define LFM2_BLOCK_Q5_K_DEFINED
typedef struct {
    uint16_t d;           /* FP16 super-block scale for quantized scales */
    uint16_t dmin;        /* FP16 super-block scale for quantized mins   */
    uint8_t  scales[12];  /* 6-bit packed scales and mins (8 sub-blocks) */
    uint8_t  qh[32];      /* 5th high bit for each of the 256 weights    */
    uint8_t  qs[128];     /* Low 4 bits for each of the 256 weights      */
} block_q5_K_t;
#endif

#ifndef LFM2_BLOCK_Q6_K_DEFINED
#define LFM2_BLOCK_Q6_K_DEFINED
typedef struct {
    uint8_t  ql[128];     /* Lower 4 bits of 256 6-bit weights           */
    uint8_t  qh[64];      /* Upper 2 bits of 256 6-bit weights           */
    int8_t   scales[16];  /* 8-bit signed sub-block scales               */
    uint16_t d;           /* FP16 super-block scale                      */
} block_q6_K_t;
#endif

#ifndef LFM2_BLOCK_IQ4_XS_DEFINED
#define LFM2_BLOCK_IQ4_XS_DEFINED
typedef struct {
    uint16_t d;           /* FP16 super-block scale                      */
    uint16_t scales_h;    /* High 2 bits of 8 6-bit sub-block scales     */
    uint8_t  scales_l[4]; /* Low 4 bits of 8 6-bit sub-block scales      */
    uint8_t  qs[128];     /* 256 4-bit indices into kvalues_iq4nl[16]    */
} block_iq4_xs_t;
#endif

#ifndef LFM2_BLOCK_Q8_K_DEFINED
#define LFM2_BLOCK_Q8_K_DEFINED
typedef struct {
    float   d;            /* FP32 scale: max_abs / 127.0f                */
    int8_t  qs[256];      /* 256 signed 8-bit quantized activations      */
    int16_t bsums[16];    /* Precomputed sum of each 16-element sub-block*/
} block_q8_K_t;
#endif

#pragma pack(pop)

#ifndef LFM2_KVALUES_IQ4NL_DEFINED
#define LFM2_KVALUES_IQ4NL_DEFINED
/* Official GGML non-linear 4-bit importance grid for IQ4_NL and IQ4_XS */
static const _Alignas(16) int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};
#endif

#ifndef GGML_TYPE_COUNT
#define GGML_TYPE_COUNT 32
#endif

/* Type indices matching gguf.h / GGML */
#ifndef GGML_TYPE_DEFINITIONS
#define GGML_TYPE_DEFINITIONS
#define LFM2_GGML_TYPE_F32     0
#define LFM2_GGML_TYPE_F16     1
#define LFM2_GGML_TYPE_Q4_0    2
#define LFM2_GGML_TYPE_Q8_0    8
#define LFM2_GGML_TYPE_Q4_K    12
#define LFM2_GGML_TYPE_Q5_K    13
#define LFM2_GGML_TYPE_Q6_K    14
#define LFM2_GGML_TYPE_Q8_K    15
#define LFM2_GGML_TYPE_IQ4_XS  23
#endif

/* ============================================================================
 * 2. Function Pointer Types & Mobile Kernel Table
 * ============================================================================ */
#ifndef LFM2_KERNEL_FN_TYPES_DEFINED
#define LFM2_KERNEL_FN_TYPES_DEFINED
typedef void  (*lfm2_quantize_q8_K_fn)(const float * restrict x,
                                       block_q8_K_t * restrict y,
                                       uint32_t n_cols);

typedef float (*lfm2_vec_dot_1x1_fn)(const void * restrict w_row,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols);

typedef void  (*lfm2_vec_dot_1x4_fn)(const void * restrict w_row0,
                                     size_t row_blocks,
                                     const block_q8_K_t * restrict x_q8,
                                     uint32_t n_cols,
                                     float out4[4]);
#endif

typedef struct {
    const char *isa_name;
    bool        has_neon;
    bool        has_fp16;
    bool        has_dotprod;
    bool        has_i8mm;

    lfm2_quantize_q8_K_fn quantize_q8_K;
    lfm2_vec_dot_1x1_fn   vec_dot_1x1[GGML_TYPE_COUNT];
    lfm2_vec_dot_1x4_fn   vec_dot_1x4[GGML_TYPE_COUNT];
} LFM2MobileKernelTable;

/* Dynamic initializer */
void lfm2_init_mobile_kernels(LFM2MobileKernelTable *kt);

/* Scalar reference kernels */
void  lfm2_quantize_row_q8_K_scalar(const float * restrict x, block_q8_K_t * restrict y, uint32_t n_cols);
float lfm2_vec_dot_q4_K_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);
float lfm2_vec_dot_q6_K_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);
float lfm2_vec_dot_q5_K_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);
float lfm2_vec_dot_iq4_xs_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);
float lfm2_vec_dot_q4_0_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);
float lfm2_vec_dot_q8_0_q8_K_scalar(const void * restrict w_raw, const block_q8_K_t * restrict x_q8, uint32_t n_cols);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_SIMD_ARM64_H */
