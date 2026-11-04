#define _POSIX_C_SOURCE 200809L
#include "lfm_fused.h"
#include "lfm_simd.h"

#if LFM_HAS_AVX2
#include <immintrin.h>
#endif

void lfm_fused_liv_gate_conv_step(
    float *y_out,
    const float *bcx,
    const float *conv_w,
    const float *conv_bias,
    LFMConvState *state,
    uint32_t hidden_dim
) {
    const float *B = bcx;
    const float *C = bcx + hidden_dim;
    const float *x = bcx + 2 * hidden_dim;

    float *s0 = state->s0;
    float *s1 = state->s1;

#if LFM_HAS_AVX2
    uint32_t c = 0;
    for (; c + 8 <= hidden_dim; c += 8) {
        // Load B, C, x
        __m256 vb = _mm256_loadu_ps(B + c);
        __m256 vc = _mm256_loadu_ps(C + c);
        __m256 vx = _mm256_loadu_ps(x + c);

        // Gate 1: u = B * x
        __m256 vu = _mm256_mul_ps(vb, vx);

        // Load historical states
        __m256 vs0 = _mm256_loadu_ps(s0 + c);
        __m256 vs1 = _mm256_loadu_ps(s1 + c);

        // Load convolution weights (3 taps per channel)
        float w0_arr[8], w1_arr[8], w2_arr[8];
        if (conv_w) {
            for (int j = 0; j < 8; ++j) {
                w0_arr[j] = conv_w[(c + j) * 3 + 0];
                w1_arr[j] = conv_w[(c + j) * 3 + 1];
                w2_arr[j] = conv_w[(c + j) * 3 + 2];
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                w0_arr[j] = 0.0f;
                w1_arr[j] = 0.0f;
                w2_arr[j] = 1.0f;
            }
        }
        __m256 vw0 = _mm256_loadu_ps(w0_arr);
        __m256 vw1 = _mm256_loadu_ps(w1_arr);
        __m256 vw2 = _mm256_loadu_ps(w2_arr);
        __m256 vbias = conv_bias ? _mm256_loadu_ps(conv_bias + c) : _mm256_setzero_ps();

        // Depthwise Conv1D: conv_out = w0*s0 + w1*s1 + w2*u + bias
        __m256 vconv = _mm256_fmadd_ps(vw0, vs0, vbias);
        vconv = _mm256_fmadd_ps(vw1, vs1, vconv);
        vconv = _mm256_fmadd_ps(vw2, vu, vconv);

        // Update rolling ring-buffer states: s0 = s1, s1 = u
        _mm256_storeu_ps(s0 + c, vs1);
        _mm256_storeu_ps(s1 + c, vu);

        // Gate 2: y = C * conv_out
        __m256 vy = _mm256_mul_ps(vc, vconv);

        // Store output directly to projection buffer
        _mm256_storeu_ps(y_out + c, vy);
    }

    // Scalar cleanup
    for (; c < hidden_dim; ++c) {
        float u = B[c] * x[c];
        float w0 = conv_w ? conv_w[c * 3 + 0] : 0.0f;
        float w1 = conv_w ? conv_w[c * 3 + 1] : 0.0f;
        float w2 = conv_w ? conv_w[c * 3 + 2] : 1.0f;
        float b  = conv_bias ? conv_bias[c] : 0.0f;

        float conv_out = (w0 * s0[c]) + (w1 * s1[c]) + (w2 * u) + b;
        s0[c] = s1[c];
        s1[c] = u;

        y_out[c] = C[c] * conv_out;
    }
#else
    for (uint32_t c = 0; c < hidden_dim; ++c) {
        float u = B[c] * x[c];
        float w0 = conv_w ? conv_w[c * 3 + 0] : 0.0f;
        float w1 = conv_w ? conv_w[c * 3 + 1] : 0.0f;
        float w2 = conv_w ? conv_w[c * 3 + 2] : 1.0f;
        float b  = conv_bias ? conv_bias[c] : 0.0f;

        float conv_out = (w0 * s0[c]) + (w1 * s1[c]) + (w2 * u) + b;
        s0[c] = s1[c];
        s1[c] = u;

        y_out[c] = C[c] * conv_out;
    }
#endif
}
