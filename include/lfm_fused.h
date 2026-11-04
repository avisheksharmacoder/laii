#ifndef LFM_FUSED_H
#define LFM_FUSED_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "lfm_model.h"
#include "lfm_forward.h"

// Fused LIV ShortConv kernel:
// Given projected B, C, x vectors from in_proj, computes:
//   Gate 1: u = B * x
//   Depthwise Causal Conv1D: conv_out = w0*s0 + w1*s1 + w2*u + bias
//   Ring buffer state update: s0 = s1, s1 = u
//   Gate 2: y = C * conv_out
// In a single register-resident pass with zero intermediate memory writes!
void lfm_fused_liv_gate_conv_step(
    float *y_out,               // [hidden_dim] output for out_proj
    const float *bcx,           // [3 * hidden_dim] (B, C, x concatenated)
    const float *conv_w,        // [hidden_dim, 3] convolution kernel weights
    const float *conv_bias,     // [hidden_dim] optional bias
    LFMConvState *state,        // rolling 2-step state buffers
    uint32_t hidden_dim
);

#endif // LFM_FUSED_H
