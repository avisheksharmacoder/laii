#!/usr/bin/env python3
"""
Generates reference tensors and test fixtures using PyTorch
to verify numerical parity of the LFM C inference engine.
"""

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def export_float_array(f, arr: np.ndarray):
    arr_f32 = arr.astype(np.float32).flatten()
    f.write(struct.pack(f"{len(arr_f32)}f", *arr_f32))

def generate_conv1d_fixtures(filename="tests/fixtures_conv1d.bin"):
    torch.manual_seed(42)
    hidden_dim = 64
    k = 3

    # Generate random input
    x_norm = torch.randn(hidden_dim, dtype=torch.float32)

    # Weights
    in_proj_w = torch.randn(3 * hidden_dim, hidden_dim, dtype=torch.float32) * 0.1
    conv_w = torch.randn(hidden_dim, 1, k, dtype=torch.float32) * 0.1
    conv_b = torch.randn(hidden_dim, dtype=torch.float32) * 0.05
    out_proj_w = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.1

    # Initial states
    s0 = torch.randn(hidden_dim, dtype=torch.float32) * 0.2
    s1 = torch.randn(hidden_dim, dtype=torch.float32) * 0.2

    # PyTorch computation
    # 1. in_proj
    bcx = F.linear(x_norm, in_proj_w)
    B, C, x = bcx.chunk(3, dim=-1)

    # 2. Gate 1
    u = B * x

    # 3. Depthwise Conv1D
    # State concatenation: [s0, s1, u]
    # In PyTorch F.conv1d, input is (1, hidden_dim, 3)
    conv_in = torch.stack([s0, s1, u], dim=-1).unsqueeze(0)
    conv_out = F.conv1d(conv_in, conv_w, conv_b, groups=hidden_dim).squeeze(0).squeeze(-1)

    # State update
    next_s0 = s1.clone()
    next_s1 = u.clone()

    # 4. Gate 2
    y = C * conv_out

    # 5. out_proj
    out = F.linear(y, out_proj_w)

    with open(filename, "wb") as f:
        # Dimensions header
        f.write(struct.pack("II", hidden_dim, k))
        # Inputs & weights
        export_float_array(f, x_norm.numpy())
        export_float_array(f, in_proj_w.numpy())
        export_float_array(f, conv_w.squeeze(1).numpy()) # [hidden_dim, 3]
        export_float_array(f, conv_b.numpy())
        export_float_array(f, out_proj_w.numpy())
        export_float_array(f, s0.numpy())
        export_float_array(f, s1.numpy())
        # Expected outputs
        export_float_array(f, out.numpy())
        export_float_array(f, next_s0.numpy())
        export_float_array(f, next_s1.numpy())

    print(f"Generated Conv1D reference fixture: {filename}")

def generate_attention_fixtures(filename="tests/fixtures_attention.bin"):
    torch.manual_seed(42)
    hidden_dim = 64
    n_heads_q = 4
    n_heads_kv = 2
    head_dim = hidden_dim // n_heads_q # 16
    pos = 5
    eps = 1e-5
    theta = 1000000.0

    x_norm = torch.randn(hidden_dim, dtype=torch.float32)

    q_proj_w = torch.randn(n_heads_q * head_dim, hidden_dim) * 0.1
    k_proj_w = torch.randn(n_heads_kv * head_dim, hidden_dim) * 0.1
    v_proj_w = torch.randn(n_heads_kv * head_dim, hidden_dim) * 0.1
    q_norm_w = torch.ones(head_dim)
    k_norm_w = torch.ones(head_dim)
    attn_out_w = torch.randn(hidden_dim, hidden_dim) * 0.1

    # Historical KV cache for pos 0..4
    k_hist = torch.randn(pos, n_heads_kv, head_dim) * 0.2
    v_hist = torch.randn(pos, n_heads_kv, head_dim) * 0.2

    # Projections
    q = F.linear(x_norm, q_proj_w).view(n_heads_q, head_dim)
    k = F.linear(x_norm, k_proj_w).view(n_heads_kv, head_dim)
    v = F.linear(x_norm, v_proj_w).view(n_heads_kv, head_dim)

    # QK RMSNorm
    def rms_norm(t, w):
        variance = t.pow(2).mean(-1, keepdim=True)
        return t * torch.rsqrt(variance + eps) * w

    q = rms_norm(q, q_norm_w)
    k = rms_norm(k, k_norm_w)

    # RoPE
    def apply_rope(t, pos_val):
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
        phi = pos_val * inv_freq
        cos_v = phi.cos()
        sin_v = phi.sin()
        t1, t2 = t[..., :half], t[..., half:]
        rot1 = t1 * cos_v - t2 * sin_v
        rot2 = t2 * cos_v + t1 * sin_v
        return torch.cat([rot1, rot2], dim=-1)

    q_rot = apply_rope(q, pos)
    k_rot = apply_rope(k, pos)

    # Combine with history
    k_all = torch.cat([k_hist, k_rot.unsqueeze(0)], dim=0) # [pos+1, n_heads_kv, head_dim]
    v_all = torch.cat([v_hist, v.unsqueeze(0)], dim=0)     # [pos+1, n_heads_kv, head_dim]

    # Attention
    scale = 1.0 / (head_dim ** 0.5)
    group_size = n_heads_q // n_heads_kv
    out_heads = []

    for hq in range(n_heads_q):
        hkv = hq // group_size
        qh = q_rot[hq] # [head_dim]
        kh = k_all[:, hkv, :] # [pos+1, head_dim]
        vh = v_all[:, hkv, :] # [pos+1, head_dim]

        scores = torch.matmul(kh, qh) * scale
        weights = F.softmax(scores, dim=-1)
        head_out = torch.matmul(weights, vh)
        out_heads.append(head_out)

    attn_concat = torch.cat(out_heads, dim=-1) # [hidden_dim]
    out = F.linear(attn_concat, attn_out_w)

    with open(filename, "wb") as f:
        f.write(struct.pack("IIIIII", hidden_dim, n_heads_q, n_heads_kv, head_dim, pos, 0))
        export_float_array(f, x_norm.numpy())
        export_float_array(f, q_proj_w.numpy())
        export_float_array(f, k_proj_w.numpy())
        export_float_array(f, v_proj_w.numpy())
        export_float_array(f, q_norm_w.numpy())
        export_float_array(f, k_norm_w.numpy())
        export_float_array(f, attn_out_w.numpy())
        export_float_array(f, k_hist.numpy())
        export_float_array(f, v_hist.numpy())
        export_float_array(f, out.numpy())

    print(f"Generated Attention reference fixture: {filename}")

if __name__ == "__main__":
    generate_conv1d_fixtures()
    generate_attention_fixtures()
