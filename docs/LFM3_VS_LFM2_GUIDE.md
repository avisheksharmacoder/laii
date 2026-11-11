# LFM3 vs. LFM2 / LFM2.5 Architecture, GGUF Layout & LAII Optimization Guide

> **Status & Scope:** This reference documents the architectural, mathematical, binary GGUF, and performance-engineering differences between Liquid AI's **LFM2 / LFM2.5** (`lfm2` / `lfm2moe`) and **LFM3** (`lfm3`) model families, along with a zero-regression porting and optimization blueprint for **LAII (`LFM2.c`)**.
>
> Currently, LAII natively accelerates **LFM2 / LFM2.5** checkpoints. **LFM3** requires targeted additions described in [§5](#5-laii-codebase-compatibility-audit) and [§6](#6-lfm3-optimization-blueprint-zero-regression-on-lfm2lfm25).

---

## Contents

1. [Generational Overview: LFM2 → LFM2.5 → LFM3](#1-generational-overview-lfm2--lfm25--lfm3)
2. [Layer Dataflow Comparison (Dual-Hybrid vs. Tri-Hybrid)](#2-layer-dataflow-comparison-dual-hybrid-vs-tri-hybrid)
3. [Empirical GGUF Comparison (Local Checkpoints on Disk)](#3-empirical-gguf-comparison-local-checkpoints-on-disk)
   - [3.1 Global Hyperparameters & Payload Breakdown](#31-global-hyperparameters--payload-breakdown)
   - [3.2 Layer-by-Layer Routing Patterns](#32-layer-by-layer-routing-patterns)
   - [3.3 Complete Tensor & Quantization Inventory](#33-complete-tensor--quantization-inventory)
4. [Detailed Operator, State & Tokenizer Differences](#4-detailed-operator-state--tokenizer-differences)
   - [4.1 ShortConv: 3-Tap (`l_cache = 3`) vs. 7-Tap (`l_cache = 7`)](#41-shortconv-3-tap-l_cache--3-vs-7-tap-l_cache--7)
   - [4.2 Attention: Dual-Hybrid Full GQA vs. Tri-Hybrid SWA + Global GQA (`head_dim = 128`)](#42-attention-dual-hybrid-full-gqa-vs-tri-hybrid-swa--global-gqa-head_dim--128)
   - [4.3 Head-Wise Sigmoid Attention Gate (`attn_gate.weight`)](#43-head-wise-sigmoid-attention-gate-attn_gateweight)
   - [4.4 Sandwich Normalization (`post_attention_norm`, `post_ffw_norm`)](#44-sandwich-normalization-post_attention_norm-post_ffw_norm)
   - [4.5 Hybrid & Partial RoPE (`rope_freqs.weight` & `freq_base_swa`)](#45-hybrid--partial-rope-rope_freqsweight--freq_base_swa)
   - [4.6 Vocabulary, Tokenizer, Special Tokens & ChatML Template](#46-vocabulary-tokenizer-special-tokens--chatml-template)
   - [4.7 Multimodal Extensions (`LFM3-VL`, `LFM3-Audio`, `LFM3-Omni`)](#47-multimodal-extensions-lfm3-vl-lfm3-audio-lfm3-omni)
5. [LAII Codebase Compatibility Audit](#5-laii-codebase-compatibility-audit)
6. [LFM3 Optimization Blueprint (Zero Regression on LFM2/LFM2.5)](#6-lfm3-optimization-blueprint-zero-regression-on-lfm2lfm25)
7. [Verification & Parity Certification Protocol](#7-verification--parity-certification-protocol)

---

## 1. Generational Overview: LFM2 → LFM2.5 → LFM3

| Generation | GGUF `general.architecture` | Hybrid Paradigm | Key Architectural Traits | Representative Checkpoints |
| :--- | :--- | :--- | :--- | :--- |
| **LFM2** | `lfm2`, `lfm2moe` | **Dual-Hybrid** (`conv` + `full_attention`) | Double-gated LIV ShortConv (`l_cache = 3`); Full GQA with QK-Norm & NeoX RoPE (`head_dim = 64`); pre-norms only; `65,536` vocab; dense SwiGLU (`350M`–`2.6B`) or auxiliary-loss-free Sigmoid Top-4 MoE (`8B-A1B`, `24B-A2B`) | `LFM2-350M`, `LFM2-1.2B`, `LFM2-8B-A1B`, `LFM2-24B-A2B` |
| **LFM2.5** | `lfm2` | **Dual-Hybrid** (`conv` + `full_attention`) | Same runtime graph and tensor conventions as LFM2; improved pretraining/RL/thinking (`1.2B-Thinking`); `l_cache = 3`; `128K` context (`rope.freq_base = 1e6`); `65,536` vocab (`add_bos_token = True`) | `LFM2.5-230M`, `LFM2.5-350M`, `LFM2.5-1.2B-Instruct`, `LFM2.5-1.2B-Thinking` |
| **LFM3** | `lfm3` | **Tri-Hybrid** (`conv` + `sliding_attention` + `full_attention`) | Expanded 7-tap ShortConv (`l_cache = 7`); `head_dim = 128`; Sliding-Window Attention (`sliding_window = 256`); head-wise Sigmoid `attn_gate`; **Sandwich Norms** (`post_attention_norm`, `post_ffw_norm`); split RoPE (`freq_base = 1e6` + `rope_freqs.weight` 50% partial RoPE on global layers, `freq_base_swa = 10000` on SWA layers); `256K` context; `128,000` vocab (`add_bos_token = False`); native VL/Audio/Omni variants | `LFM3-1.7B`, `LFM3-VL-2B`, `LFM3-Audio-2B`, `LFM3-Omni-2B` |

---

## 2. Layer Dataflow Comparison (Dual-Hybrid vs. Tri-Hybrid)

```text
  LFM2 / LFM2.5 (Dual-Hybrid Block)                 LFM3 (Tri-Hybrid + Sandwich-Norm Block)
  =================================                 ========================================
            x (residual)                                      x (residual)
            |                                                 |
            v                                                 v
    [ attn_norm (RMSNorm) ]                           [ attn_norm (RMSNorm) ]
            |                                                 |
     +------+------+                                   +------+------+------+
     |             |                                   |             |      |
     v (conv)      v (full_attn)                       v (conv)      v (SWA) v (full_attn)
  in_proj       Q, K, V proj                        in_proj       Q, K, V, + attn_gate proj
  Gate 1        QK-RMSNorm                          Gate 1        QK-RMSNorm
  Conv1D k=3    Full RoPE (d=64)                    Conv1D k=7    RoPE (SWA: full d=128, theta=10k;
  Gate 2        Causal Full GQA                     Gate 2              Full: 50% partial, theta=1M)
  out_proj      attn_output (wo)                    out_proj      Attn(Q,K,V) [SWA: w=256 / Full]
     |             |                                   |          * sigmoid(attn_gate[h])
     +------+------+                                   |          attn_output (wo)
            |                                          +------+-------------+
            v                                                 |
     (no post-norm)                                           v
            |                                         [ post_attention_norm (RMSNorm) ]
            v                                                 |
     x = x + op_out                                           v
            |                                          x = x + post_op_out  (see §4.4 note)
            v                                                 |
    [ ffn_norm (RMSNorm) ]                                    v
            |                                         [ ffn_norm (RMSNorm) ]
            v                                                 |
    [ SwiGLU FFN ]                                            v
            |                                         [ SwiGLU FFN ]
     (no post-norm)                                           |
            |                                                 v
            v                                         [ post_ffw_norm (RMSNorm) ]
     x = x + ffn_out                                          |
                                                              v
                                                       x = x + post_ffn_out (see §4.4 note)
```

---

## 3. Empirical GGUF Comparison (Local Checkpoints on Disk)

All figures below were measured directly from the four `Q4_K_M` GGUF v3 checkpoints in [`models/`](../models).

### 3.1 Global Hyperparameters & Payload Breakdown

| Metric / GGUF Key | `LFM2.5-230M` | `LFM2.5-350M` | `LFM2.5-1.2B-Instruct` | `LFM3-1.7B` |
| :--- | :---: | :---: | :---: | :---: |
| **GGUF File Size** | `146.30 MiB` | `218.69 MiB` | `697.04 MiB` | `1,048.69 MiB` |
| **Tensor Count / Payload** | `132` / `144.02 MiB` | `148` / `216.41 MiB` | `148` / `694.76 MiB` | `287` / `1,040.82 MiB` |
| `general.architecture` | `lfm2` | `lfm2` | `lfm2` | `lfm3` |
| `block_count` (`n_layers`) | `14` | `16` | `16` | `24` |
| `embedding_length` (`D`) | `1,024` | `1,024` | `2,048` | `2,048` |
| `feed_forward_length` (`n_ff`) | `2,560` | `4,608` | `8,192` | `7,680` |
| `context_length` (`n_ctx`) | `128,000` | `128,000` | `128,000` | `262,144` |
| `attention.head_count` (`H_q`) | `16` | `16` | `32` | `16` |
| `head_dim` (`key_length`/`value_length`) | `64` | `64` | `64` | `128` |
| `shortconv.l_cache` | `3` | `3` | `3` | `7` |
| `attention.sliding_window` | — | — | — | `256` |
| `rope.freq_base` / `freq_base_swa` | `1,000,000` / — | `1,000,000` / — | `1,000,000` / — | `1,000,000` / `10,000` |
| `rope.dimension_count` | — | — | — | `128` (+ `rope_freqs.weight[64]`) |
| `attention.layer_norm_rms_epsilon` | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| `vocab_size` | `65,536` | `65,536` | `65,536` | `128,000` |
| `tokenizer.ggml.merges` count | `63,683` | `63,683` | `63,683` | `293,320` |
| `tokenizer.ggml.add_bos_token` | `True` | `True` | `True` | `False` |
| `bos` / `eos` / `pad` token IDs | `1` / `7` / `0` | `1` / `7` / `0` | `1` / `7` / `0` | `124894` / `124900` / `124893` |
| **FFN Weights** | `62.98 MiB` (`43.7%`) | `130.84 MiB` (`60.5%`) | `465.12 MiB` (`66.9%`) | `654.09 MiB` (`62.8%`) |
| **Tied `token_embd` / LM Head** | `52.50 MiB` (`36.5%`) | `52.50 MiB` (`24.3%`) | `105.00 MiB` (`15.1%`) | `205.08 MiB` (`19.7%`) |
| **ShortConv Weights** | `18.09 MiB` (`12.6%`) | `22.62 MiB` (`10.5%`) | `90.23 MiB` (`13.0%`) | `117.71 MiB` (`11.3%`) |
| **Attention Weights** | `10.44 MiB` (`7.2%`) | `10.45 MiB` (`4.8%`) | `34.39 MiB` (`5.0%`) | `63.56 MiB` (`6.1%`) |
| **Norms (`F32`)** | `< 0.01 MiB` | `< 0.01 MiB` | `0.01 MiB` | `0.38 MiB` |

*Note: `LFM2.5-350M` and `LFM2.5-1.2B` share the exact same 16-layer hybrid topology (10 ShortConv + 6 Full GQA).*

---

### 3.2 Layer-by-Layer Routing Patterns

* **`LFM2.5-230M` (14 layers: 8 ShortConv + 6 Full GQA):**
  `head_count_kv = [0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0]`
* **`LFM2.5-350M` & `LFM2.5-1.2B` (16 layers: 10 ShortConv + 6 Full GQA):**
  `head_count_kv = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]`
* **`LFM3-1.7B` (24 layers: 13 ShortConv + 5 Sliding-Window GQA + 6 Global GQA):**
  - `head_count_kv = [0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]`
  - `sliding_window_pattern = [F, F, F, F, T, F, F, F, T, F, F, F, T, F, F, F, T, F, F, F, T, F, F, F]`

| Layer Index | `LFM2.5-1.2B` Block | `LFM3-1.7B` Block | LFM3 RoPE Profile |
| :---: | :--- | :--- | :--- |
| `0, 1` | `SHORTCONV` (`k=3`) | `SHORTCONV` (`k=7`) | — |
| `2` | `GQA_FULL` (`kv=8, d=64`) | **`GQA_FULL`** (`kv=4, d=128`) | `theta=1e6`, 50% partial (`64/128` dims) |
| `3` | `SHORTCONV` (`k=3`) | `SHORTCONV` (`k=7`) | — |
| `4` | `SHORTCONV` (`k=3`) | **`GQA_SWA`** (`kv=4, d=128, w=256`) | `theta=10k`, full `128` dims |
| `5` | `GQA_FULL` | `SHORTCONV` (`k=7`) | — |
| `6` | `SHORTCONV` | **`GQA_FULL`** | `theta=1e6`, 50% partial |
| `7` | `SHORTCONV` | `SHORTCONV` (`k=7`) | — |
| `8` | `GQA_FULL` | **`GQA_SWA`** (`w=256`) | `theta=10k`, full `128` dims |
| `9` | `SHORTCONV` | `SHORTCONV` (`k=7`) | — |
| `10` | `GQA_FULL` | **`GQA_FULL`** | `theta=1e6`, 50% partial |
| `11` | `SHORTCONV` | `SHORTCONV` (`k=7`) | — |
| `12` | `GQA_FULL` | **`GQA_SWA`** (`w=256`) | `theta=10k`, full `128` dims |
| `13` | `SHORTCONV` | `SHORTCONV` (`k=7`) | — |
| `14` | `GQA_FULL` | **`GQA_FULL`** | `theta=1e6`, 50% partial |
| `15` | `SHORTCONV` | `SHORTCONV` (`k=7`) | — |
| `16` | — | **`GQA_SWA`** (`w=256`) | `theta=10k`, full `128` dims |
| `17` | — | `SHORTCONV` (`k=7`) | — |
| `18` | — | **`GQA_FULL`** | `theta=1e6`, 50% partial |
| `19` | — | `SHORTCONV` (`k=7`) | — |
| `20` | — | **`GQA_SWA`** (`w=256`) | `theta=10k`, full `128` dims |
| `21` | — | `SHORTCONV` (`k=7`) | — |
| `22` | — | **`GQA_FULL`** | `theta=1e6`, 50% partial |
| `23` | — | `SHORTCONV` (`k=7`) | — |

---

### 3.3 Complete Tensor & Quantization Inventory

#### Global Tensors
| Tensor Name | `LFM2.5-1.2B-Instruct-Q4_K_M` | `LFM3-1.7B-Q4_K_M` | Notes |
| :--- | :--- | :--- | :--- |
| `token_embd.weight` | `[2048, 65536]` `Q6_K` | `[2048, 128000]` `Q6_K` | Tied as `output.weight` in both |
| Final RMSNorm | `token_embd_norm.weight` `[2048]` `F32` | `output_norm.weight` `[2048]` `F32` | `lfm2_loader.c` already checks both |
| `rope_freqs.weight` | *Absent* | `[64]` `F32` | Partial RoPE frequency divisors (`[0..31]=1.0`, `[32..63]=1e30`) |

#### Per-Layer Tensors (`blk.i.*`)
| Tensor Suffix | `LFM2.5-1.2B` (`Q4_K_M`) | `LFM3-1.7B` (`Q4_K_M`) |
| :--- | :--- | :--- |
| `attn_norm.weight` | `[2048]` `F32` (all 16) | `[2048]` `F32` (all 24) |
| `post_attention_norm.weight` | *Absent* | **`[2048]` `F32` (all 24)** |
| `ffn_norm.weight` | `[2048]` `F32` (all 16) | `[2048]` `F32` (all 24) |
| `post_ffw_norm.weight` | *Absent* | **`[2048]` `F32` (all 24)** |
| `shortconv.in_proj.weight` | `[2048, 6144]` `Q4_K` (10 layers) | `[2048, 6144]` `Q4_K` (13 layers) |
| `shortconv.conv.weight` | `[3, 2048]` `F32` (10 layers) | **`[7, 2048]` `F32` (13 layers)** |
| `shortconv.out_proj.weight` | `[2048, 2048]` `Q4_K` (10 layers) | `[2048, 2048]` `Q4_K` (13 layers) |
| `attn_q.weight` | `[2048, 2048]` `Q4_K` (6 layers) | `[2048, 2048]` `Q4_K` (11 layers) |
| `attn_k.weight` | `[2048, 512]` `Q4_K` (6 layers) | `[2048, 512]` `Q4_K` (11 layers) |
| `attn_v.weight` | `[2048, 512]` `Q4_K` (4) / `Q6_K` (2) | `[2048, 512]` `Q4_K` (6) / `Q6_K` (5) |
| `attn_q_norm.weight` | `[64]` `F32` (6 layers) | **`[128]` `F32` (11 layers)** |
| `attn_k_norm.weight` | `[64]` `F32` (6 layers) | **`[128]` `F32` (11 layers)** |
| `attn_gate.weight` | *Absent* | **`[2048, 16]` `Q4_K` (11 layers)** |
| `attn_output.weight` | `[2048, 2048]` `Q4_K` (6 layers) | `[2048, 2048]` `Q4_K` (11 layers) |
| `ffn_gate.weight` | `[2048, 8192]` `Q4_K` (16 layers) | `[2048, 7680]` `Q4_K` (24 layers) |
| `ffn_up.weight` | `[2048, 8192]` `Q4_K` (16 layers) | `[2048, 7680]` `Q4_K` (24 layers) |
| `ffn_down.weight` | `[8192, 2048]` `Q4_K` (8) / `Q6_K` (8) | `[7680, 2048]` `Q4_K` (12) / `Q6_K` (12) |

*(Total `LFM3-1.7B-Q4_K_M` quant counts across 287 tensors: `133` `F32`, `136` `Q4_K`, `18` `Q6_K`.)*

---

## 4. Detailed Operator, State & Tokenizer Differences

### 4.1 ShortConv: 3-Tap (`l_cache = 3`) vs. 7-Tap (`l_cache = 7`)

Both LFM2/2.5 and LFM3 use Double-Gated LIV ShortConv (`in_proj` splits `[B, C, x_tilde]`, Gate 1 computes $u_t = B \odot \tilde{x}_t$, causal depthwise Conv1D across $K = l\_cache$ taps, Gate 2 computes $y_t = C \odot \text{ConvOut}_t$, followed by `out_proj`), **but** $K$ increases from `3` to `7`:

* **State history length (`hist = l_cache - 1`):**
  - LFM2.5-1.2B: `hist = 2` tokens $\rightarrow$ `2 * 2048 * 4 = 16 KB/layer` (`160 KB` across 10 layers).
  - LFM3-1.7B: `hist = 6` tokens $\rightarrow$ `6 * 2048 * 4 = 48 KB/layer` (`624 KB` across 13 layers).
* **Convolution equation at step $t$ ($K = 7$):**
  $$\text{ConvOut}[c] = \left(\sum_{k=0}^{5} w[k][c] \cdot s_{t-6+k}[c]\right) + w[6][c] \cdot u_t[c]$$
* **Why `memmove` hurts at $K=7$:** Shifting 2 vectors (`s0 <- s1, s1 <- u_t`) costs almost nothing in LFM2, whereas shifting `5 * 2048 * 4 = 40 KB` via `memmove` across 13 layers moves **`~520 KB/token`** unnecessarily. Using a **6-slot circular ring index** eliminates 100% of state shifting.
* **Speculative Rollback (`B = 8`):** `conv_trace[8][13]` grows from `1.28 MB` (LFM2.5-1.2B) to `4.99 MB` (`8 * 13 * 48 KB`), still trivial (`~48 KB/layer` `memcpy` on draft rejection).

---

### 4.2 Attention: Dual-Hybrid Full GQA vs. Tri-Hybrid SWA + Global GQA (`head_dim = 128`)

| Property | `LFM2.5-1.2B` | `LFM3-1.7B` |
| :--- | :--- | :--- |
| Attention Layers | `6` Global GQA | `5` SWA (`w=256`) + `6` Global GQA |
| `H_q` / `H_kv` / `group_ratio` | `32` / `8` / `4` | `16` / `4` / `4` |
| `head_dim` | `64` | `128` |
| `kv_dim` (`H_kv * head_dim`) | `512` | `512` (identical!) |
| Global FP16 KV-Cache / Token | $6 \times 2 \times 512 \times 2\text{ B} = \mathbf{12\text{ KB/tok}}$ | $6 \times 2 \times 512 \times 2\text{ B} = \mathbf{12\text{ KB/tok}}$ |
| SWA FP16 KV-Cache (`w = 256`) | `0` | Constant $5 \times 2 \times 256 \times 512 \times 2\text{ B} = \mathbf{2.5\text{ MB}}$ total ($O(1)$) |

Notice the key architectural win of LFM3: even though `LFM3-1.7B` grows to 24 layers and `256K` max context, its **dynamic KV growth rate stays at `12 KB/token`** (identical to `LFM2.5-1.2B`) because:
1. Only `6` layers are global attention (`[2, 6, 10, 14, 18, 22]`).
2. `H_kv = 4` with `head_dim = 128` yields the exact same `kv_dim = 512` as LFM2.5's `H_kv = 8` with `head_dim = 64`.
3. The `5` SWA layers (`[4, 8, 12, 16, 20]`) have a capped `256`-token window, consuming a constant **`2.5 MB`** in ring-buffer mode (instead of `2.56 GB` across 5 layers if naively allocated at 256K context!).

---

### 4.3 Head-Wise Sigmoid Attention Gate (`attn_gate.weight`)

On every attention layer (`11` layers), `LFM3-1.7B-Q4_K_M` includes `blk.i.attn_gate.weight` with shape `[2048, 16]` (`ne[0] = 2048` input dim, `ne[1] = 16 = n_head_q` output heads):

* Each attention head $h \in [0, 15]$ gets a scalar gate logit $g_h = \mathbf{W}_{\text{gate}}[h] \cdot \hat{\mathbf{x}}$, squashed via $\sigma(g_h) = 1 / (1 + e^{-g_h})$.
* The head's attention output vector $\mathbf{o}_h \in \mathbb{R}^{128}$ is scaled by $\sigma(g_h)$ prior to `attn_output.weight` (`wo`), enabling heads to dynamically suppress outlier activations and preserve 4-bit quantization accuracy.
* **Size:** `16` rows × `8` `Q4_K` blocks (`1,152 B/row`) = **`18.4 KB/layer`** (`202 KB` total across all 11 attention layers).

---

### 4.4 Sandwich Normalization (`post_attention_norm`, `post_ffw_norm`)

Unlike LFM2/LFM2.5 (pre-norm only), **all 24 layers** of `LFM3-1.7B` carry four `F32` RMSNorm vectors of dimension `2048`:
1. `blk.i.attn_norm.weight` (pre-operator norm)
2. `blk.i.post_attention_norm.weight` (post-operator norm)
3. `blk.i.ffn_norm.weight` (pre-FFN norm)
4. `blk.i.post_ffw_norm.weight` (post-FFN norm)

> [!IMPORTANT]
> **Implementation Verification Check:** Before finalizing the LFM3 forward pass, verify the exact placement of `post_attention_norm` and `post_ffw_norm` relative to the residual addition against the Hugging Face reference (`modeling_lfm3.py` / `llama.cpp` `build_lfm3`)—specifically confirming `x = x + RMSNorm_post(op_out)` (Gemma-2/sandwich style) vs. post-residual normalization—via Tier 5 golden logit parity in [`tests/verify_parity.py`](../tests/verify_parity.py).

---

### 4.5 Hybrid & Partial RoPE (`rope_freqs.weight` & `freq_base_swa`)

LFM3 splits RoPE configuration by attention layer type:
* **Sliding-Window Attention (`[4, 8, 12, 16, 20]`):** Uses `lfm3.rope.freq_base_swa = 10000.0` across the full `head_dim = 128` (`64` pairs).
* **Global Attention (`[2, 6, 10, 14, 18, 22]`):** Uses `lfm3.rope.freq_base = 1000000.0` together with global tensor `rope_freqs.weight` (`[64]` `F32`):
  - Measured values in `LFM3-1.7B-Q4_K_M.gguf`:
    - `rope_freqs.weight[0..31]  = 1.0f` (rotate pairs `0..31`, i.e., **first 64 dimensions**)
    - `rope_freqs.weight[32..63] = 1e30f` (in GGML RoPE, frequency $\theta_i / 10^{30} \approx 0 \Rightarrow \cos(0)=1, \sin(0)=0$, leaving **upper 64 dimensions unrotated** — `partial_rotary_factor = 0.5`).
  - Also verify NeoX half-split pairing (`(i, i + rot_dim/2)` vs `(i, i + head_dim/2)`) against `modeling_lfm3.py` / `llama.cpp` `build_lfm3` during parity verification.

---

### 4.6 Vocabulary, Tokenizer, Special Tokens & ChatML Template

* **Vocab & Merges:** `128,000` tokens (nearly 2× LFM2.5's `65,536`) and `293,320` BPE merges (4.6× LFM2.5's `63,683`), with `tokenizer.ggml.model = gpt2` and `tokenizer.ggml.pre = lfm2`.
* **BOS Behavior (`add_bos_token = False`):** LFM2.5 sets `add_bos_token = True` (`BOS = 1`), whereas `LFM3-1.7B` sets **`add_bos_token = False`** (`BOS = 124894`, `EOS = 124900`, `PAD = 124893`) while its `tokenizer.chat_template` explicitly prepends `{{- bos_token -}}` at the start of the conversation.
* **Non-Normal / Special Tokens (`3,107` tokens):**
  - `124893..124900`: `<|pad|>`, `<|startoftext|>`, `<|endoftext|>`, `<|fim_pre|>`, `<|fim_mid|>`, `<|fim_suf|>`, `<|im_start|>`, `<|im_end|>`
  - `124901`, `124902`: `<think>`, `</think>` (`token_type = 4`, without `<|...|>` brackets)
  - `124903..124906`: `<|tool_list_start|>`, `<|tool_list_end|>`, `<|tool_call_start|>`, `<|tool_call_end|>`
  - `124907+`: `<image>` and ~3,080 `<|img_row_*_col_*|>` vision grid tokens.
* **Thinking Template Default:** LFM3's `chat_template` appends `<|im_start|>assistant\n<think>` when `enable_thinking` is true (default), or `<|im_start|>assistant\n<think></think>` when disabled, and strips earlier `<think>...</think>` turns unless `preserve_thinking=true`.

---

### 4.7 Multimodal Extensions (`LFM3-VL`, `LFM3-Audio`, `LFM3-Omni`)

LFM3's multimodal siblings share the exact same `lfm3` text backbone plus external projector GGUFs (`mmproj-*.gguf`) containing vision (`v.patch_embd.*`) and audio/codec (`a.conv1d.*`, `a.gen.*`) weights. Supporting the `lfm3` text backbone also lays the foundation for feeding vision/audio embeddings via LAII's existing [`lfm2_forward_embeddings`](../include/lfm2.h) hook.

---

## 5. LAII Codebase Compatibility Audit

Audit of existing LAII files when loading or executing `models/LFM3-1.7B-Q4_K_M.gguf`:

| Component & Location | Ready for LFM3? | Observed Code Behavior / Exact Blocker |
| :--- | :---: | :--- |
| **Suffix-based GGUF Metadata** [`tools/lfm2_loader.c`](../tools/lfm2_loader.c#L308-L357) | **Partial** | Parses `.block_count`, `.embedding_length`, `.feed_forward_length`, `.shortconv.l_cache`, `.attention.head_count_kv[]`, `.rope.freq_base`. **Missing:** `.attention.sliding_window`, `.attention.sliding_window_pattern`, `.rope.freq_base_swa`, `.attention.key_length`, `.rope.dimension_count`, `rope_freqs.weight`, `attn_gate.weight`, `post_attention_norm.weight`, `post_ffw_norm.weight`. |
| **Gen-1 Model Loader** [`src/lfm_model.c`](../src/lfm_model.c#L35-L120) | **No** | Hardcodes `"lfm2.*"` metadata keys (e.g. `lfm2.block_count`, `lfm2.vocab_size`) and classifies any `layer_types[i] != "full_attention"` as ShortConv. |
| **MatVec / GEMM / SwiGLU Kernels** [`tools/lfm2_decode_1x4.c`](../tools/lfm2_decode_1x4.c), [`tools/lfm2_cpuid_quants.c`](../tools/lfm2_cpuid_quants.c) | **Yes** | `n_embd = 2048` (`8 × 256`) and `n_ff = 7680` (`30 × 256`) are exact multiples of `256`. All `Q4_K`/`Q6_K` `1x4` MatVec, batched GEMM, and fused SwiGLU kernels apply directly without tail overhead. |
| **ShortConv Fast Path** [`tools/lfm2_kernels_fast.c`](../tools/lfm2_kernels_fast.c#L100-L201), [`tools/lfm2_forward.c`](../tools/lfm2_forward.c#L582) | **Fallback only** | `LFM2PlanarConvWeights` and `conv_w[l][3]` are hardcoded for `l_cache == 3`. `l_cache = 7` falls back to scalar loops and `memmove` of `40 KB/layer/token`. |
| **Fast Context Init & Online Softmax** [`tools/lfm2_kernels_fast.c`](../tools/lfm2_kernels_fast.c#L1090-L1093) | **No** | Rejects initialization unless `head_dim == 64 && (D % 256) == 0`. Online softmax workers hardcode 64-dim heads. |
| **Public C11 API RoPE & Sink Shift** [`src/lfm2_api.c`](../src/lfm2_api.c#L551), [`src/lfm2_api.c`](../src/lfm2_api.c#L825) | **No** | Hardcodes `lfm2_rope_lut_init(..., 64, freq_base)` and skips attention-sink KV shifting when `head_dim != 64`. |
| **OOM Guard & Speculative State** [`tools/lfm2_sys_cache.c`](../tools/lfm2_sys_cache.c#L161), [`tools/lfm2_speculative_pld.c`](../tools/lfm2_speculative_pld.c#L236) | **Partial** | Correctly computes `hist_len = shortconv_l_cache - 1` dynamically (`6`), **but** treats all attention layers as full `max_seq_len` KV-caches rather than capping SWA layers to `sliding_window = 256`. |
| **BPE Tokenizer** [`tools/lfm2_bpe.c`](../tools/lfm2_bpe.c#L80), [`tools/lfm2_bpe.c`](../tools/lfm2_bpe.c#L288-L309) | **Partial** | Handles `128K` vocab & `293K` merges via power-of-2 hash tables, **but** `specials[64]` caps special tokens at 64 (LFM3 has `3,107` non-normal tokens), `<think>`/`</think>` lack `<|...|>` delimiters, and `add_bos_token` metadata isn't parsed. |

---

## 6. LFM3 Optimization Blueprint (Zero Regression on LFM2/LFM2.5)

> [!TIP]
> **Zero-Regression Invariant:** Keep all LFM2/LFM2.5 fast paths (`l_cache == 3` 3-tap planar ShortConv, `head_dim == 64` online softmax, pre-norm-only layer loop) untouched and dispatch at **model load / layer block-type** granularity—never add extra conditional branches inside inner SIMD loops.

1. **Amortize the `215 MB` LM-Head Sweep (`[2048, 128000]` `Q6_K`)**
   - At `128,000` rows × `1,680 B/row` (`Q6_K`), `output.weight` streams **`~205 MiB` (`215 MB`) per decode step** (~19.7% of total model size, nearly **2×** LFM2.5-1.2B's `105 MiB`).
   - Prompt-Lookup Speculative Verification (`B = 8` batched GEMM) gives even larger speedups on LFM3 because one `Q6_K` weight sweep verifies up to 8 tokens at once.
   - Keep `compute_last_logits = false` during chunked prefill and retain 4-row unrolled `Q6_K` + `_MM_HINT_NTA` non-temporal streaming across the 128K rows.

2. **7-Tap Planar ShortConv + Circular Ring Buffer (`13` layers)**
   - Pre-transpose `[7, 2048]` weights into 7 64-byte-aligned planar vectors `w0..w6[2048]`.
   - Replace the `40 KB` `memmove` per layer per token with a **circular ring cursor** (`ring_idx = (ring_idx + 1) % 6`) over the `48 KB/layer` rolling state (`624 KB` total across 13 layers — L2-cache resident).
   - Add a dedicated AVX2/NEON 7-tap fused pass (`Gate 1 -> 7-tap FMA -> Gate 2 -> L1-hot Q8_K`) alongside the existing 3-tap LFM2 fast path.

3. **Constant `2.5 MB` Ring-Buffered FP16 KV-Cache for SWA Layers**
   - Add `LFM2_BLOCK_SWA_ATTN` (`sliding_window_pattern[l] == true`).
   - Allocate only `min(max_seq_len, 256)` FP16 KV slots per SWA layer (`512 KB/layer`, **`2.5 MB` total** across all 5 SWA layers) and index via `pos & 255` (power-of-2 bitmask since `sliding_window = 256`!), bounding attention loops to at most 256 steps even at 256K context.

4. **Register-Pinned `head_dim = 128` Online Softmax & 50% Partial RoPE**
   - Add a dedicated `head_dim = 128` FP16 online-softmax worker:
     - **x86_64 AVX2:** Keep all `16` `ymm` accumulators (`vo0..vo15` = 128 floats) in registers across the KV pass.
     - **ARM64 NEON:** Process `128` dims as two `64`-dim half-tiles (`16` `float32x4_t` accumulators per tile) to preserve zero register spills.
   - Precompute two RoPE LUTs at context creation:
     - `rope_lut_swa` (`freq_base_swa = 10000.0`, full `128` dims).
     - `rope_lut_full` (`freq_base = 1000000.0`, rotating only the active `64` dims where `rope_freqs.weight[i] < 1e10f`, skipping arithmetic on the unrotated half).

5. **Fused `attn_gate` & Fused Sandwich Norms**
   - Compute `attn_gate` (`16` rows, `18 KB` `Q4_K`) in the **same `Q8_K` barrier/pass** as `Q`, `K`, `V` (reusing L1-hot `xq_scratch`), then multiply each head's 128-dim output by `sigmoid(gate[h])` before `attn_output.weight`.
   - Fuse `post_attention_norm + residual add + ffn_norm + Q8_K quantize` (and `post_ffw_norm + residual add + next attn_norm + Q8_K quantize`) in L1 cache so the 48 extra `2048`-float norms incur zero extra DRAM traffic.

6. **Dynamic Special-Token Indexing & `add_bos_token` Handling**
   - Replace fixed `specials[64]` scanning with dynamic storage + hash/prefix indexing covering `<|...|>` tokens, `<think>`, and `</think>` (`token_type` 3/4) while honoring `tokenizer.ggml.add_bos_token`.

---

## 7. Verification & Parity Certification Protocol

1. **Inspect LFM3 GGUF Metadata & Tensors (once `lfm3` loader updates are added):**
   ```bash
   ./bin/lfm2_loader models/LFM3-1.7B-Q4_K_M.gguf
   ```
   Verify all 24 layers map cleanly (`13` `SHORTCONV`, `5` `GQA_SWA`, `6` `GQA_FULL`), `ShortConv L-Cache = 7`, `head_dim = 128`, `sliding_window = 256`, `attn_gate`, `post_attention_norm`, `post_ffw_norm`, and `rope_freqs`.

2. **Certify LFM3 Numerical Parity Against Golden Oracle:**
   ```bash
   python3 tests/verify_parity.py \
       -m models/LFM3-1.7B-Q4_K_M.gguf \
       --lib ./lib/liblfm2.so \
       --hf-model LiquidAI/LFM3-1.7B \
       -q Q4_K_M
   ```
   Must pass all 5 tiers: exact BPE token IDs, batched prefill parity, bit-exact checkpoint rollback, L2 norm embeddings, and `Q4_K_M` logit SLA (`Cosine >= 0.99300`, `KL <= 0.0650`, `Top-5 >= 4/5`).

3. **Certify Zero Regression on LFM2.5 & Memory Safety:**
   ```bash
   make test
   make test-asan
   python3 tests/verify_parity.py -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf --lib ./lib/liblfm2.so -q Q4_K_M
   ```
