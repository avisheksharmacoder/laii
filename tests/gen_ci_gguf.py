#!/usr/bin/env python3
"""
tests/gen_ci_gguf.py
Generates a miniature, mathematically valid 4-layer Hybrid LFM2 GGUF v3 model
(~1.6 MB, D=256, n_ff=512, 2 ShortConv + 2 GQA layers, Q4_K + Q6_K + F32)
for fast, air-gapped CI sanitizer (ASan/UBSan/TSan) and parity verification.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

GGUF_MAGIC   = 0x46554747  # "GGUF"
GGUF_VERSION = 3
ALIGNMENT    = 32

# GGUF Metadata Value Types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9

# GGML Tensor Types
GGML_TYPE_F32  = 0
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(key: str, val: str) -> bytes:
    return pack_str(key) + struct.pack("<I", GGUF_TYPE_STRING) + pack_str(val)


def kv_u32(key: str, val: int) -> bytes:
    return pack_str(key) + struct.pack("<II", GGUF_TYPE_UINT32, val)


def kv_f32(key: str, val: float) -> bytes:
    return pack_str(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, val)


def kv_u32_array(key: str, vals: list[int]) -> bytes:
    body = struct.pack(f"<{len(vals)}I", *vals)
    return pack_str(key) + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_UINT32, len(vals)) + body


def kv_str_array(key: str, vals: list[str]) -> bytes:
    body = b"".join(pack_str(v) for v in vals)
    return pack_str(key) + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING, len(vals)) + body


def make_q4_k_blocks(n_blocks: int, seed: int) -> bytes:
    # Each Q4_K block is 144 bytes (256 elements): uint16 d, uint16 dmin, scales[12], qs[128]
    out = bytearray()
    for b in range(n_blocks):
        d = 0x2800     # ~0.0078 in FP16
        dmin = 0x2000  # ~0.0020 in FP16
        scales = bytes(((seed + b * 3 + j * 7) & 0x1F) for j in range(12))
        qs = bytes(((seed + b * 13 + j * 29) & 0xFF) for j in range(128))
        out += struct.pack("<HH12s128s", d, dmin, scales, qs)
    return bytes(out)


def make_q6_k_blocks(n_blocks: int, seed: int) -> bytes:
    # Each Q6_K block is 210 bytes (256 elements): ql[128], qh[64], scales[16], uint16 d
    out = bytearray()
    for b in range(n_blocks):
        ql = bytes(((seed + b * 17 + j * 11) & 0xFF) for j in range(128))
        qh = bytes(((seed + b * 23 + j * 19) & 0xFF) for j in range(64))
        scales = bytes((((seed + b + j) % 31) - 15) & 0xFF for j in range(16))
        d = 0x2600
        out += struct.pack("<128s64s16sH", ql, qh, scales, d)
    return bytes(out)


def make_f32_vec(n_elems: int, val: float) -> bytes:
    return struct.pack(f"<{n_elems}f", *([val] * n_elems))


def gpt2_bytes_to_unicode() -> list[str]:
    """Exact GPT-2 byte encoder mapping (256 unique characters)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(n) for n in cs]


def generate_ci_gguf(out_path: Path) -> None:
    D = 256
    n_ff = 512
    n_layers = 4
    n_head_q = 4       # head_dim = 64
    kv_heads = [0, 2, 0, 2]  # Layers 0,2 = ShortConv; Layers 1,3 = GQA

    # Build 512-token vocabulary:
    # 0..7: Special control tokens
    # 8..263: 256 GPT-2 unicode byte symbols
    # 264..511: Subword tokens & common merges
    tokens: list[str] = [
        "<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unk|>",
        "<|tool_call_start|>", "<|tool_call_end|>", "<|im_start|>", "<|im_end|>",
    ]
    tokens.extend(gpt2_bytes_to_unicode())
    while len(tokens) < 512:
        idx = len(tokens)
        tokens.append(f"tok_{idx}")

    n_vocab = len(tokens)
    merges = ["t o", "to k", "i n", "e n"]

    kvs = [
        kv_str("general.architecture", "lfm2"),
        kv_u32("lfm2.block_count", n_layers),
        kv_u32("lfm2.embedding_length", D),
        kv_u32("lfm2.feed_forward_length", n_ff),
        kv_u32("lfm2.vocab_size", n_vocab),
        kv_u32("lfm2.context_length", 2048),
        kv_u32("lfm2.attention.head_count", n_head_q),
        kv_u32_array("lfm2.attention.head_count_kv", kv_heads),
        kv_u32("lfm2.shortconv.l_cache", 3),
        kv_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5),
        kv_f32("lfm2.rope.freq_base", 1000000.0),
        kv_str_array("tokenizer.ggml.tokens", tokens),
        kv_str_array("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.bos_token_id", 1),
        kv_u32("tokenizer.ggml.eos_token_id", 7),
    ]

    # Collect (name, shape, ggml_type, raw_bytes)
    tensors: list[tuple[str, list[int], int, bytes]] = []

    def add_tensor(name: str, shape: list[int], ttype: int, data: bytes) -> None:
        tensors.append((name, shape, ttype, data))

    add_tensor("token_embd.weight", [D, n_vocab], GGML_TYPE_Q4_K, make_q4_k_blocks(n_vocab * (D // 256), 1))
    add_tensor("output_norm.weight", [D], GGML_TYPE_F32, make_f32_vec(D, 1.0))
    add_tensor("output.weight", [D, n_vocab], GGML_TYPE_Q4_K, make_q4_k_blocks(n_vocab * (D // 256), 2))

    for l in range(n_layers):
        add_tensor(f"blk.{l}.attn_norm.weight", [D], GGML_TYPE_F32, make_f32_vec(D, 1.0))
        add_tensor(f"blk.{l}.ffn_norm.weight", [D], GGML_TYPE_F32, make_f32_vec(D, 1.0))
        add_tensor(f"blk.{l}.ffn_gate.weight", [D, n_ff], GGML_TYPE_Q4_K, make_q4_k_blocks(n_ff * (D // 256), 10 + l))
        add_tensor(f"blk.{l}.ffn_up.weight", [D, n_ff], GGML_TYPE_Q4_K, make_q4_k_blocks(n_ff * (D // 256), 20 + l))
        add_tensor(f"blk.{l}.ffn_down.weight", [n_ff, D], GGML_TYPE_Q6_K, make_q6_k_blocks(D * (n_ff // 256), 30 + l))

        if kv_heads[l] == 0:
            # ShortConv Layer
            add_tensor(f"blk.{l}.shortconv.in_proj.weight", [D, 3 * D], GGML_TYPE_Q4_K, make_q4_k_blocks(3 * D * (D // 256), 40 + l))
            conv_vals = [0.25 * math.cos(i * 0.1) for i in range(3 * D)]
            add_tensor(f"blk.{l}.shortconv.conv.weight", [3, D], GGML_TYPE_F32, struct.pack(f"<{3 * D}f", *conv_vals))
            add_tensor(f"blk.{l}.shortconv.out_proj.weight", [D, D], GGML_TYPE_Q4_K, make_q4_k_blocks(D * (D // 256), 50 + l))
        else:
            # GQA Layer (n_head_kv = 2 -> kv_dim = 128)
            kv_dim = kv_heads[l] * 64
            add_tensor(f"blk.{l}.attn_q.weight", [D, D], GGML_TYPE_Q4_K, make_q4_k_blocks(D * (D // 256), 60 + l))
            add_tensor(f"blk.{l}.attn_k.weight", [D, kv_dim], GGML_TYPE_Q4_K, make_q4_k_blocks(kv_dim * (D // 256), 70 + l))
            add_tensor(f"blk.{l}.attn_v.weight", [D, kv_dim], GGML_TYPE_Q4_K, make_q4_k_blocks(kv_dim * (D // 256), 80 + l))
            add_tensor(f"blk.{l}.attn_output.weight", [D, D], GGML_TYPE_Q4_K, make_q4_k_blocks(D * (D // 256), 90 + l))
            add_tensor(f"blk.{l}.attn_q_norm.weight", [64], GGML_TYPE_F32, make_f32_vec(64, 1.0))
            add_tensor(f"blk.{l}.attn_k_norm.weight", [64], GGML_TYPE_F32, make_f32_vec(64, 1.0))

    header = bytearray()
    header += struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(kvs))
    for kv in kvs:
        header += kv

    tensor_payload = bytearray()
    for name, shape, ttype, data in tensors:
        pad = (ALIGNMENT - (len(tensor_payload) % ALIGNMENT)) % ALIGNMENT
        tensor_payload += b"\x00" * pad
        rel_offset = len(tensor_payload)
        tensor_payload += data

        header += pack_str(name)
        header += struct.pack("<I", len(shape))
        for dim in shape:
            header += struct.pack("<Q", dim)
        header += struct.pack("<IQ", ttype, rel_offset)

    hdr_pad = (ALIGNMENT - (len(header) % ALIGNMENT)) % ALIGNMENT
    header += b"\x00" * hdr_pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + tensor_payload)
    print(f"[gen_ci_gguf] Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(tensors)} tensors)")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/ci_lfm2_tiny.gguf")
    generate_ci_gguf(target)

