#!/usr/bin/env python3
"""
Creates a minimal valid GGUF v3 model file for testing LFM2.
Architecture: 2-layer hybrid (1 LIV Conv layer + 1 GQA Attention layer)
"""

import struct
import numpy as np

def write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)

def write_kv_u32(f, key: str, val: int):
    write_string(f, key)
    f.write(struct.pack("<I", 4)) # GGUF_TYPE_UINT32
    f.write(struct.pack("<I", val))

def write_kv_f32(f, key: str, val: float):
    write_string(f, key)
    f.write(struct.pack("<I", 6)) # GGUF_TYPE_FLOAT32
    f.write(struct.pack("<f", val))

def write_kv_string(f, key: str, val: str):
    write_string(f, key)
    f.write(struct.pack("<I", 8)) # GGUF_TYPE_STRING
    write_string(f, val)

def write_kv_string_array(f, key: str, strings: list):
    write_string(f, key)
    f.write(struct.pack("<I", 9)) # GGUF_TYPE_ARRAY
    f.write(struct.pack("<I", 8)) # item type: STRING
    f.write(struct.pack("<Q", len(strings)))
    for s in strings:
        write_string(f, s)

def write_kv_u32_array(f, key: str, values: list):
    write_string(f, key)
    f.write(struct.pack("<I", 9)) # GGUF_TYPE_ARRAY
    f.write(struct.pack("<I", 4)) # item type: UINT32
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        f.write(struct.pack("<I", v))


def main():
    out_path = "tests/lfm2_toy.gguf"
    np.random.seed(42)

    vocab_size = 32
    hidden_dim = 64
    ffn_dim = 128
    n_layers = 2
    n_heads_q = 4
    n_heads_kv = 2
    head_dim = 16
    conv_L_cache = 3

    # Define tokens
    tokens = [
        "<unk>", "<s>", "</s>", "Liquid", " Foundation", " Models",
        " are", " fast", " on", " edge", " CPUs", " because",
        " of", " hybrid", " LIV", " short", " convolutions", " and",
        " reduced", " KV", " cache", " memory", " bandwidth", " zero",
        " latency", " Linux", " engine", " C", " AVX2", " offline",
        ".", "!"
    ]

    # Prepare tensors dictionary
    tensors = {}

    def add_tensor(name, shape):
        # shape is (dim0, dim1, ...)
        data = (np.random.randn(*shape) * 0.05).astype(np.float32)
        tensors[name] = data

    add_tensor("token_embd.weight", (vocab_size, hidden_dim))
    add_tensor("token_embd_norm.weight", (hidden_dim,))
    add_tensor("output.weight", (vocab_size, hidden_dim))

    # Layer 0: LIV Conv
    add_tensor("blk.0.attn_norm.weight", (hidden_dim,))
    add_tensor("blk.0.shortconv.in_proj.weight", (3 * hidden_dim, hidden_dim))
    add_tensor("blk.0.shortconv.conv.weight", (hidden_dim, conv_L_cache))
    add_tensor("blk.0.shortconv.out_proj.weight", (hidden_dim, hidden_dim))
    add_tensor("blk.0.ffn_norm.weight", (hidden_dim,))
    add_tensor("blk.0.ffn_gate.weight", (ffn_dim, hidden_dim))
    add_tensor("blk.0.ffn_up.weight", (ffn_dim, hidden_dim))
    add_tensor("blk.0.ffn_down.weight", (hidden_dim, ffn_dim))

    # Layer 1: Attention
    add_tensor("blk.1.attn_norm.weight", (hidden_dim,))
    add_tensor("blk.1.attn_q.weight", (n_heads_q * head_dim, hidden_dim))
    add_tensor("blk.1.attn_k.weight", (n_heads_kv * head_dim, hidden_dim))
    add_tensor("blk.1.attn_v.weight", (n_heads_kv * head_dim, hidden_dim))
    add_tensor("blk.1.attn_q_norm.weight", (head_dim,))
    add_tensor("blk.1.attn_k_norm.weight", (head_dim,))
    add_tensor("blk.1.attn_output.weight", (hidden_dim, hidden_dim))
    add_tensor("blk.1.ffn_norm.weight", (hidden_dim,))
    add_tensor("blk.1.ffn_gate.weight", (ffn_dim, hidden_dim))
    add_tensor("blk.1.ffn_up.weight", (ffn_dim, hidden_dim))
    add_tensor("blk.1.ffn_down.weight", (hidden_dim, ffn_dim))

    with open(out_path, "wb") as f:
        # Header
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3)) # version 3
        f.write(struct.pack("<Q", len(tensors)))
        
        # We will write 13 metadata KV pairs
        n_kv = 13
        f.write(struct.pack("<Q", n_kv))

        # Metadata KVs
        write_kv_string(f, "general.architecture", "lfm2")
        write_kv_u32(f, "general.alignment", 32)
        write_kv_u32(f, "lfm2.block_count", n_layers)
        write_kv_u32(f, "lfm2.embedding_length", hidden_dim)
        write_kv_u32(f, "lfm2.feed_forward_length", ffn_dim)
        write_kv_u32(f, "lfm2.vocab_size", vocab_size)
        write_kv_u32(f, "lfm2.context_length", 128)
        write_kv_u32(f, "lfm2.attention.head_count", n_heads_q)
        write_kv_u32_array(f, "lfm2.attention.head_count_kv", [0, n_heads_kv])
        write_kv_u32(f, "lfm2.shortconv.l_cache", conv_L_cache)
        write_kv_f32(f, "lfm2.attention.layer_norm_rms_epsilon", 1e-5)
        write_kv_f32(f, "lfm2.rope.freq_base", 1000000.0)
        write_kv_string_array(f, "tokenizer.ggml.tokens", tokens)

        # Precompute tensor data offsets
        # In GGUF, dimensions in tensor info are written in reverse (dim0, dim1...)
        tensor_bytes = {}
        total_data_offset = 0
        alignment = 32

        # Write Tensor Infos
        current_offset = 0
        for name, data in tensors.items():
            write_string(f, name)
            n_dims = len(data.shape)
            f.write(struct.pack("<I", n_dims))
            # GGUF tensor dimensions order: innermost dimension first
            for d in reversed(data.shape):
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", 0)) # GGML_TYPE_F32 = 0
            f.write(struct.pack("<Q", current_offset))

            data_bytes = data.tobytes()
            tensor_bytes[name] = data_bytes
            size = len(data_bytes)
            # Align next offset
            aligned_size = (size + alignment - 1) & ~(alignment - 1)
            current_offset += aligned_size

        # Align to data segment
        cur_pos = f.tell()
        pad = (alignment - (cur_pos % alignment)) % alignment
        f.write(b"\x00" * pad)

        # Write Tensor Data
        for name, data_bytes in tensor_bytes.items():
            f.write(data_bytes)
            size = len(data_bytes)
            pad = (alignment - (size % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)

    print(f"Generated toy GGUF model: {out_path} ({len(tensors)} tensors)")

if __name__ == "__main__":
    main()
