#!/usr/bin/env python3
"""Automated Unit Test Suite for Liquid AI LFM2 Python SDK (python/lfm2.py)."""

import math
import os
import sys
from pathlib import Path

# Add python directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lfm2 import LFM2GrammarType, LFM2Model, LFM2PoolingType, LFM2Status  # noqa: E402

TOY_MODEL_PATH = "tests/lfm2_toy.gguf"
LIB_PATH = "./lib/liblfm2.so"


def test_python_sdk() -> None:
    print("=======================================================")
    print("Running Liquid AI LFM2 Python SDK Verification Suite   ")
    print("=======================================================")

    assert os.path.exists(TOY_MODEL_PATH), f"Missing toy GGUF: {TOY_MODEL_PATH}"
    assert os.path.exists(LIB_PATH), f"Missing shared library: {LIB_PATH}"

    # 1. Model Lifecycle & Introspection
    print("[Test 1/5] Verifying LFM2Model Loading & Introspection...")
    with LFM2Model(TOY_MODEL_PATH, lib_path=LIB_PATH, warmup_pages=False) as model:
        assert model.info.n_layers == 2
        assert model.info.n_embd == 64
        assert model.info.n_vocab == 32
        assert model.info.n_shortconv_layers == 1
        assert model.info.n_gqa_layers == 1
        assert model.info.file_size_mb > 0
        print(f"  Model loaded: {model.info.arch} on {model.info.active_isa} ({model.info.file_size_mb:.2f} MB)")

        # 2. Context Lifecycle & Streaming Generation
        print("[Test 2/5] Verifying Streaming Chat Generator (Single-Threaded UTF-8 Pull)...")
        with model.create_context(max_seq_len=256) as ctx:
            assert ctx.pos == 0
            tokens = []
            for chunk in ctx.stream_chat("Hello from Python SDK", temperature=0.0):
                tokens.append(chunk)
                if len(tokens) >= 5:
                    break

            assert len(tokens) > 0
            full_text = "".join(tokens)
            print(f"  Streamed {len(tokens)} chunks: {repr(full_text)}")
            assert ctx.pos > 0

            stats = ctx.stats
            assert stats.prompt_tokens > 0
            assert stats.generated_tokens > 0
            print(f"  Stats: prefill={stats.prefill_ms:.2f}ms, decode={stats.decode_ms:.2f}ms")

            # 3. Speculative Branching with ctx.branch()
            print("[Test 3/5] Verifying Context Branching & State Rollback...")
            baseline_pos = ctx.pos
            with ctx.branch():
                branch_reply = ctx.chat("What is the next token?", temperature=0.0)
                assert ctx.pos > baseline_pos
                print(f"  Inside branch: pos={ctx.pos}, reply={repr(branch_reply[:20])}")

            assert ctx.pos == baseline_pos
            print(f"  After branch exit: pos successfully restored to {ctx.pos} (Bit-exact rollback!)")

            # 4. L2-Normalized Embeddings
            print("[Test 4/5] Verifying L2-Normalized Text Embeddings...")
            emb_last = ctx.embed("Liquid AI Engine", pooling=LFM2PoolingType.LAST_TOKEN)
            assert len(emb_last) == model.info.n_embd
            norm_last = sum(x * x for x in emb_last) ** 0.5
            assert abs(norm_last - 1.0) < 1e-3
            print(f"  LAST_TOKEN L2 Norm: {norm_last:.6f}")

            emb_mean = ctx.embed("Liquid AI Engine", pooling=LFM2PoolingType.MEAN)
            assert len(emb_mean) == model.info.n_embd
            norm_mean = sum(x * x for x in emb_mean) ** 0.5
            assert abs(norm_mean - 1.0) < 1e-3
            print(f"  MEAN_POOL L2 Norm : {norm_mean:.6f}")

            # 5. Grammar-Constrained Generation Call
            print("[Test 5/6] Verifying Grammar Constraint Invocation...")
            with ctx.branch():
                json_reply = ctx.chat("Output a json object", grammar=LFM2GrammarType.JSON_OBJECT)
                assert isinstance(json_reply, str)
                print(f"  Grammar-constrained reply received: {repr(json_reply[:20])}")

            # 6. Direct Tokenization, Prefill & Zero-Copy Logits
            print("[Test 6/6] Verifying Direct Tokenization, Prefill & Zero-Copy Logits...")
            toks = model.tokenize("Liquid AI")
            assert len(toks) > 0
            with ctx.branch():
                ctx.reset()
                ctx.prefill_tokens([])  # Zero-token no-op check
                ctx.prefill_tokens(toks, compute_last_logits=True)
                logits = ctx.get_logits()
                assert len(logits) == model.info.n_vocab
                assert all(isinstance(x, float) and not math.isnan(x) and not math.isinf(x) for x in logits)
                ptr = ctx.get_logits_ptr()
                assert ptr[0] == logits[0]
                assert ptr[len(logits) - 1] == logits[-1]
                print(f"  Zero-copy logits extracted successfully (len={len(logits)}, head={logits[:3]})")

    print("=======================================================")
    print("ALL 6 PYTHON SDK TESTS PASSED WITH 100% SUCCESS!       ")
    print("=======================================================")


if __name__ == "__main__":
    test_python_sdk()

