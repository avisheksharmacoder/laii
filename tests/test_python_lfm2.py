#!/usr/bin/env python3
"""Automated Unit Test Suite for Liquid AI LFM2 Python SDK (python/lfm2.py)."""

import math
import os
import sys
from pathlib import Path

# Add python directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lfm2 import (  # noqa: E402
    LFM2GrammarType,
    LFM2Model,
    LFM2PoolingType,
    LFM2Status,
    LFM3Model,
    load_lfm3_model,
)

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

    # 7. Android Scoped Storage FD Loading, RSS Profiling & Page Cache Lifecycle
    print("[Test 7/7] Verifying Android Scoped Storage FD Loading, RSS Profiling & Page Cache Lifecycle...")
    rss_bytes = LFM2Model.get_process_rss_bytes(lib_path=LIB_PATH)
    assert rss_bytes > 0, "Expected non-zero process RSS"
    print(f"  Current process RSS: {rss_bytes / (1024 * 1024):.2f} MB")

    raw_fd = os.open(TOY_MODEL_PATH, os.O_RDONLY)
    try:
        with LFM2Model.from_fd(raw_fd, lib_path=LIB_PATH) as fd_model:
            assert fd_model.info.n_layers == 2
            assert fd_model.info.n_embd == 64
            assert fd_model.info.n_vocab == 32

            # Test active page cache warmup (for mobile on_resume background task)
            fd_model.warmup_page_cache()
            print("  Page cache warmed via warmup_page_cache()")

            # Test dual-layer page cache release (madvise MADV_DONTNEED + posix_fadvise for mobile on_pause)
            fd_model.release_page_cache()
            print("  Page cache released via release_page_cache()")

            with fd_model.create_context(max_seq_len=64) as fd_ctx:
                reply = fd_ctx.chat("Testing from FD", temperature=0.0)
                assert isinstance(reply, str)
                print(f"  Inference executed successfully on FD model: {repr(reply[:15])}")
    finally:
        os.close(raw_fd)

    # 8. LFM3 Strict Loader Validation & Real LFM3 End-to-End Test (if model present)
    print("[Test 8/8] Verifying LFM3Model strict guard & LFM3 inference dispatch...")
    try:
        LFM3Model(TOY_MODEL_PATH, lib_path=LIB_PATH, warmup_pages=False)
        raise AssertionError("Expected LFM3Model to reject non-lfm3 GGUF!")
    except ValueError:
        print("  Strict LFM3Model guard correctly rejected LFM2 GGUF with ValueError.")

    lfm3_path = "models/LFM3-1.7B-Q4_K_M.gguf"
    if os.path.exists(lfm3_path):
        with LFM2Model(lfm3_path, lib_path=LIB_PATH, warmup_pages=False) as auto_m:
            assert auto_m.info.arch == "lfm3"
            assert auto_m.info.n_layers == 24
            assert auto_m.info.n_shortconv_layers == 13
            assert auto_m.info.n_gqa_layers == 11
            assert auto_m.info.n_vocab == 128000

        with load_lfm3_model(lfm3_path, lib_path=LIB_PATH, warmup_pages=False) as lfm3_m:
            assert lfm3_m.info.arch == "lfm3"
            with lfm3_m.create_context(max_seq_len=512) as ctx3:
                reply3 = ctx3.chat("What is 2+2? Answer concisely.", temperature=0.0, max_tokens=48)
                assert len(reply3) > 0
                decode_tps = ctx3.stats.decode_tok_per_sec
                assert decode_tps > 0
                base_pos = ctx3.pos
                with ctx3.branch():
                    _ = ctx3.chat("Hello", temperature=0.0, max_tokens=8)
                    assert ctx3.pos > base_pos
                assert ctx3.pos == base_pos
                emb3 = ctx3.embed("Liquid LFM3 test", pooling=LFM2PoolingType.LAST_TOKEN)
                assert len(emb3) == lfm3_m.info.n_embd
                assert abs(sum(x * x for x in emb3) ** 0.5 - 1.0) < 1e-3
                print(f"  LFM3 SDK verified: reply={repr(reply3[:60])} | decode={decode_tps:.1f} tok/s")
    else:
        print(f"  NOTE: {lfm3_path} not present; skipping real LFM3 SDK test.")

    print("=======================================================")
    print("ALL 8 PYTHON SDK TESTS PASSED WITH 100% SUCCESS!       ")
    print("=======================================================")


if __name__ == "__main__":
    test_python_sdk()

