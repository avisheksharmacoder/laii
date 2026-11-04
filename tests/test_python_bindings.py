#!/usr/bin/env python3
"""
Test suite for Python ctypes bindings (laii.py).
"""

import os
import sys

# Ensure repo root is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "bindings", "python"))

from laii import LFMEngine


def main():
    print("[Test] Running Python ctypes bindings unit test...")
    model_path = os.path.join(REPO_ROOT, "tests", "lfm2_toy.gguf")
    lib_path = os.path.join(REPO_ROOT, "lib", "liblfm.so")

    assert os.path.exists(model_path), f"Toy model missing at {model_path}"
    assert os.path.exists(lib_path), f"Shared library missing at {lib_path}"

    with LFMEngine(model_path, num_threads=2, lib_path=lib_path) as engine:
        print(f"  Initialized Python LFMEngine: arch={engine.architecture}")
        assert engine.architecture == "lfm2"
        assert engine.hidden_dim == 64
        assert engine.num_layers == 2
        assert engine.vocab_size == 32
        print(
            f"  Metadata verified: {engine.num_layers} layers, {engine.hidden_dim} dim, {engine.vocab_size} vocab."
        )

        # 1. Tokenize
        tokens = engine.tokenize("Liquid Foundation Models")
        print(f"  Tokenized 'Liquid Foundation Models' -> {tokens}")
        assert tokens == [1, 3, 4, 5]

        # 2. Detokenize
        piece = engine.detokenize(3)
        print(f"  Detokenized token 3 -> '{piece}'")
        assert piece == "Liquid"

        # 3. Synchronous Generate
        text = engine.generate("Liquid Foundation Models", max_tokens=8, temperature=0.7)
        print(f"  Synchronous generation output: '{text}'")
        assert len(text) > 0

        # 4. Streaming Generate
        streamed = list(engine.generate_stream("Liquid", max_tokens=6, temperature=0.8))
        print(f"  Streaming generation pieces: {streamed}")
        assert len(streamed) > 0

    print("PASSED: Python ctypes bindings verified successfully!")


if __name__ == "__main__":
    main()
