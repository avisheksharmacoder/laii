# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Avishek and Contributors
"""
Liquid AI (LFM2) Engine Python FFI Bindings.

Independent open-source implementation compatible with Liquid AI's LFM2 models.
Not affiliated with, endorsed by, or sponsored by Liquid AI, Inc.
"""

import ctypes
import os
import sys
from typing import Iterator, List, Optional


class _LFMSamplerParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("max_new_tokens", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("echo_prompt", ctypes.c_bool),
    ]


_LFM_TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_void_p)


def _find_liblfm(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return custom_path

    env_path = os.environ.get("LIBLFM_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Search common build relative paths
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")),
        "/usr/local/lib",
        "/usr/lib",
    ]

    for d in search_dirs:
        candidate = os.path.join(d, "liblfm.so")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find liblfm.so. Build it with `make lib` or set LIBLFM_PATH."
    )


class LFMEngine:
    """High-level Python wrapper for the Liquid AI (LFM2) specialized inference engine."""

    def __init__(self, model_path: str, num_threads: int = 0, lib_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        so_path = _find_liblfm(lib_path)
        self._lib = ctypes.CDLL(so_path)
        self._setup_signatures()

        self._handle = self._lib.lfm_create(model_path.encode("utf-8"), ctypes.c_int(num_threads))
        if not self._handle:
            raise RuntimeError(f"Failed to initialize LFM engine with model: {model_path}")

    def _setup_signatures(self):
        self._lib.lfm_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.lfm_create.restype = ctypes.c_void_p

        self._lib.lfm_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lfm_destroy.restype = None

        self._lib.lfm_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(_LFMSamplerParams),
        ]
        self._lib.lfm_generate.restype = ctypes.c_int

        self._lib.lfm_generate_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_LFMSamplerParams),
            _LFM_TOKEN_CALLBACK,
            ctypes.c_void_p,
        ]
        self._lib.lfm_generate_stream.restype = ctypes.c_int

        self._lib.lfm_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        self._lib.lfm_tokenize.restype = ctypes.c_int

        self._lib.lfm_detokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.lfm_detokenize.restype = ctypes.c_int

        self._lib.lfm_get_hidden_dim.argtypes = [ctypes.c_void_p]
        self._lib.lfm_get_hidden_dim.restype = ctypes.c_uint32

        self._lib.lfm_get_num_layers.argtypes = [ctypes.c_void_p]
        self._lib.lfm_get_num_layers.restype = ctypes.c_uint32

        self._lib.lfm_get_vocab_size.argtypes = [ctypes.c_void_p]
        self._lib.lfm_get_vocab_size.restype = ctypes.c_uint32

        self._lib.lfm_get_architecture.argtypes = [ctypes.c_void_p]
        self._lib.lfm_get_architecture.restype = ctypes.c_char_p

    def close(self):
        if self._handle:
            self._lib.lfm_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @property
    def hidden_dim(self) -> int:
        return int(self._lib.lfm_get_hidden_dim(self._handle))

    @property
    def num_layers(self) -> int:
        return int(self._lib.lfm_get_num_layers(self._handle))

    @property
    def vocab_size(self) -> int:
        return int(self._lib.lfm_get_vocab_size(self._handle))

    @property
    def architecture(self) -> str:
        res = self._lib.lfm_get_architecture(self._handle)
        return res.decode("utf-8") if res else "unknown"

    def tokenize(self, text: str, max_tokens: int = 1024) -> List[int]:
        out_arr = (ctypes.c_int32 * max_tokens)()
        n = self._lib.lfm_tokenize(
            self._handle, text.encode("utf-8"), out_arr, ctypes.c_int(max_tokens)
        )
        if n < 0:
            raise RuntimeError(f"Tokenization failed for text: {text}")
        return [int(out_arr[i]) for i in range(n)]

    def detokenize(self, token_id: int) -> str:
        buf = ctypes.create_string_buffer(256)
        n = self._lib.lfm_detokenize(
            self._handle, ctypes.c_int32(token_id), buf, ctypes.c_size_t(len(buf))
        )
        if n < 0:
            return ""
        return buf.value.decode("utf-8", errors="replace")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: int = 42,
    ) -> str:
        params = _LFMSamplerParams(
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_tokens,
            seed=seed,
            echo_prompt=False,
        )
        out_buf = ctypes.create_string_buffer(65536)
        n = self._lib.lfm_generate(
            self._handle,
            prompt.encode("utf-8"),
            out_buf,
            ctypes.c_size_t(len(out_buf)),
            ctypes.byref(params),
        )
        if n < 0:
            raise RuntimeError("Generation failed")
        return out_buf.value.decode("utf-8", errors="replace")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: int = 42,
    ) -> Iterator[str]:
        tokens = []

        def callback(token_id, piece, _):
            if piece:
                tokens.append(piece.decode("utf-8", errors="replace"))

        cb_func = _LFM_TOKEN_CALLBACK(callback)
        params = _LFMSamplerParams(
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_tokens,
            seed=seed,
            echo_prompt=False,
        )

        n = self._lib.lfm_generate_stream(
            self._handle,
            prompt.encode("utf-8"),
            ctypes.byref(params),
            cb_func,
            None,
        )
        if n < 0:
            raise RuntimeError("Streaming generation failed")

        for t in tokens:
            yield t
