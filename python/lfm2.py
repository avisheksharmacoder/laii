# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Avishek and Contributors
"""
python/lfm2.py - Zero-dependency Python SDK & ctypes bindings for the C11 LFM2 Engine.

Independent open-source implementation compatible with Liquid AI's LFM2 models.
Not affiliated with, endorsed by, or sponsored by Liquid AI, Inc.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterator, Sequence


__all__ = [
    "LFM2Checkpoint",
    "LFM2Context",
    "LFM2GrammarType",
    "LFM2Model",
    "LFM2PoolingType",
    "LFM2Status",
    "ModelInfo",
    "PerfStats",
    "run_server",
]


# ============================================================================
# 1. C11 ABI Enums & Fixed-Width ctypes Structures (Matches include/lfm2.h)
# ============================================================================
class LFM2Status(IntEnum):
    OK              =  0
    DONE            =  1
    ERR_INVALID_ARG = -1
    ERR_IO          = -2
    ERR_CORRUPT     = -3
    ERR_OOM         = -4
    ERR_CTX_FULL    = -5
    ERR_CHECKPOINT  = -6


class LFM2GrammarType(IntEnum):
    NONE        = 0
    JSON_OBJECT = 1
    TOOL_CALL   = 2


class LFM2PoolingType(IntEnum):
    LAST_TOKEN = 0
    MEAN       = 1


class _CModelInfo(ctypes.Structure):
    _fields_ = [
        ("arch",               ctypes.c_char * 32),
        ("active_isa",         ctypes.c_char * 32),
        ("n_layers",           ctypes.c_uint32),
        ("n_shortconv_layers", ctypes.c_uint32),
        ("n_gqa_layers",       ctypes.c_uint32),
        ("n_embd",             ctypes.c_uint32),
        ("n_ff",               ctypes.c_uint32),
        ("n_vocab",            ctypes.c_uint32),
        ("n_ctx_train",        ctypes.c_uint32),
        ("n_experts",          ctypes.c_uint32),
        ("n_experts_used",     ctypes.c_uint32),
        ("file_size_bytes",    ctypes.c_uint64),
    ]


class _CContextParams(ctypes.Structure):
    _fields_ = [
        ("max_seq_len",    ctypes.c_uint32),
        ("n_threads",      ctypes.c_uint32),
        ("n_sink_tokens",  ctypes.c_uint32),
        ("auto_clamp_ram", ctypes.c_int32),
        ("warmup_mmap",    ctypes.c_int32),
    ]


class _CSampleParams(ctypes.Structure):
    _fields_ = [
        ("temperature",      ctypes.c_float),
        ("min_p",            ctypes.c_float),
        ("top_k",            ctypes.c_uint32),
        ("rep_penalty",      ctypes.c_float),
        ("rep_window",       ctypes.c_uint32),
        ("seed",             ctypes.c_uint64),
        ("grammar_type",     ctypes.c_int32),
        ("max_think_tokens", ctypes.c_uint32),
    ]


class _CPerfStats(ctypes.Structure):
    _fields_ = [
        ("prompt_tokens",       ctypes.c_uint32),
        ("generated_tokens",    ctypes.c_uint32),
        ("current_pos",         ctypes.c_uint32),
        ("prefill_ms",          ctypes.c_double),
        ("decode_ms",           ctypes.c_double),
        ("prefill_tok_per_sec", ctypes.c_double),
        ("decode_tok_per_sec",  ctypes.c_double),
    ]


@dataclass(frozen=True, slots=True)
class ModelInfo:
    arch: str
    active_isa: str
    n_layers: int
    n_shortconv_layers: int
    n_gqa_layers: int
    n_embd: int
    n_ff: int
    n_vocab: int
    n_ctx_train: int
    n_experts: int
    n_experts_used: int
    file_size_mb: float


@dataclass(frozen=True, slots=True)
class PerfStats:
    prompt_tokens: int
    generated_tokens: int
    current_pos: int
    prefill_ms: float
    decode_ms: float
    prefill_tok_per_sec: float
    decode_tok_per_sec: float


# ============================================================================
# 2. Shared Library Loader & Signature Binder
# ============================================================================
def _find_liblfm2(lib_path: str | Path | None = None) -> Path:
    if lib_path is not None:
        p = Path(lib_path)
        if p.exists():
            return p

    # 1. Check explicit environment override
    env_path = os.environ.get("LFM2_LIB_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Package directory (for wheel/pip installed distributions) and workspace
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "liblfm2.so",
        pkg_dir / "lib" / "liblfm2.so",
        pkg_dir.parent / "lib" / "liblfm2.so",
        pkg_dir.parent / "liblfm2.so",
        Path("./lib/liblfm2.so"),
        Path("./liblfm2.so"),
    ]

    # 3. Android Scoped Storage / Private native library locations
    for env_var in ("ANDROID_PRIVATE", "ANDROID_APP_PATH"):
        android_dir = os.environ.get(env_var)
        if android_dir:
            candidates.extend([
                Path(android_dir) / "lib" / "liblfm2.so",
                Path(android_dir) / "liblfm2.so",
            ])

    for c in candidates:
        if c.exists():
            return c

    # 4. System library fallback via ld.so
    sys_lib = ctypes.util.find_library("lfm2")
    if sys_lib:
        return Path(sys_lib)

    return Path(lib_path or "./lib/liblfm2.so")


def _bind_liblfm2(lib_path: str | Path | None = None) -> ctypes.CDLL:
    resolved = _find_liblfm2(lib_path)
    lib = ctypes.CDLL(str(resolved))

    # Parameter defaults
    lib.lfm2_context_default_params.argtypes = []
    lib.lfm2_context_default_params.restype  = _CContextParams
    lib.lfm2_sample_default_params.argtypes  = []
    lib.lfm2_sample_default_params.restype   = _CSampleParams

    # Model lifecycle
    lib.lfm2_model_load.argtypes     = [ctypes.c_char_p, ctypes.c_int32]
    lib.lfm2_model_load.restype      = ctypes.c_void_p
    lib.lfm2_model_load_from_fd.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_int32]
    lib.lfm2_model_load_from_fd.restype  = ctypes.c_void_p
    lib.lfm2_model_release_page_cache.argtypes = [ctypes.c_void_p]
    lib.lfm2_model_release_page_cache.restype  = None
    lib.lfm2_model_warmup_page_cache.argtypes  = [ctypes.c_void_p]
    lib.lfm2_model_warmup_page_cache.restype   = None
    lib.lfm2_get_process_rss_bytes.argtypes    = []
    lib.lfm2_get_process_rss_bytes.restype     = ctypes.c_uint64
    lib.lfm2_model_free.argtypes     = [ctypes.c_void_p]
    lib.lfm2_model_free.restype      = None
    lib.lfm2_model_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CModelInfo)]
    lib.lfm2_model_get_info.restype  = ctypes.c_int32

    # Context lifecycle
    lib.lfm2_context_create.argtypes    = [ctypes.c_void_p, ctypes.POINTER(_CContextParams)]
    lib.lfm2_context_create.restype     = ctypes.c_void_p
    lib.lfm2_context_free.argtypes      = [ctypes.c_void_p]
    lib.lfm2_context_free.restype       = None
    lib.lfm2_context_reset.argtypes     = [ctypes.c_void_p]
    lib.lfm2_context_reset.restype      = None
    lib.lfm2_context_get_pos.argtypes   = [ctypes.c_void_p]
    lib.lfm2_context_get_pos.restype    = ctypes.c_uint32
    lib.lfm2_context_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CPerfStats)]
    lib.lfm2_context_get_stats.restype  = None

    # Prompting, Stops & Streaming
    lib.lfm2_context_set_system_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.lfm2_context_set_system_prompt.restype  = ctypes.c_int32
    lib.lfm2_context_clear_stops.argtypes       = [ctypes.c_void_p]
    lib.lfm2_context_clear_stops.restype        = None
    lib.lfm2_context_add_stop.argtypes          = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lfm2_context_add_stop.restype           = ctypes.c_int32
    lib.lfm2_begin_chat_turn.argtypes           = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_CSampleParams)
    ]
    lib.lfm2_begin_chat_turn.restype            = ctypes.c_int32
    lib.lfm2_context_next_utf8.argtypes         = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.lfm2_context_next_utf8.restype          = ctypes.c_int32

    # Embeddings
    lib.lfm2_embed_text.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)
    ]
    lib.lfm2_embed_text.restype  = ctypes.c_int32

    # Direct Tokenization, Prefill & Zero-Copy Logits
    lib.lfm2_tokenize_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.lfm2_tokenize_text.restype = ctypes.c_int32

    lib.lfm2_prefill_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_bool,
    ]
    lib.lfm2_prefill_tokens.restype = ctypes.c_int32

    lib.lfm2_context_get_logits_ptr.argtypes = [ctypes.c_void_p]
    lib.lfm2_context_get_logits_ptr.restype  = ctypes.POINTER(ctypes.c_float)

    # Hybrid Checkpoints & Prompt Cache
    lib.lfm2_checkpoint_create.argtypes          = [ctypes.c_void_p]
    lib.lfm2_checkpoint_create.restype           = ctypes.c_void_p
    lib.lfm2_checkpoint_free.argtypes            = [ctypes.c_void_p]
    lib.lfm2_checkpoint_free.restype             = None
    lib.lfm2_context_save_checkpoint.argtypes    = [ctypes.c_void_p, ctypes.c_void_p]
    lib.lfm2_context_save_checkpoint.restype     = ctypes.c_int32
    lib.lfm2_context_restore_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.lfm2_context_restore_checkpoint.restype  = ctypes.c_int32
    lib.lfm2_context_save_prompt_cache.argtypes  = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lfm2_context_save_prompt_cache.restype   = ctypes.c_int32

    # HTTP/SSE Server
    lib.lfm2_run_http_server.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint16,
    ]
    lib.lfm2_run_http_server.restype = ctypes.c_int

    return lib


# ============================================================================
# 3. Pythonic Model, Checkpoint & Context Classes
# ============================================================================
class LFM2Checkpoint:
    """Holds an in-RAM snapshot of LFM2 ShortConv ring buffers + GQA KV cursor."""

    def __init__(self, lib: ctypes.CDLL, model_ptr: int) -> None:
        self._lib = lib
        self._ptr = self._lib.lfm2_checkpoint_create(model_ptr)
        if not self._ptr:
            raise MemoryError("Failed to allocate LFM2Checkpoint.")

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.lfm2_checkpoint_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> LFM2Checkpoint:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class LFM2Model:
    """Zero-copy mmap'd LFM2 / LFM2.5 GGUF model."""

    def __init__(
        self,
        gguf_path: str | Path,
        lib_path: str | Path | None = None,
        warmup_pages: bool = True,
    ) -> None:
        self._lib = _bind_liblfm2(lib_path)
        encoded_path = str(gguf_path).encode("utf-8")
        self._ptr = self._lib.lfm2_model_load(encoded_path, int(warmup_pages))
        if not self._ptr:
            raise RuntimeError(f"Failed to load GGUF model: {gguf_path}")

        c_info = _CModelInfo()
        self._lib.lfm2_model_get_info(self._ptr, ctypes.byref(c_info))
        self.info = ModelInfo(
            arch=c_info.arch.decode("utf-8", errors="replace"),
            active_isa=c_info.active_isa.decode("utf-8", errors="replace"),
            n_layers=c_info.n_layers,
            n_shortconv_layers=c_info.n_shortconv_layers,
            n_gqa_layers=c_info.n_gqa_layers,
            n_embd=c_info.n_embd,
            n_ff=c_info.n_ff,
            n_vocab=c_info.n_vocab,
            n_ctx_train=c_info.n_ctx_train,
            n_experts=c_info.n_experts,
            n_experts_used=c_info.n_experts_used,
            file_size_mb=c_info.file_size_bytes / (1024.0 * 1024.0),
        )

    @classmethod
    def from_fd(
        cls,
        fd: int,
        file_size: int = 0,
        warmup_pages: bool = False,
        lib_path: str | Path | None = None,
    ) -> LFM2Model:
        """Loads a model directly from an open UNIX file descriptor.

        Ideal for Android Scoped Storage / Content URIs.
        The C engine creates a duplicated descriptor internally via dup(fd).
        """
        instance = cls.__new__(cls)
        instance._lib = _bind_liblfm2(lib_path)
        instance._ptr = instance._lib.lfm2_model_load_from_fd(fd, file_size, int(warmup_pages))
        if not instance._ptr:
            raise RuntimeError(f"Failed to load GGUF model from file descriptor: {fd}")

        c_info = _CModelInfo()
        instance._lib.lfm2_model_get_info(instance._ptr, ctypes.byref(c_info))
        instance.info = ModelInfo(
            arch=c_info.arch.decode("utf-8", errors="replace"),
            active_isa=c_info.active_isa.decode("utf-8", errors="replace"),
            n_layers=c_info.n_layers,
            n_shortconv_layers=c_info.n_shortconv_layers,
            n_gqa_layers=c_info.n_gqa_layers,
            n_embd=c_info.n_embd,
            n_ff=c_info.n_ff,
            n_vocab=c_info.n_vocab,
            n_ctx_train=c_info.n_ctx_train,
            n_experts=c_info.n_experts,
            n_experts_used=c_info.n_experts_used,
            file_size_mb=c_info.file_size_bytes / (1024.0 * 1024.0),
        )
        return instance

    def release_page_cache(self) -> None:
        """Releases mmap page cache via madvise(MADV_DONTNEED) and posix_fadvise.

        Designed for mobile lifecycle events (e.g. Android on_pause / Low Memory Killer survival).
        Pages fault back in automatically on subsequent decode steps.
        """
        if self._ptr:
            self._lib.lfm2_model_release_page_cache(self._ptr)

    def warmup_page_cache(self) -> None:
        """Actively warms GGUF weight pages sequentially into physical RAM.

        Designed for background execution during mobile on_resume() to eliminate first-token latency spikes.
        """
        if self._ptr:
            self._lib.lfm2_model_warmup_page_cache(self._ptr)

    @classmethod
    def get_process_rss_bytes(cls, lib_path: str | Path | None = None) -> int:
        """Returns current resident set size (RSS) of the calling process in bytes."""
        lib = _bind_liblfm2(lib_path)
        return int(lib.lfm2_get_process_rss_bytes())

    def create_context(
        self,
        max_seq_len: int = 4096,
        n_threads: int = 0,
        system_prompt: str | None = None,
        tools_json: str | None = None,
    ) -> LFM2Context:
        return LFM2Context(
            self,
            max_seq_len=max_seq_len,
            n_threads=n_threads,
            system_prompt=system_prompt,
            tools_json=tools_json,
        )

    def run_server(
        self,
        ctx: LFM2Context | None = None,
        bind_ip: str = "127.0.0.1",
        port: int = 8080,
    ) -> int:
        """Runs the single-threaded zero-dependency Linux epoll OpenAI HTTP/SSE server."""
        if ctx is not None:
            return ctx.run_server(bind_ip=bind_ip, port=port)
        with self.create_context() as c:
            return c.run_server(bind_ip=bind_ip, port=port)

    def tokenize(self, text: str, max_tokens: int = 4096) -> list[int]:
        """Encodes text into a list of BPE token IDs."""
        buf = (ctypes.c_uint32 * max_tokens)()
        n = self._lib.lfm2_tokenize_text(self._ptr, text.encode("utf-8"), buf, max_tokens)
        if n < 0:
            raise RuntimeError("lfm2_tokenize_text failed")
        return list(buf[:n])

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.lfm2_model_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> LFM2Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class LFM2Context:
    """Stateful LFM2 execution arena (ShortConv ring buffers + FP16 GQA KV-cache)."""

    def __init__(
        self,
        model: LFM2Model,
        max_seq_len: int = 4096,
        n_threads: int = 0,
        system_prompt: str | None = None,
        tools_json: str | None = None,
    ) -> None:
        self.model = model
        self._lib = model._lib

        params = self._lib.lfm2_context_default_params()
        params.max_seq_len = max_seq_len
        params.n_threads = n_threads

        self._ptr = self._lib.lfm2_context_create(model._ptr, ctypes.byref(params))
        if not self._ptr:
            raise MemoryError("Failed to allocate LFM2Context arena.")

        if system_prompt or tools_json:
            self.set_system_prompt(system_prompt, tools_json)

        # Pre-allocate reusable 512-byte UTF-8 streaming buffer for zero per-token allocation
        self._utf8_buf = ctypes.create_string_buffer(512)
        self._out_len  = ctypes.c_size_t(0)
        self._out_phase = ctypes.c_int32(0)

    def set_system_prompt(
        self, system_prompt: str | None = None, tools_json: str | None = None
    ) -> None:
        sys_b = system_prompt.encode("utf-8") if system_prompt else None
        tls_b = tools_json.encode("utf-8") if tools_json else None
        st = self._lib.lfm2_context_set_system_prompt(self._ptr, sys_b, tls_b)
        if st != LFM2Status.OK:
            raise RuntimeError(f"lfm2_context_set_system_prompt failed with code {st}")

    def set_stops(self, stop_strings: Sequence[str]) -> None:
        self._lib.lfm2_context_clear_stops(self._ptr)
        for s in stop_strings:
            self._lib.lfm2_context_add_stop(self._ptr, s.encode("utf-8"))

    def reset(self) -> None:
        self._lib.lfm2_context_reset(self._ptr)

    def prefill_tokens(self, tokens: Sequence[int], compute_last_logits: bool = True) -> None:
        """Prefills raw token IDs, optionally evaluating logits on the final token."""
        n = len(tokens)
        if n == 0:
            return
        c_arr = (ctypes.c_uint32 * n)(*tokens)
        st = self._lib.lfm2_prefill_tokens(self._ptr, c_arr, n, compute_last_logits)
        if st != LFM2Status.OK:
            raise RuntimeError(f"lfm2_prefill_tokens failed with status {st}")

    def get_logits(self) -> list[float]:
        """Returns the raw float logit distribution for the most recent prefill / decode step."""
        ptr = self._lib.lfm2_context_get_logits_ptr(self._ptr)
        if not ptr:
            raise RuntimeError("lfm2_context_get_logits_ptr returned NULL (logits not computed)")
        return [ptr[i] for i in range(self.model.info.n_vocab)]

    def get_logits_ptr(self) -> ctypes.POINTER(ctypes.c_float):
        """Returns the zero-copy ctypes float pointer directly to ctx->state.logits."""
        ptr = self._lib.lfm2_context_get_logits_ptr(self._ptr)
        if not ptr:
            raise RuntimeError("lfm2_context_get_logits_ptr returned NULL")
        return ptr

    @property
    def pos(self) -> int:
        return int(self._lib.lfm2_context_get_pos(self._ptr))

    @property
    def stats(self) -> PerfStats:
        cs = _CPerfStats()
        self._lib.lfm2_context_get_stats(self._ptr, ctypes.byref(cs))
        return PerfStats(
            prompt_tokens=cs.prompt_tokens,
            generated_tokens=cs.generated_tokens,
            current_pos=cs.current_pos,
            prefill_ms=cs.prefill_ms,
            decode_ms=cs.decode_ms,
            prefill_tok_per_sec=cs.prefill_tok_per_sec,
            decode_tok_per_sec=cs.decode_tok_per_sec,
        )

    def stream_chat(
        self,
        user_message: str,
        *,
        max_tokens: int = 0,
        temperature: float = 0.30,
        min_p: float = 0.15,
        top_k: int = 50,
        rep_penalty: float = 1.05,
        grammar: LFM2GrammarType = LFM2GrammarType.NONE,
        max_think_tokens: int = 0,
        stops: Sequence[str] | None = None,
    ) -> Iterator[str]:
        """Streams verified UTF-8 text chunks for a multi-turn ChatML conversation."""
        if stops is not None:
            self.set_stops(stops)

        sp = self._lib.lfm2_sample_default_params()
        sp.temperature      = temperature
        sp.min_p            = min_p
        sp.top_k            = top_k
        sp.rep_penalty      = rep_penalty
        sp.grammar_type     = int(grammar)
        sp.max_think_tokens = max_think_tokens

        st = self._lib.lfm2_begin_chat_turn(
            self._ptr, user_message.encode("utf-8"), ctypes.byref(sp)
        )
        if st != LFM2Status.OK:
            raise RuntimeError(f"lfm2_begin_chat_turn failed with status {LFM2Status(st).name}")

        tokens_emitted = 0
        while True:
            rc = self._lib.lfm2_context_next_utf8(
                self._ptr,
                self._utf8_buf,
                ctypes.sizeof(self._utf8_buf),
                ctypes.byref(self._out_len),
                ctypes.byref(self._out_phase),
            )
            if self._out_len.value > 0:
                yield self._utf8_buf.raw[: self._out_len.value].decode("utf-8")
                tokens_emitted += 1
                if max_tokens > 0 and tokens_emitted >= max_tokens:
                    break

            if rc == LFM2Status.DONE:
                break
            if rc < 0:
                raise RuntimeError(f"Decode error: {LFM2Status(rc).name}")

    def chat(self, user_message: str, **kwargs: object) -> str:
        return "".join(self.stream_chat(user_message, **kwargs))  # type: ignore[arg-type]

    def embed(
        self,
        text: str,
        pooling: LFM2PoolingType = LFM2PoolingType.LAST_TOKEN,
    ) -> list[float]:
        """Returns an L2-normalized embedding vector of length `model.info.n_embd`."""
        dim = self.model.info.n_embd
        buf = (ctypes.c_float * dim)()
        rc = self._lib.lfm2_embed_text(
            self._ptr, text.encode("utf-8"), int(pooling), buf
        )
        if rc != LFM2Status.OK:
            raise RuntimeError(f"lfm2_embed_text failed with status {LFM2Status(rc).name}")
        return list(buf)

    def create_checkpoint(self) -> LFM2Checkpoint:
        ckpt = LFM2Checkpoint(self._lib, self.model._ptr)
        self.save_checkpoint(ckpt)
        return ckpt

    def save_checkpoint(self, ckpt: LFM2Checkpoint) -> None:
        rc = self._lib.lfm2_context_save_checkpoint(self._ptr, ckpt._ptr)
        if rc != LFM2Status.OK:
            raise RuntimeError("Failed to save LFM2 hybrid checkpoint.")

    def restore_checkpoint(self, ckpt: LFM2Checkpoint) -> None:
        rc = self._lib.lfm2_context_restore_checkpoint(self._ptr, ckpt._ptr)
        if rc != LFM2Status.OK:
            raise RuntimeError("Failed to restore LFM2 hybrid checkpoint.")

    @contextmanager
    def branch(self) -> Iterator[None]:
        """Context manager that snapshots ShortConv + GQA state and rolls back on exit."""
        with LFM2Checkpoint(self._lib, self.model._ptr) as ckpt:
            self.save_checkpoint(ckpt)
            try:
                yield
            finally:
                self.restore_checkpoint(ckpt)

    def run_server(self, bind_ip: str = "127.0.0.1", port: int = 8080) -> int:
        """Runs the single-threaded zero-dependency Linux epoll OpenAI HTTP/SSE server."""
        return int(
            self._lib.lfm2_run_http_server(
                self.model._ptr, self._ptr, bind_ip.encode("utf-8"), port
            )
        )

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.lfm2_context_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> LFM2Context:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def run_server(
    model: LFM2Model,
    ctx: LFM2Context | None = None,
    bind_ip: str = "127.0.0.1",
    port: int = 8080,
) -> int:
    """Runs the single-threaded zero-dependency Linux epoll OpenAI HTTP/SSE server."""
    return model.run_server(ctx=ctx, bind_ip=bind_ip, port=port)


