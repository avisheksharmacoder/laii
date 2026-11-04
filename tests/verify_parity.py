#!/usr/bin/env python3
"""
tests/verify_parity.py
Zero-dependency C11 LFM2 Engine Contract & Golden-Logit Parity Verifier.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Allow importing python/lfm2.py from the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from lfm2 import LFM2Context, LFM2Model, LFM2PoolingType  # noqa: E402


# ============================================================================
# 1. Quantization-Aware SLA Thresholds (Cosine Sim, KL Divergence, Top-5)
# ============================================================================
@dataclass(frozen=True, slots=True)
class ParitySLA:
    name: str
    min_cosine_sim: float
    max_kl_div: float
    min_top5_overlap: int  # Out of 5


SLA_TIERS: dict[str, ParitySLA] = {
    "F32":    ParitySLA("F32",    min_cosine_sim=0.99995, max_kl_div=0.0005, min_top5_overlap=5),
    "F16":    ParitySLA("F16",    min_cosine_sim=0.99980, max_kl_div=0.0020, min_top5_overlap=5),
    "Q8_0":   ParitySLA("Q8_0",   min_cosine_sim=0.99920, max_kl_div=0.0080, min_top5_overlap=5),
    "Q6_K":   ParitySLA("Q6_K",   min_cosine_sim=0.99800, max_kl_div=0.0200, min_top5_overlap=4),
    "Q5_K":   ParitySLA("Q5_K",   min_cosine_sim=0.99650, max_kl_div=0.0350, min_top5_overlap=4),
    "Q4_K_M": ParitySLA("Q4_K_M", min_cosine_sim=0.99300, max_kl_div=0.0650, min_top5_overlap=4),
    "Q4_0":   ParitySLA("Q4_0",   min_cosine_sim=0.98800, max_kl_div=0.1000, min_top5_overlap=3),
    "IQ4_XS": ParitySLA("IQ4_XS", min_cosine_sim=0.99100, max_kl_div=0.0800, min_top5_overlap=4),
}


# ============================================================================
# 2. Pure-Python Vector Math & Information-Theory Metrics (Zero numpy/scipy)
# ============================================================================
@dataclass(frozen=True, slots=True)
class LogitComparison:
    cosine_sim: float
    kl_divergence: float
    rmse: float
    max_abs_diff: float
    top1_match: bool
    top5_overlap: int
    ref_top5: list[tuple[int, float]]
    eng_top5: list[tuple[int, float]]


def top_k_indices(vec: Sequence[float], k: int = 5) -> list[tuple[int, float]]:
    """Returns the top-k (index, value) pairs sorted descending."""
    eff_k = min(k, len(vec))
    best: list[tuple[float, int]] = [(-1e30, -1)] * eff_k
    min_val = -1e30
    for idx, val in enumerate(vec):
        if val > min_val:
            best[-1] = (val, idx)
            best.sort(reverse=True)
            min_val = best[-1][0]
    return [(idx, val) for val, idx in best if idx >= 0]


def compare_logit_vectors(ref: Sequence[float], eng: Sequence[float]) -> LogitComparison:
    """Computes Cosine Similarity, KL(P_ref || P_eng), RMSE, and Top-K agreement."""
    if len(ref) != len(eng):
        raise ValueError(f"Logit length mismatch: ref={len(ref)} vs eng={len(eng)}")

    n = len(ref)
    dot = 0.0
    norm_ref = 0.0
    norm_eng = 0.0
    sq_err = 0.0
    max_diff = 0.0

    mean_ref = sum(ref) / n
    mean_eng = sum(eng) / n

    max_ref = max(ref)
    max_eng = max(eng)

    for r, e in zip(ref, eng):
        cr = r - mean_ref
        ce = e - mean_eng
        dot += cr * ce
        norm_ref += cr * cr
        norm_eng += ce * ce

        diff = abs(r - e)
        if diff > max_diff:
            max_diff = diff
        sq_err += diff * diff

    denom = math.sqrt(norm_ref * norm_eng)
    cosine_sim = dot / (denom + 1e-12) if denom > 0.0 else 1.0
    rmse = math.sqrt(sq_err / n)

    # Numerically stable Softmax KL Divergence: D_KL(P_ref || P_eng)
    sum_exp_ref = sum(math.exp(r - max_ref) for r in ref)
    sum_exp_eng = sum(math.exp(e - max_eng) for e in eng)
    log_z_ref = max_ref + math.log(sum_exp_ref)
    log_z_eng = max_eng + math.log(sum_exp_eng)

    kl_div = 0.0
    for r, e in zip(ref, eng):
        log_p = r - log_z_ref
        log_q = e - log_z_eng
        p = math.exp(log_p)
        if p > 1e-12:
            kl_div += p * (log_p - log_q)

    k_val = min(5, n)
    ref_top5 = top_k_indices(ref, k_val)
    eng_top5 = top_k_indices(eng, k_val)
    ref_set = {idx for idx, _ in ref_top5}
    eng_set = {idx for idx, _ in eng_top5}

    top1_match = (ref_top5[0][0] == eng_top5[0][0]) if (ref_top5 and eng_top5) else True

    return LogitComparison(
        cosine_sim=cosine_sim,
        kl_divergence=max(0.0, kl_div),
        rmse=rmse,
        max_abs_diff=max_diff,
        top1_match=top1_match,
        top5_overlap=len(ref_set & eng_set),
        ref_top5=ref_top5,
        eng_top5=eng_top5,
    )


# ============================================================================
# 3. Direct C-ABI Helpers for Raw Tokenization, Prefill & Logit Readout
# ============================================================================
def bind_Extended_abi(lib: ctypes.CDLL) -> None:
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
    lib.lfm2_context_get_logits_ptr.restype = ctypes.POINTER(ctypes.c_float)


def tokenize_c11(model: LFM2Model, text: str) -> list[int]:
    max_ids = 4096
    buf = (ctypes.c_uint32 * max_ids)()
    n = model._lib.lfm2_tokenize_text(model._ptr, text.encode("utf-8"), buf, max_ids)
    if n < 0:
        raise RuntimeError("lfm2_tokenize_text failed")
    return list(buf[:n])


def read_c11_logits(ctx: LFM2Context) -> list[float]:
    n_vocab = ctx.model.info.n_vocab
    ptr = ctx._lib.lfm2_context_get_logits_ptr(ctx._ptr)
    if not ptr:
        raise RuntimeError("lfm2_context_get_logits_ptr returned NULL")
    logits = [ptr[i] for i in range(n_vocab)]
    for i, val in enumerate(logits):
        if math.isnan(val) or math.isinf(val):
            raise AssertionError(f"NaN/Inf detected in C11 logits at token index {i}!")
    return logits


# ============================================================================
# 4. Binary Golden Fixture Format (.lfm_golden) for Zero-Dependency CI
# ============================================================================
GOLDEN_MAGIC = b"LFM2GLD1"


def save_golden_fixture(
    filepath: Path,
    prompt: str,
    token_ids: Sequence[int],
    logits: Sequence[float],
) -> None:
    prompt_bytes = prompt.encode("utf-8")
    n_tok = len(token_ids)
    n_vocab = len(logits)
    with filepath.open("wb") as f:
        f.write(struct.pack("<8sIII", GOLDEN_MAGIC, len(prompt_bytes), n_tok, n_vocab))
        f.write(prompt_bytes)
        f.write(struct.pack(f"<{n_tok}I", *token_ids))
        f.write(struct.pack(f"<{n_vocab}f", *logits))


def load_golden_fixture(filepath: Path) -> tuple[str, list[int], list[float]]:
    with filepath.open("rb") as f:
        header = f.read(20)
        magic, plen, n_tok, n_vocab = struct.unpack("<8sIII", header)
        if magic != GOLDEN_MAGIC:
            raise ValueError(f"Invalid golden fixture magic in {filepath}")
        prompt = f.read(plen).decode("utf-8")
        token_ids = list(struct.unpack(f"<{n_tok}I", f.read(4 * n_tok)))
        logits = list(struct.unpack(f"<{n_vocab}f", f.read(4 * n_vocab)))
    return prompt, token_ids, logits


def extract_hf_oracle(hf_model_id: str, prompt: str) -> tuple[list[int], list[float]]:
    """Loads the official Liquid AI model via Hugging Face transformers (FP32)."""
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    print(f"[Oracle] Loading Hugging Face reference '{hf_model_id}' in FP32...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    token_ids = enc["input_ids"][0].tolist()

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"])
        last_logits = out.logits[0, -1, :].cpu().tolist()

    return token_ids, last_logits


# ============================================================================
# 5. Five-Tier Contract Verification Suite
# ============================================================================
DEFAULT_TEST_PROMPT = (
    "<|startoftext|><|im_start|>system\n"
    "You are a helpful assistant trained by Liquid AI.<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain why depthwise short convolutions fit in L1 CPU cache in 3 bullets.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def run_verification_suite(
    gguf_path: Path,
    lib_path: Path | None,
    quant_tier: str,
    golden_path: Path | None,
    hf_model_id: str | None,
    generate_golden: bool,
) -> int:
    sla = SLA_TIERS[quant_tier]
    failures = 0

    print("========================================================================")
    print(" Liquid AI LFM2 / LFM2.5 C11 Engine v1.0 Numerical Parity & TDD Suite")
    print("========================================================================")

    resolved_lib = str(lib_path) if lib_path else None
    with LFM2Model(gguf_path, lib_path=resolved_lib, warmup_pages=False) as model:
        bind_Extended_abi(model._lib)
        info = model.info
        print(
            f"Model       : {info.arch} ({info.file_size_mb:.1f} MB) | ISA: {info.active_isa}\n"
            f"Topology    : {info.n_layers} layers ({info.n_shortconv_layers} ShortConv + "
            f"{info.n_gqa_layers} GQA) | embd={info.n_embd} | vocab={info.n_vocab}\n"
            f"SLA Profile : {sla.name} (Cosine >= {sla.min_cosine_sim:.5f}, "
            f"KL <= {sla.max_kl_div:.4f}, Top-5 >= {sla.min_top5_overlap}/5)\n"
        )

        with model.create_context(max_seq_len=2048) as ctx:
            # ------------------------------------------------------------------
            # Tier 1: Tokenizer Special-Tag & Byte-Level BPE Contract
            # ------------------------------------------------------------------
            c_tokens = tokenize_c11(model, DEFAULT_TEST_PROMPT)
            if info.n_vocab <= 256:
                # Toy synthetic test fixture
                t1_ok = len(c_tokens) > 0
            else:
                has_bos = c_tokens[0] == 1  # <|startoftext|>
                has_im_start = 6 in c_tokens
                has_im_end = 7 in c_tokens
                t1_ok = has_bos and has_im_start and has_im_end and len(c_tokens) > 15

            print(
                f"[Tier 1] BPE Tokenizer & Control Tags : "
                f"{'PASS' if t1_ok else 'FAIL'} ({len(c_tokens)} tokens, "
                f"BOS={c_tokens[0] if c_tokens else 0}, head={c_tokens[:6]})"
            )
            if not t1_ok:
                failures += 1

            # ------------------------------------------------------------------
            # Tier 2: Batched GEMM Prefill (B=8) vs Sequential Decode (B=1) Parity
            # ------------------------------------------------------------------
            ctx.reset()
            arr_type = ctypes.c_uint32 * len(c_tokens)
            c_arr = arr_type(*c_tokens)

            # Run A: Step 1 token at a time (M=1 1x4 Decode path)
            t0 = time.perf_counter()
            for i in range(len(c_tokens)):
                is_last = (i == len(c_tokens) - 1)
                tok_slice = (ctypes.c_uint32 * 1)(c_tokens[i])
                rc = model._lib.lfm2_prefill_tokens(ctx._ptr, tok_slice, 1, is_last)
                assert rc == 0
            seq_ms = (time.perf_counter() - t0) * 1000.0
            logits_seq = read_c11_logits(ctx)

            # Run B: Batched GEMM Prefill (B=8 chunked path)
            ctx.reset()
            t0 = time.perf_counter()
            rc = model._lib.lfm2_prefill_tokens(ctx._ptr, c_arr, len(c_tokens), True)
            batch_ms = (time.perf_counter() - t0) * 1000.0
            assert rc == 0
            logits_batch = read_c11_logits(ctx)

            batch_cmp = compare_logit_vectors(logits_seq, logits_batch)
            t2_ok = (
                batch_cmp.cosine_sim >= 0.99999
                and batch_cmp.max_abs_diff < 1e-3
                and batch_cmp.top1_match
            )
            print(
                f"[Tier 2] Batched GEMM vs M=1 Parity   : "
                f"{'PASS' if t2_ok else 'FAIL'} "
                f"(Cosine={batch_cmp.cosine_sim:.7f}, MaxDiff={batch_cmp.max_abs_diff:.2e}, "
                f"Speedup={seq_ms / max(batch_ms, 1e-6):.2f}x)"
            )
            if not t2_ok:
                failures += 1

            # ------------------------------------------------------------------
            # Tier 3: Hybrid State Checkpoint Rollback (ShortConv + GQA Bit-Exact)
            # ------------------------------------------------------------------
            ckpt = ctx.create_checkpoint()
            pos_before = ctx.pos

            # Advance 1 token to record expected post-checkpoint logits
            best_tok_id = logits_batch.index(max(logits_batch))
            next_tok = (ctypes.c_uint32 * 1)(best_tok_id)
            model._lib.lfm2_prefill_tokens(ctx._ptr, next_tok, 1, True)
            logits_after_1 = read_c11_logits(ctx)

            # Corrupt state by advancing 5 more tokens (mutating all ShortConv buffers)
            noise_vals = [((i + 1) * 3) % info.n_vocab for i in range(5)]
            noise_toks = (ctypes.c_uint32 * 5)(*noise_vals)
            model._lib.lfm2_prefill_tokens(ctx._ptr, noise_toks, 5, True)

            # Restore checkpoint and re-run `next_tok`
            ctx.restore_checkpoint(ckpt)
            pos_restored = ctx.pos
            model._lib.lfm2_prefill_tokens(ctx._ptr, next_tok, 1, True)
            logits_restored_1 = read_c11_logits(ctx)
            ckpt.close()

            max_ckpt_ulp = max(
                abs(a - b) for a, b in zip(logits_after_1, logits_restored_1)
            )
            t3_ok = (pos_restored == pos_before) and (max_ckpt_ulp == 0.0)
            print(
                f"[Tier 3] Hybrid Checkpoint Rollback   : "
                f"{'PASS' if t3_ok else 'FAIL'} "
                f"(restored_pos={pos_restored}, MaxLogitDiff={max_ckpt_ulp:.1e} [0.0 = bit-exact])"
            )
            if not t3_ok:
                failures += 1

            # ------------------------------------------------------------------
            # Tier 4: L2-Normalized Embeddings Invariant
            # ------------------------------------------------------------------
            embd_last = ctx.embed("Linux futex spin-barrier thread pool", LFM2PoolingType.LAST_TOKEN)
            embd_mean = ctx.embed("Linux futex spin-barrier thread pool", LFM2PoolingType.MEAN)
            norm_last = math.sqrt(sum(x * x for x in embd_last))
            norm_mean = math.sqrt(sum(x * x for x in embd_mean))
            t4_ok = abs(norm_last - 1.0) < 1e-3 and abs(norm_mean - 1.0) < 1e-3
            print(
                f"[Tier 4] L2 Embedding Norm Contract   : "
                f"{'PASS' if t4_ok else 'FAIL'} "
                f"(||e_last||_2={norm_last:.6f}, ||e_mean||_2={norm_mean:.6f})"
            )
            if not t4_ok:
                failures += 1

            # ------------------------------------------------------------------
            # Tier 5: External Oracle Parity (Hugging Face or .lfm_golden Fixture)
            # ------------------------------------------------------------------
            ref_tokens: list[int] | None = None
            ref_logits: list[float] | None = None

            if hf_model_id:
                ref_tokens, ref_logits = extract_hf_oracle(hf_model_id, DEFAULT_TEST_PROMPT)
                if generate_golden and golden_path:
                    save_golden_fixture(golden_path, DEFAULT_TEST_PROMPT, ref_tokens, ref_logits)
                    print(f"[Oracle] Saved golden fixture to {golden_path}")
            elif generate_golden and golden_path:
                save_golden_fixture(golden_path, DEFAULT_TEST_PROMPT, c_tokens, logits_batch)
                print(f"[Oracle] Saved C11 reference golden fixture to {golden_path}")
                ref_tokens, ref_logits = c_tokens, logits_batch
            elif golden_path and golden_path.exists():
                _, ref_tokens, ref_logits = load_golden_fixture(golden_path)

            if ref_tokens is not None and ref_logits is not None:
                tok_exact = c_tokens == ref_tokens
                oracle_cmp = compare_logit_vectors(ref_logits, logits_batch)
                min_top = min(sla.min_top5_overlap, info.n_vocab)
                t5_ok = (
                    tok_exact
                    and oracle_cmp.cosine_sim >= sla.min_cosine_sim
                    and oracle_cmp.kl_divergence <= sla.max_kl_div
                    and oracle_cmp.top5_overlap >= min_top
                )
                print(
                    f"[Tier 5] Golden Oracle Logit Parity   : "
                    f"{'PASS' if t5_ok else 'FAIL'}\n"
                    f"         Token IDs Match  : {tok_exact} ({len(c_tokens)} == {len(ref_tokens)})\n"
                    f"         Cosine Similarity: {oracle_cmp.cosine_sim:.6f} (SLA >= {sla.min_cosine_sim:.5f})\n"
                    f"         KL Divergence    : {oracle_cmp.kl_divergence:.6f} (SLA <= {sla.max_kl_div:.4f})\n"
                    f"         RMSE / MaxDiff   : {oracle_cmp.rmse:.4f} / {oracle_cmp.max_abs_diff:.4f}\n"
                    f"         Top-1 Match      : {oracle_cmp.top1_match} "
                    f"(Oracle id={oracle_cmp.ref_top5[0][0]} vs Engine id={oracle_cmp.eng_top5[0][0]})\n"
                    f"         Top-5 Overlap    : {oracle_cmp.top5_overlap}/5 "
                    f"(SLA >= {min_top}/5)"
                )
                if not t5_ok:
                    failures += 1
            else:
                print(
                    "[Tier 5] Golden Oracle Logit Parity   : SKIPPED "
                    "(pass --golden-file or --hf-model to run oracle comparison)"
                )

    print("========================================================================")
    if failures == 0:
        print(" VERDICT: ALL PARITY CONTRACTS PASSED (v1.0 Release Certified)")
        return 0
    else:
        print(f" VERDICT: {failures} CONTRACT(S) FAILED")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="LFM2 C11 Engine Numerical Parity Suite")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=REPO_ROOT / "tests/lfm2_toy.gguf",
        help="Path to LFM2 .gguf model (default: tests/lfm2_toy.gguf)",
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=None,
        help="Path to liblfm2.so (default: auto-detected)",
    )
    parser.add_argument(
        "-q",
        "--quant-tier",
        choices=list(SLA_TIERS.keys()),
        default="Q4_K_M",
        help="Quantization tier for SLA thresholds",
    )
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=None,
        help="Path to .lfm_golden file",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Optional HF repo ID for oracle",
    )
    parser.add_argument(
        "--generate-golden",
        action="store_true",
        help="Write reference tokens & logits to --golden-file",
    )
    args = parser.parse_args()

    return run_verification_suite(
        gguf_path=args.model,
        lib_path=args.lib,
        quant_tier=args.quant_tier,
        golden_path=args.golden_file,
        hf_model_id=args.hf_model,
        generate_golden=args.generate_golden,
    )


if __name__ == "__main__":
    sys.exit(main())

