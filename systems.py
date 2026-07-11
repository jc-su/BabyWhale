"""A systems lens — the recurring "by the numbers" beat (TinyTorch's signature).

baby_whale is a *systems* project: MLA, MoE, the KV cache, quantization are all
memory/compute tradeoffs. TinyTorch makes memory and complexity first-class in
every module; these helpers let each of ours end on a concrete, countable number
instead of a vibe. Everything here is derived from a real config or model.
"""

from __future__ import annotations

from dataclasses import dataclass

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.model import BabyWhaleV4Model

BYTES_PER_ELEM: dict[str, float] = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp4": 0.5}


def param_count(cfg: BabyWhaleV4Config) -> int:
    return BabyWhaleV4Model(cfg).num_parameters()


def weight_memory_mb(n_params: int, precision: str = "bf16") -> float:
    if precision not in BYTES_PER_ELEM:
        raise ValueError(f"unknown precision {precision!r}")
    return n_params * BYTES_PER_ELEM[precision] / 1e6


def attention_flops(seq_len: int, d_model: int, *, window: int | None = None) -> int:
    """~MAC count for one attention layer over a sequence.

    Each of ``seq_len`` tokens attends to ``reach`` keys, ~2·reach·d work each
    (scores, then weighted values). Full attention: reach = seq_len -> O(n^2·d).
    Sliding window: reach = window -> O(n·W·d). This is the whole reason windows
    (Module 02) and compression (Module 04) exist.
    """
    reach = seq_len if window is None else min(window, seq_len)
    return seq_len * 2 * reach * d_model


def moe_sparsity(cfg: BabyWhaleV4Config) -> tuple[int, int]:
    """(active experts per token, total experts) — capacity decoupled from FLOPs."""
    return cfg.experts_per_token + cfg.n_shared_expert, cfg.n_expert + cfg.n_shared_expert


@dataclass(frozen=True)
class SystemsSummary:
    params: int
    weight_mb_bf16: float
    weight_mb_fp4: float
    active_experts: int
    total_experts: int
    attn_flops_full: int
    attn_flops_windowed: int


def summarize(cfg: BabyWhaleV4Config) -> SystemsSummary:
    params = param_count(cfg)
    active, total = moe_sparsity(cfg)
    return SystemsSummary(
        params=params,
        weight_mb_bf16=weight_memory_mb(params, "bf16"),
        weight_mb_fp4=weight_memory_mb(params, "fp4"),
        active_experts=active,
        total_experts=total,
        attn_flops_full=attention_flops(cfg.context_length, cfg.n_embd),
        attn_flops_windowed=attention_flops(
            cfg.context_length, cfg.n_embd, window=cfg.sliding_window
        ),
    )


def print_systems(cfg: BabyWhaleV4Config) -> None:
    s = summarize(cfg)
    print(f"params:            {s.params:,}")
    print(f"weights (bf16):    {s.weight_mb_bf16:8.2f} MB")
    print(
        f"weights (fp4):     {s.weight_mb_fp4:8.2f} MB   ({s.weight_mb_bf16 / s.weight_mb_fp4:.0f}x smaller)"
    )
    print(f"experts/token:     {s.active_experts} of {s.total_experts} active")
    print(f"attn MACs (full):  {s.attn_flops_full:,}")
    print(
        f"attn MACs (window):{s.attn_flops_windowed:,}   ({s.attn_flops_full / max(1, s.attn_flops_windowed):.1f}x less)"
    )
