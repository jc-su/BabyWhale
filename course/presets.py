"""Progressive-motivation presets.

Each preset turns *one more* feature on, so a learner can feel a wall before
climbing it: train ``gpt-minimal``, hit the pain, flip on the next preset, and
measure what changed. All are small (4 layers) so they train on a laptop in
seconds. They are ordinary :class:`BabyWhaleV4Config` objects — nothing course-
specific in the library.

    from course.presets import load_preset
    cfg = load_preset("plus-mla")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from baby_whale_v4.config import BabyWhaleV4Config


def _base() -> BabyWhaleV4Config:
    return BabyWhaleV4Config.tiny(vocab_size=256, context_length=128)


def gpt_minimal() -> BabyWhaleV4Config:
    """The baseline: every layer sliding-window MQA, no MTP, no balancing, no mHC.

    This is roughly "a competent small GPT". Everything after this is an upgrade
    you turn on only once you've felt why you want it.
    """
    return dataclasses.replace(
        _base(),
        name="course-gpt-minimal",
        n_layer=4,
        layer_schedule=("sliding_mqa", "sliding_mqa", "sliding_mqa", "sliding_mqa"),
        mtp_heads=0,
        aux_free_bias_rate=0.0,
        hc_mult=1,
    )


def plus_mla() -> BabyWhaleV4Config:
    """Swap alternating layers to MLA — low-rank latent KV (Module 03)."""
    return dataclasses.replace(
        gpt_minimal(),
        name="course-plus-mla",
        layer_schedule=("sliding_mqa", "mla", "sliding_mqa", "mla"),
    )


def plus_compressed() -> BabyWhaleV4Config:
    """Add HCA + CSA compressed attention for long-range reach (Module 04)."""
    return dataclasses.replace(
        plus_mla(),
        name="course-plus-compressed",
        layer_schedule=("sliding_mqa", "mla", "hca", "csa"),
    )


def plus_moe_balanced() -> BabyWhaleV4Config:
    """Turn on aux-loss-free load balancing — the per-expert bias (Module 05)."""
    return dataclasses.replace(
        plus_compressed(),
        name="course-plus-moe-balanced",
        aux_free_bias_rate=1e-3,
    )


def plus_mtp() -> BabyWhaleV4Config:
    """Add multi-token-prediction heads — unlocks speculative decoding (Modules 07/16)."""
    return dataclasses.replace(
        plus_moe_balanced(),
        name="course-plus-mtp",
        mtp_heads=2,
    )


def full() -> BabyWhaleV4Config:
    """Everything on, including learned multi-branch residuals (Module 06)."""
    return dataclasses.replace(plus_mtp(), name="course-full", hc_mult=2)


PRESETS: dict[str, Callable[[], BabyWhaleV4Config]] = {
    "gpt-minimal": gpt_minimal,
    "plus-mla": plus_mla,
    "plus-compressed": plus_compressed,
    "plus-moe-balanced": plus_moe_balanced,
    "plus-mtp": plus_mtp,
    "full": full,
}

# The intended learning order — each row adds exactly one capability.
LADDER: tuple[str, ...] = (
    "gpt-minimal",
    "plus-mla",
    "plus-compressed",
    "plus-moe-balanced",
    "plus-mtp",
    "full",
)


def load_preset(name: str) -> BabyWhaleV4Config:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]()
