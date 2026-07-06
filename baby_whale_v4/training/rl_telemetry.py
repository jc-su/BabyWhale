"""Shared policy telemetry for RL trainers.

Computes KL(pi || ref) and entropy(pi) on rolled-out samples as a diagnostic.
Kept separate from the gradient path so GRPO/PPO/RLOO loss functions stay
focused on the policy-gradient term.
"""

from __future__ import annotations

import mlx.core as mx
from jaxtyping import Float, Int

from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.training.dpo import _log_softmax


def _kl_per_token(
    pi_logits: Float[mx.array, "G R V"], ref_logits: Float[mx.array, "G R V"]
) -> Float[mx.array, "G R"]:
    log_pi = _log_softmax(pi_logits)
    log_ref = _log_softmax(ref_logits)
    pi = mx.exp(log_pi)
    return mx.sum(pi * (log_pi - log_ref), axis=-1)


def policy_telemetry(
    *,
    model: BabyWhaleV4Model,
    ref: BabyWhaleV4Model,
    prompt: Int[mx.array, "P"],
    samples: Int[mx.array, "G R"],
) -> dict[str, float]:
    """Compute (kl_mean, entropy_mean, response_len_mean) for `samples`.

    ``prompt`` is 1-D ``[P]``, ``samples`` is 2-D ``[G, R]``. No gradients are
    taken; this is for metrics emission only.
    """
    if prompt.ndim != 1:
        raise ValueError("prompt must be 1D")
    if samples.ndim != 2:
        raise ValueError("samples must be 2D [G, R]")
    G, R = samples.shape
    P = prompt.shape[0]
    full = mx.concatenate([mx.broadcast_to(prompt[None, :], (G, P)), samples], axis=1)
    pi_logits = model(full).logits[:, P - 1 : P - 1 + R, :]
    ref_logits = ref(full).logits[:, P - 1 : P - 1 + R, :]
    kl_per_tok = _kl_per_token(pi_logits, ref_logits)
    log_pi = _log_softmax(pi_logits)
    pi_probs = mx.exp(log_pi)
    entropy_per_tok = -mx.sum(pi_probs * log_pi, axis=-1)
    return {
        "kl_mean": float(mx.mean(kl_per_tok)),
        "entropy_mean": float(mx.mean(entropy_per_tok)),
        "response_len_mean": float(R),
    }
