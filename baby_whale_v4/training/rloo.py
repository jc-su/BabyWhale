"""REINFORCE Leave-One-Out.

For each sample i in a group of size K, the baseline is the mean reward of
the OTHER K-1 samples in the group. The advantage A_i = r_i - mean(r_{j≠i})
is unbiased and reduces variance compared to a global mean baseline. With
K=2 it reduces to A_i = r_i - r_{1-i}.

The gradient is the on-policy REINFORCE gradient with this leave-one-out
baseline: log π is recomputed from the *current* policy each step, and there is
no importance ratio — unlike PPO, RLOO does not consume the rollout-time
log-probs (``rloo_step`` isn't even passed them). Because there is exactly one
update per rollout, the current policy *is* the sampling policy, so this is
correct on-policy. KL against a frozen reference policy is optional (educational).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.rl.buffer import SyncRolloutBuffer
from baby_whale_v4.rl.reward_host import LocalRewardHost, RewardHost
from baby_whale_v4.rl.rollout import InProcessRolloutEngine, RolloutEngine
from baby_whale_v4.rl.types import RolloutRequest, RolloutSample, ScoredSample
from baby_whale_v4.training.dpo import _log_softmax, make_reference
from baby_whale_v4.training.grpo import _kl_per_token, _std
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import AdamW, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported
from baby_whale_v4.typing import TokenizerHash, array_to_int_tuple

RewardFn = Callable[[mx.array], float]


@dataclass
class RLOOConfig:
    lr: float = 5e-5
    beta_kl: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    group_size: int = 4
    response_len: int = 8
    max_steps: int = 50
    log_every: int = 10
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 for leave-one-out baseline")
        if self.response_len <= 0:
            raise ValueError("response_len must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.beta_kl < 0:
            raise ValueError("beta_kl must be >= 0")


def _leave_one_out_advantage(rewards: mx.array) -> mx.array:
    """For ``rewards`` of shape ``[G]`` return ``A`` of shape ``[G]`` with
    ``A_i = r_i - mean(r_{j != i})``.

    With G==2 this reduces to ``r_i - r_{1-i}``.
    """

    if rewards.ndim != 1:
        raise ValueError("rewards must be 1D")
    G = rewards.shape[0]
    if G < 2:
        raise ValueError("RLOO requires group_size >= 2")
    total = mx.sum(rewards)
    others_mean = (total - rewards) / (G - 1)
    return rewards - others_mean


def rloo_step(
    *,
    model: BabyWhaleV4Model,
    ref: BabyWhaleV4Model,
    prompt: mx.array,
    samples: mx.array,
    rewards: mx.array,
    beta_kl: float,
) -> mx.array:
    if rewards.ndim != 1 or rewards.shape[0] != samples.shape[0]:
        raise ValueError("rewards must be 1D matching group_size")
    G, R = samples.shape
    advantages = _leave_one_out_advantage(rewards)
    advantages = mx.broadcast_to(advantages[:, None], (G, R))

    full = mx.concatenate([mx.broadcast_to(prompt[None, :], (G, prompt.shape[0])), samples], axis=1)
    out = model(full)
    P = prompt.shape[0]
    pi_logits = out.logits[:, P - 1 : P - 1 + R, :]
    log_pi_all = _log_softmax(pi_logits)
    log_pi = mx.take_along_axis(log_pi_all, samples[:, :, None], axis=-1).squeeze(-1)

    policy_loss = -mx.mean(advantages * log_pi)
    if beta_kl > 0:
        ref_out = ref(full)
        ref_logits = ref_out.logits[:, P - 1 : P - 1 + R, :]
        kl = _kl_per_token(pi_logits, ref_logits)
        return policy_loss + beta_kl * mx.mean(kl)
    return policy_loss


def _default_rollout_engine(model: BabyWhaleV4Model) -> InProcessRolloutEngine:
    return InProcessRolloutEngine(
        model=model,
        config=model.config,
        tokenizer_hash=TokenizerHash(f"rloo-{model.config.config_hash()}"),
    )


def _wrap_reward_fn(reward_fn: RewardFn) -> LocalRewardHost:
    def adapter(sample: RolloutSample) -> float:
        ids = mx.array(list(sample.response_ids), dtype=mx.int32)
        return float(reward_fn(ids))

    return LocalRewardHost(adapter)


def rloo(
    *,
    model: BabyWhaleV4Model,
    prompts: Sequence[mx.array],
    reward_fn: RewardFn | None = None,
    rloo_config: RLOOConfig,
    out_dir: Path | str,
    rollout_engine: RolloutEngine | None = None,
    reward_host: RewardHost | None = None,
) -> BabyWhaleV4Model:
    if reward_fn is None and reward_host is None:
        raise ValueError("rloo requires either reward_fn or reward_host")
    ensure_training_precision_supported(model.config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not prompts:
        raise ValueError("RLOO prompts must be non-empty")
    mx.random.seed(rloo_config.seed)

    engine = rollout_engine if rollout_engine is not None else _default_rollout_engine(model)
    host: RewardHost
    if reward_host is not None:
        host = reward_host
    elif reward_fn is not None:
        host = _wrap_reward_fn(reward_fn)
    else:
        raise ValueError("rloo requires either reward_host or reward_fn")

    ref = make_reference(model)
    optimizer = AdamW(
        learning_rate=rloo_config.lr,
        weight_decay=rloo_config.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(
        m: BabyWhaleV4Model, prompt: mx.array, samples: mx.array, rewards: mx.array
    ) -> mx.array:
        return rloo_step(
            model=m,
            ref=ref,
            prompt=prompt,
            samples=samples,
            rewards=rewards,
            beta_kl=rloo_config.beta_kl,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "rloo_metrics.jsonl")
    options = GenerationOptions(
        max_new_tokens=rloo_config.response_len,
        mode="sample",
        temperature=rloo_config.temperature,
    )
    step = 0
    while step < rloo_config.max_steps:
        for prompt in prompts:
            buffer = SyncRolloutBuffer()
            requests = [
                RolloutRequest(prompt_ids=array_to_int_tuple(prompt), options=options)
                for _ in range(rloo_config.group_size)
            ]
            samples_out = engine.generate_batch(requests)
            for s in samples_out:
                buffer.add(ScoredSample(sample=s, reward=host.score(s)))
            scored = buffer.drain()
            sample_ids = mx.array([list(s.sample.response_ids) for s in scored], dtype=mx.int32)
            rewards = mx.array([s.reward for s in scored], dtype=mx.float32)

            loss, grads = loss_and_grad(model, prompt, sample_ids, rewards)
            if rloo_config.grad_clip > 0:
                grads = clip_grad_norm(grads, rloo_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            engine.sync_weights(model)
            step += 1
            if step % rloo_config.log_every == 0:
                from baby_whale_v4.training.rl_telemetry import policy_telemetry

                tele = policy_telemetry(model=model, ref=ref, prompt=prompt, samples=sample_ids)
                metrics.log(
                    {
                        "step": step,
                        "rloo_loss": float(loss),
                        "reward_mean": float(mx.mean(rewards)),
                        "reward_std": float(_std(rewards)),
                        **tele,
                    }
                )
            if step >= rloo_config.max_steps:
                break
    metrics.close()
    return model
