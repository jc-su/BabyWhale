"""Clipped-importance PPO on top of the typed RolloutSample log-probs.

The policy gradient uses the rollout-time π_old captured by
``inference.Engine.decode_step``. KL against a frozen reference policy keeps
the trained model close to its starting point. This is the policy-only PPO
(no value head); pair it with `grpo_step`-style group-relative advantages or
plug in an external advantage estimator if you add a critic later.
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
class PPOConfig:
    lr: float = 5e-5
    clip_eps: float = 0.2
    beta_kl: float = 0.04
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
            raise ValueError("group_size must be >= 2 for advantage standardization")
        if self.response_len <= 0:
            raise ValueError("response_len must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < self.clip_eps < 1.0):
            raise ValueError("clip_eps must be in (0, 1)")
        if self.beta_kl < 0:
            raise ValueError("beta_kl must be >= 0")


def ppo_step(
    *,
    model: BabyWhaleV4Model,
    ref: BabyWhaleV4Model,
    prompt: mx.array,
    samples: mx.array,
    log_probs_old: mx.array,
    rewards: mx.array,
    clip_eps: float,
    beta_kl: float,
) -> mx.array:
    """One PPO update step. Inputs:

    * ``samples``: ``[G, R]`` int token ids
    * ``log_probs_old``: ``[G, R]`` float, rollout-time π_old per response token
    * ``rewards``: ``[G]`` float, scalar reward per sample
    """

    if rewards.ndim != 1 or rewards.shape[0] != samples.shape[0]:
        raise ValueError("rewards must be 1D matching group_size")
    if log_probs_old.shape != samples.shape:
        raise ValueError(
            f"log_probs_old shape {tuple(log_probs_old.shape)} must match samples "
            f"shape {tuple(samples.shape)}"
        )
    G, R = samples.shape
    advantages = (rewards - mx.mean(rewards)) / (_std(rewards) + 1e-6)
    advantages = mx.broadcast_to(advantages[:, None], (G, R))

    full = mx.concatenate([mx.broadcast_to(prompt[None, :], (G, prompt.shape[0])), samples], axis=1)
    out = model(full)
    P = prompt.shape[0]
    pi_logits = out.logits[:, P - 1 : P - 1 + R, :]
    log_pi_all = _log_softmax(pi_logits)
    log_pi = mx.take_along_axis(log_pi_all, samples[:, :, None], axis=-1).squeeze(-1)

    ratio = mx.exp(log_pi - log_probs_old)
    clipped = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = mx.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -mx.mean(surrogate)

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
        tokenizer_hash=TokenizerHash(f"ppo-{model.config.config_hash()}"),
    )


def _wrap_reward_fn(reward_fn: RewardFn) -> LocalRewardHost:
    def adapter(sample: RolloutSample) -> float:
        ids = mx.array(list(sample.response_ids), dtype=mx.int32)
        return float(reward_fn(ids))

    return LocalRewardHost(adapter)


def ppo(
    *,
    model: BabyWhaleV4Model,
    prompts: Sequence[mx.array],
    reward_fn: RewardFn | None = None,
    ppo_config: PPOConfig,
    out_dir: Path | str,
    rollout_engine: RolloutEngine | None = None,
    reward_host: RewardHost | None = None,
) -> BabyWhaleV4Model:
    """Clipped-IS PPO with rollout-time π_old captured by the rollout engine."""

    if reward_fn is None and reward_host is None:
        raise ValueError("ppo requires either reward_fn or reward_host")
    ensure_training_precision_supported(model.config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not prompts:
        raise ValueError("PPO prompts must be non-empty")
    mx.random.seed(ppo_config.seed)

    engine = rollout_engine if rollout_engine is not None else _default_rollout_engine(model)
    host: RewardHost
    if reward_host is not None:
        host = reward_host
    elif reward_fn is not None:
        host = _wrap_reward_fn(reward_fn)
    else:
        raise ValueError("ppo requires either reward_host or reward_fn")

    ref = make_reference(model)
    optimizer = AdamW(
        learning_rate=ppo_config.lr,
        weight_decay=ppo_config.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(
        m: BabyWhaleV4Model,
        prompt: mx.array,
        samples: mx.array,
        log_probs_old: mx.array,
        rewards: mx.array,
    ) -> mx.array:
        return ppo_step(
            model=m,
            ref=ref,
            prompt=prompt,
            samples=samples,
            log_probs_old=log_probs_old,
            rewards=rewards,
            clip_eps=ppo_config.clip_eps,
            beta_kl=ppo_config.beta_kl,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "ppo_metrics.jsonl")
    options = GenerationOptions(
        max_new_tokens=ppo_config.response_len,
        mode="sample",
        temperature=ppo_config.temperature,
    )
    step = 0
    while step < ppo_config.max_steps:
        for prompt in prompts:
            buffer = SyncRolloutBuffer()
            requests = [
                RolloutRequest(prompt_ids=array_to_int_tuple(prompt), options=options)
                for _ in range(ppo_config.group_size)
            ]
            rollout_samples = engine.generate_batch(requests)
            for s in rollout_samples:
                buffer.add(ScoredSample(sample=s, reward=host.score(s)))
            scored = buffer.drain()
            sample_ids = mx.array([list(s.sample.response_ids) for s in scored], dtype=mx.int32)
            log_probs_old = mx.array([list(s.sample.log_probs) for s in scored], dtype=mx.float32)
            rewards = mx.array([s.reward for s in scored], dtype=mx.float32)

            loss, grads = loss_and_grad(model, prompt, sample_ids, log_probs_old, rewards)
            if ppo_config.grad_clip > 0:
                grads = clip_grad_norm(grads, ppo_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            engine.sync_weights(model)
            step += 1
            if step % ppo_config.log_every == 0:
                from baby_whale_v4.training.rl_telemetry import policy_telemetry

                tele = policy_telemetry(model=model, ref=ref, prompt=prompt, samples=sample_ids)
                metrics.log(
                    {
                        "step": step,
                        "ppo_loss": float(loss),
                        "reward_mean": float(mx.mean(rewards)),
                        "reward_std": float(_std(rewards)),
                        **tele,
                    }
                )
            if step >= ppo_config.max_steps:
                break
    metrics.close()
    return model
