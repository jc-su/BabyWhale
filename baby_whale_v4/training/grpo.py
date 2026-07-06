from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from jaxtyping import Float, Int

from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.rl.buffer import SyncRolloutBuffer
from baby_whale_v4.rl.reward_host import LocalRewardHost, RewardHost
from baby_whale_v4.rl.rollout import InProcessRolloutEngine, RolloutEngine
from baby_whale_v4.rl.types import RolloutRequest, RolloutSample, ScoredSample
from baby_whale_v4.training.dpo import _log_softmax, make_reference
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import AdamW, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported
from baby_whale_v4.typing import TokenizerHash, array_to_int_tuple

RewardFn = Callable[[mx.array], float]


@dataclass
class GRPOConfig:
    lr: float = 5e-5
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


def _kl_per_token(
    pi_logits: Float[mx.array, "G R V"], ref_logits: Float[mx.array, "G R V"]
) -> Float[mx.array, "G R"]:
    """KL(pi || ref) per token."""

    log_pi = _log_softmax(pi_logits)
    log_ref = _log_softmax(ref_logits)
    pi = mx.exp(log_pi)
    return mx.sum(pi * (log_pi - log_ref), axis=-1)


def _std(x: Float[mx.array, "G"]) -> Float[mx.array, ""]:
    centered = x - mx.mean(x)
    return mx.sqrt(mx.mean(mx.square(centered)))


def _samples_to_arrays(scored: Sequence[ScoredSample]) -> tuple[mx.array, mx.array]:
    samples = mx.array([list(item.sample.response_ids) for item in scored], dtype=mx.int32)
    rewards = mx.array([item.reward for item in scored], dtype=mx.float32)
    return samples, rewards


def grpo_step(
    *,
    model: BabyWhaleV4Model,
    ref: BabyWhaleV4Model,
    prompt: Int[mx.array, "P"],
    samples: Int[mx.array, "G R"],
    rewards: Float[mx.array, "G"],
    beta_kl: float,
) -> Float[mx.array, ""]:
    """One GRPO update pass for a single prompt.

    ``G`` is group_size, ``R`` is response_len, ``P`` is prompt_len, ``V`` is
    vocab_size. The returned scalar is ``policy_loss + beta_kl * mean_kl``.
    """

    if rewards.ndim != 1 or rewards.shape[0] != samples.shape[0]:
        raise ValueError("rewards must be 1D matching group_size")
    G, R = samples.shape
    advantages = (rewards - mx.mean(rewards)) / (_std(rewards) + 1e-6)

    full = mx.concatenate([mx.broadcast_to(prompt[None, :], (G, prompt.shape[0])), samples], axis=1)
    out = model(full)
    P = prompt.shape[0]
    pi_logits = out.logits[:, P - 1 : P - 1 + R, :]
    log_pi_all = _log_softmax(pi_logits)
    log_pi = mx.take_along_axis(log_pi_all, samples[:, :, None], axis=-1).squeeze(-1)

    ref_out = ref(full)
    ref_logits = ref_out.logits[:, P - 1 : P - 1 + R, :]

    kl = _kl_per_token(pi_logits, ref_logits)
    policy_loss = -mx.mean(advantages[:, None] * log_pi)
    kl_loss = beta_kl * mx.mean(kl)
    return policy_loss + kl_loss


def _default_rollout_engine(model: BabyWhaleV4Model) -> InProcessRolloutEngine:
    return InProcessRolloutEngine(
        model=model,
        config=model.config,
        tokenizer_hash=TokenizerHash(f"grpo-{model.config.config_hash()}"),
    )


def _wrap_reward_fn(reward_fn: RewardFn) -> LocalRewardHost:
    def adapter(sample: RolloutSample) -> float:
        ids = mx.array(list(sample.response_ids), dtype=mx.int32)
        return float(reward_fn(ids))

    return LocalRewardHost(adapter)


def grpo(
    *,
    model: BabyWhaleV4Model,
    prompts: Sequence[mx.array],
    reward_fn: RewardFn | None = None,
    grpo_config: GRPOConfig,
    out_dir: Path | str,
    rollout_engine: RolloutEngine | None = None,
    reward_host: RewardHost | None = None,
    request_metadata: Sequence[Mapping[str, str]] | None = None,
) -> BabyWhaleV4Model:
    """Group Relative Policy Optimization with a typed rollout boundary.

    The rollout side runs through `rollout_engine.generate_batch(...)` and the
    reward side through `reward_host.score_batch(...)`. Defaults are an
    in-process MLX engine wrapping `inference.Engine` (so prefix-caching,
    chunked prefill, and per-token log-prob capture come for free) and a local
    callable host wrapping the legacy `reward_fn` argument.

    ``request_metadata`` (optional) is a list of dicts the same length as
    ``prompts``; entry ``i`` is attached as ``RolloutRequest.metadata`` for
    every member of the group sampled from prompt ``i``. Reward hosts that
    need per-prompt context (e.g. ``CodeRewardHost`` looking up tests by
    ``problem_id``) read it from this metadata.
    """

    if reward_fn is None and reward_host is None:
        raise ValueError("grpo requires either reward_fn or reward_host")
    ensure_training_precision_supported(model.config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not prompts:
        raise ValueError("GRPO prompts must be non-empty")
    if request_metadata is not None and len(request_metadata) != len(prompts):
        raise ValueError(
            f"request_metadata length {len(request_metadata)} must match prompts length {len(prompts)}"
        )
    mx.random.seed(grpo_config.seed)

    engine = rollout_engine if rollout_engine is not None else _default_rollout_engine(model)
    host: RewardHost
    if reward_host is not None:
        host = reward_host
    elif reward_fn is not None:
        host = _wrap_reward_fn(reward_fn)
    else:
        raise ValueError("grpo requires either reward_host or reward_fn")

    ref = make_reference(model)
    optimizer = AdamW(
        learning_rate=grpo_config.lr,
        weight_decay=grpo_config.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(
        m: BabyWhaleV4Model, prompt: mx.array, samples: mx.array, rewards: mx.array
    ) -> mx.array:
        return grpo_step(
            model=m,
            ref=ref,
            prompt=prompt,
            samples=samples,
            rewards=rewards,
            beta_kl=grpo_config.beta_kl,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "grpo_metrics.jsonl")
    options = GenerationOptions(
        max_new_tokens=grpo_config.response_len,
        mode="sample",
        temperature=grpo_config.temperature,
    )
    step = 0
    while step < grpo_config.max_steps:
        for idx, prompt in enumerate(prompts):
            meta: Mapping[str, str] = request_metadata[idx] if request_metadata is not None else {}
            buffer = SyncRolloutBuffer()
            requests = [
                RolloutRequest(
                    prompt_ids=array_to_int_tuple(prompt),
                    options=options,
                    metadata=meta,
                )
                for _ in range(grpo_config.group_size)
            ]
            rollout_samples = engine.generate_batch(requests)
            scored = [ScoredSample(sample=s, reward=host.score(s)) for s in rollout_samples]
            buffer.add_many(scored)
            sample_ids, rewards = _samples_to_arrays(buffer.drain())
            loss, grads = loss_and_grad(model, prompt, sample_ids, rewards)
            if grpo_config.grad_clip > 0:
                grads = clip_grad_norm(grads, grpo_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            engine.sync_weights(model)
            step += 1
            if step % grpo_config.log_every == 0:
                from baby_whale_v4.training.rl_telemetry import policy_telemetry

                tele = policy_telemetry(model=model, ref=ref, prompt=prompt, samples=sample_ids)
                metrics.log(
                    {
                        "step": step,
                        "grpo_loss": float(loss),
                        "reward_mean": float(mx.mean(rewards)),
                        "reward_std": float(_std(rewards)),
                        **tele,
                    }
                )
            if step >= grpo_config.max_steps:
                break
    metrics.close()
    return model


def rejection_finetune_collect(
    *,
    model: BabyWhaleV4Model,
    prompt: mx.array,
    n_samples: int,
    response_len: int,
    reward_fn: RewardFn,
    keep_top: int,
    temperature: float = 1.0,
    rollout_engine: RolloutEngine | None = None,
) -> list[mx.array]:
    """Sample, score, and return the top-k completions by reward."""

    if keep_top <= 0 or keep_top > n_samples:
        raise ValueError("keep_top must be in (0, n_samples]")
    engine = rollout_engine if rollout_engine is not None else _default_rollout_engine(model)
    options = GenerationOptions(max_new_tokens=response_len, mode="sample", temperature=temperature)
    requests = [
        RolloutRequest(prompt_ids=array_to_int_tuple(prompt), options=options)
        for _ in range(n_samples)
    ]
    samples = engine.generate_batch(requests)
    scored: list[tuple[float, mx.array]] = []
    for sample in samples:
        ids = mx.array(list(sample.response_ids), dtype=mx.int32)
        scored.append((float(reward_fn(ids)), ids))
    scored.sort(reverse=True, key=lambda t: t[0])
    return [ids for _, ids in scored[:keep_top]]
