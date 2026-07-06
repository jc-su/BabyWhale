"""On-policy distillation.

The student generates rollouts itself (so the gradient signal is on the
student's own distribution, not on the teacher's), the teacher provides
soft labels (logits) for each accepted sample, and the loss is per-token
KL divergence between the teacher and student distributions.

Reward filtering plugs in via the existing :class:`RewardHost` boundary —
only samples whose scalar reward meets ``reward_threshold`` contribute to
the loss. This matches the DeepSeek-V4 recipe of "specialists generate
samples, filter by verifiable reward, distill into one consolidated
checkpoint" (Plan §11).
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
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import AdamW, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported
from baby_whale_v4.typing import TokenizerHash, array_to_int_tuple

RewardFn = Callable[[mx.array], float]


@dataclass
class DistillConfig:
    lr: float = 5e-5
    teacher_temperature: float = 1.0
    student_temperature: float = 1.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    group_size: int = 4
    response_len: int = 8
    max_steps: int = 50
    log_every: int = 10
    temperature: float = 1.0
    reward_threshold: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.response_len <= 0:
            raise ValueError("response_len must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.teacher_temperature <= 0 or self.student_temperature <= 0:
            raise ValueError("distillation temperatures must be positive")


def kl_divergence(teacher_logits: mx.array, student_logits: mx.array) -> mx.array:
    """Per-position KL(teacher || student). Inputs ``[..., V]`` logits."""

    log_teacher = _log_softmax(teacher_logits)
    log_student = _log_softmax(student_logits)
    teacher = mx.exp(log_teacher)
    return mx.sum(teacher * (log_teacher - log_student), axis=-1)


def distill_step(
    *,
    student: BabyWhaleV4Model,
    teacher: BabyWhaleV4Model,
    prompt: mx.array,
    samples: mx.array,
    accept_mask: mx.array,
    teacher_temperature: float,
    student_temperature: float,
) -> mx.array:
    """One distillation update step. Returns scalar KL loss averaged over
    accepted (sample, position) pairs.

    * ``samples``: ``[G, R]`` int token ids
    * ``accept_mask``: ``[G]`` bool, ``True`` for samples with reward >= threshold
    """

    if samples.ndim != 2:
        raise ValueError("samples must be [G, R]")
    if accept_mask.ndim != 1 or accept_mask.shape[0] != samples.shape[0]:
        raise ValueError("accept_mask must be [G] matching samples")

    G, R = samples.shape
    full = mx.concatenate([mx.broadcast_to(prompt[None, :], (G, prompt.shape[0])), samples], axis=1)
    P = prompt.shape[0]

    # Student forward — gradients flow through this branch.
    student_logits = student(full).logits[:, P - 1 : P - 1 + R, :] / student_temperature

    # Teacher forward — frozen reference, no gradient.
    teacher_logits = teacher(full).logits[:, P - 1 : P - 1 + R, :] / teacher_temperature
    teacher_logits = mx.stop_gradient(teacher_logits)

    kl_per_token = kl_divergence(teacher_logits, student_logits)  # [G, R]

    # Mask to accepted samples only; the trainer skips zero-accepted batches.
    mask = accept_mask.astype(student_logits.dtype)[:, None]  # [G, 1]
    weighted = kl_per_token * mask
    denom = mx.maximum(mx.sum(mask) * R, 1.0)
    return mx.sum(weighted) / denom


def _default_rollout_engine(student: BabyWhaleV4Model) -> InProcessRolloutEngine:
    return InProcessRolloutEngine(
        model=student,
        config=student.config,
        tokenizer_hash=TokenizerHash(f"distill-{student.config.config_hash()}"),
    )


def _wrap_reward_fn(reward_fn: RewardFn) -> LocalRewardHost:
    def adapter(sample: RolloutSample) -> float:
        ids = mx.array(list(sample.response_ids), dtype=mx.int32)
        return float(reward_fn(ids))

    return LocalRewardHost(adapter)


def distill(
    *,
    student: BabyWhaleV4Model,
    teacher: BabyWhaleV4Model | None = None,
    prompts: Sequence[mx.array],
    reward_fn: RewardFn | None = None,
    distill_config: DistillConfig,
    out_dir: Path | str,
    rollout_engine: RolloutEngine | None = None,
    reward_host: RewardHost | None = None,
) -> BabyWhaleV4Model:
    """On-policy distillation: student generates, teacher labels.

    If ``teacher`` is ``None`` a frozen copy of the student at this entry point
    is used (self-distillation; useful as a regression baseline). Reward
    filtering is opt-in: pass ``reward_fn`` or ``reward_host`` to drop samples
    below ``distill_config.reward_threshold``. Without either, every sample
    contributes.
    """

    ensure_training_precision_supported(student.config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not prompts:
        raise ValueError("distill prompts must be non-empty")
    mx.random.seed(distill_config.seed)

    engine = rollout_engine if rollout_engine is not None else _default_rollout_engine(student)
    if teacher is None:
        teacher = make_reference(student)
    if teacher is student:
        raise ValueError("teacher must be a distinct model from student")

    host: RewardHost | None
    if reward_host is not None:
        host = reward_host
    elif reward_fn is not None:
        host = _wrap_reward_fn(reward_fn)
    else:
        host = None

    optimizer = AdamW(
        learning_rate=distill_config.lr,
        weight_decay=distill_config.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(
        m: BabyWhaleV4Model,
        prompt: mx.array,
        samples: mx.array,
        accept_mask: mx.array,
    ) -> mx.array:
        return distill_step(
            student=m,
            teacher=teacher,
            prompt=prompt,
            samples=samples,
            accept_mask=accept_mask,
            teacher_temperature=distill_config.teacher_temperature,
            student_temperature=distill_config.student_temperature,
        )

    loss_and_grad = nn.value_and_grad(student, loss_fn)
    metrics = JsonlMetrics(out / "distill_metrics.jsonl")
    options = GenerationOptions(
        max_new_tokens=distill_config.response_len,
        mode="sample",
        temperature=distill_config.temperature,
    )
    step = 0
    attempts = 0
    max_attempts = max(distill_config.max_steps * 4, 20)
    while step < distill_config.max_steps and attempts < max_attempts:
        for prompt in prompts:
            attempts += 1
            buffer = SyncRolloutBuffer()
            requests = [
                RolloutRequest(prompt_ids=array_to_int_tuple(prompt), options=options)
                for _ in range(distill_config.group_size)
            ]
            samples_out = engine.generate_batch(requests)
            if host is not None:
                for s in samples_out:
                    buffer.add(ScoredSample(sample=s, reward=host.score(s)))
            else:
                for s in samples_out:
                    buffer.add(ScoredSample(sample=s, reward=0.0))
            scored = buffer.drain()
            sample_ids = mx.array([list(s.sample.response_ids) for s in scored], dtype=mx.int32)
            rewards = [s.reward for s in scored]
            if host is None:
                accept = [True for _ in scored]
            else:
                accept = [r >= distill_config.reward_threshold for r in rewards]
            n_accepted = sum(1 for v in accept if v)
            if n_accepted == 0:
                metrics.log(
                    {
                        "attempt": attempts,
                        "step": step,
                        "skipped": True,
                        "reason": "no accepted samples",
                    }
                )
                if attempts >= max_attempts:
                    break
                continue
            accept_mask = mx.array(accept, dtype=mx.bool_)

            loss, grads = loss_and_grad(student, prompt, sample_ids, accept_mask)
            if distill_config.grad_clip > 0:
                grads = clip_grad_norm(grads, distill_config.grad_clip)
            student.update(optimizer.step(student.parameters(), grads))
            mx.eval(student.parameters())
            engine.sync_weights(student)
            step += 1
            if step % distill_config.log_every == 0:
                metrics.log(
                    {
                        "step": step,
                        "kl_loss": float(loss),
                        "n_accepted": n_accepted,
                        "n_total": len(scored),
                        "reward_mean": (float(sum(rewards) / len(rewards)) if rewards else 0.0),
                    }
                )
            if step >= distill_config.max_steps:
                break
    metrics.close()
    return student
