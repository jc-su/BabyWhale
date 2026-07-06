import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn
from jaxtyping import Float, Int

from baby_whale_v4.data.tokenizer import Tokenizer
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import AdamW, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported


@dataclass(frozen=True)
class DPOExample:
    prompt: mx.array
    chosen: mx.array
    rejected: mx.array

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt", self.prompt),
            ("chosen", self.chosen),
            ("rejected", self.rejected),
        ):
            if not isinstance(value, mx.array):
                raise TypeError(f"DPOExample.{name} must be an MLX array")
            if value.ndim != 1 or value.shape[0] <= 0:
                raise ValueError(f"DPOExample.{name} must be a non-empty 1D array")


@dataclass
class DPOConfig:
    lr: float = 1e-4
    beta: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 2
    max_steps: int = 50
    log_every: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


def _log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _log_sigmoid(x: Float[mx.array, "B"]) -> Float[mx.array, "B"]:
    return -mx.logaddexp(mx.array(0.0, dtype=x.dtype), -x)


def _logp_response(
    model: BabyWhaleV4Model,
    prompt: Int[mx.array, "B P"],
    response: Int[mx.array, "B R"],
    response_mask: Float[mx.array, "B R"] | None = None,
) -> Float[mx.array, "B"]:
    """Sum log probability of ``response`` given ``prompt``."""

    if prompt.ndim != 2 or response.ndim != 2:
        raise ValueError("prompt and response must be 2D [B, T]")
    if prompt.shape[0] != response.shape[0]:
        raise ValueError("batch dim mismatch")
    if response_mask is not None and response_mask.shape != response.shape:
        raise ValueError("response_mask shape must match response")
    full = mx.concatenate([prompt, response], axis=1)
    out = model(full)
    P = prompt.shape[1]
    R = response.shape[1]
    logits = out.logits[:, P - 1 : P - 1 + R, :]
    logp = _log_softmax(logits)
    chosen = mx.take_along_axis(logp, response[:, :, None], axis=-1).squeeze(-1)
    if response_mask is not None:
        chosen = chosen * response_mask.astype(chosen.dtype)
    return mx.sum(chosen, axis=-1)


def _dpo_pair_loss(
    model: BabyWhaleV4Model,
    prompt: Int[mx.array, "B P"],
    chosen: Int[mx.array, "B R"],
    rejected: Int[mx.array, "B R"],
    *,
    beta: float,
    ref_logratio: Float[mx.array, "B"],
    chosen_mask: Float[mx.array, "B R"] | None = None,
    rejected_mask: Float[mx.array, "B R"] | None = None,
) -> Float[mx.array, ""]:
    """DPO loss given a *precomputed* reference log-ratio.

    The reference is frozen, so ``ref_logratio`` is constant across steps (see
    :func:`_precompute_ref_logratios`); accepting it here avoids re-running the
    reference forward passes inside the traced training loss.
    """
    pi_chosen = _logp_response(model, prompt, chosen, chosen_mask)
    pi_rejected = _logp_response(model, prompt, rejected, rejected_mask)
    pi_logratio = pi_chosen - pi_rejected
    logits = beta * (pi_logratio - ref_logratio)
    return -mx.mean(_log_sigmoid(logits))


def dpo_loss(
    model: BabyWhaleV4Model,
    ref_model: BabyWhaleV4Model,
    prompt: Int[mx.array, "B P"],
    chosen: Int[mx.array, "B R"],
    rejected: Int[mx.array, "B R"],
    *,
    beta: float,
    chosen_mask: Float[mx.array, "B R"] | None = None,
    rejected_mask: Float[mx.array, "B R"] | None = None,
) -> Float[mx.array, ""]:
    ref_chosen = _logp_response(ref_model, prompt, chosen, chosen_mask)
    ref_rejected = _logp_response(ref_model, prompt, rejected, rejected_mask)
    return _dpo_pair_loss(
        model,
        prompt,
        chosen,
        rejected,
        beta=beta,
        ref_logratio=ref_chosen - ref_rejected,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )


def _precompute_ref_logratios(ref: BabyWhaleV4Model, examples: list[DPOExample]) -> list[mx.array]:
    """Reference log-ratios (ref_chosen - ref_rejected), one per example.

    The reference model is frozen, so each example's log-ratio is constant across
    DPO steps. Computing them once here — instead of two reference forwards per
    example per step inside the traced loss — halves the forward count and is
    numerically identical.
    """
    ratios: list[mx.array] = []
    for ex in examples:
        ref_chosen = _logp_response(ref, ex.prompt[None, :], ex.chosen[None, :])
        ref_rejected = _logp_response(ref, ex.prompt[None, :], ex.rejected[None, :])
        ratios.append(ref_chosen - ref_rejected)
    mx.eval(ratios)
    return ratios


def _batch_dpo_loss_fn(
    *,
    batch: tuple[DPOExample, ...],
    ref_logratios: tuple[mx.array, ...],
    beta: float,
) -> Callable[[BabyWhaleV4Model], mx.array]:
    if not batch:
        raise ValueError("DPO batch must be non-empty")

    def loss_fn(m: BabyWhaleV4Model) -> mx.array:
        losses = [
            _dpo_pair_loss(
                m,
                ex.prompt[None, :],
                ex.chosen[None, :],
                ex.rejected[None, :],
                beta=beta,
                ref_logratio=ref_lr,
            )
            for ex, ref_lr in zip(batch, ref_logratios, strict=True)
        ]
        total = sum(losses, mx.array(0.0, dtype=losses[0].dtype))
        return total / len(losses)

    return loss_fn


def make_reference(model: BabyWhaleV4Model) -> BabyWhaleV4Model:
    ref = copy.deepcopy(model)
    ref.eval()
    return ref


def dpo_examples_from_jsonl(
    path: Path | str,
    tokenizer: Tokenizer,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> list[DPOExample]:
    if max_prompt_tokens <= 0 or max_response_tokens <= 0:
        raise ValueError("max_prompt_tokens and max_response_tokens must be positive")
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    examples: list[DPOExample] = []
    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{line_no}: invalid JSONL row: {exc}") from exc
            examples.append(
                _dpo_example_from_record(
                    raw,
                    tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    max_response_tokens=max_response_tokens,
                    source=f"{src}:{line_no}",
                )
            )
    if not examples:
        raise ValueError(f"{src} produced zero DPO examples")
    return examples


def dpo(
    *,
    model: BabyWhaleV4Model,
    examples: list[DPOExample],
    dpo_config: DPOConfig,
    out_dir: Path | str,
) -> BabyWhaleV4Model:
    ensure_training_precision_supported(model.config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not examples:
        raise ValueError("DPO examples must be non-empty")
    mx.random.seed(dpo_config.seed)
    ref = make_reference(model)
    # The reference is frozen: precompute its per-example log-ratios once.
    ref_logratios = _precompute_ref_logratios(ref, examples)
    optimizer = AdamW(
        learning_rate=dpo_config.lr,
        weight_decay=dpo_config.weight_decay,
        betas=(0.9, 0.95),
    )

    metrics = JsonlMetrics(out / "dpo_metrics.jsonl")
    step = 0
    bs = dpo_config.batch_size
    while step < dpo_config.max_steps:
        for start in range(0, len(examples), bs):
            batch = tuple(examples[start : start + bs])
            if not batch:
                continue
            batch_ratios = tuple(ref_logratios[start : start + bs])
            loss_fn = _batch_dpo_loss_fn(
                batch=batch, ref_logratios=batch_ratios, beta=dpo_config.beta
            )
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            if dpo_config.grad_clip > 0:
                grads = clip_grad_norm(grads, dpo_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            step += 1
            if step % dpo_config.log_every == 0:
                metrics.log({"step": step, "dpo_loss": float(loss)})
            if step >= dpo_config.max_steps:
                break
    metrics.close()
    return model


def _dpo_example_from_record(
    raw: object,
    tokenizer: Tokenizer,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    source: str,
) -> DPOExample:
    if not _is_str_mapping(raw):
        raise TypeError(f"{source} must be an object with string keys")
    if raw.get("kind") != "preference":
        raise ValueError(f"{source}.kind must be 'preference'")
    prompt = _read_nonempty_str(raw, "prompt", source)
    chosen = _read_nonempty_str(raw, "chosen", source)
    rejected = _read_nonempty_str(raw, "rejected", source)
    return DPOExample(
        prompt=mx.array(
            _truncate_nonempty(tokenizer.encode(prompt, add_bos=True), max_prompt_tokens),
            dtype=mx.int32,
        ),
        chosen=mx.array(
            _truncate_nonempty(tokenizer.encode(chosen, add_eos=True), max_response_tokens),
            dtype=mx.int32,
        ),
        rejected=mx.array(
            _truncate_nonempty(tokenizer.encode(rejected, add_eos=True), max_response_tokens),
            dtype=mx.int32,
        ),
    )


def _truncate_nonempty(ids: list[int], length: int) -> list[int]:
    if not ids:
        raise ValueError("encoded DPO field must not be empty")
    return ids[:length]


def _read_nonempty_str(raw: Mapping[str, object], key: str, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{key} must be a non-empty string")
    return value


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)
