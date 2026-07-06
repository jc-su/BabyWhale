import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.data.dataset import TensorPairDataset, iter_batches
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.training.checkpoint import Checkpoint, save_checkpoint
from baby_whale_v4.training.curriculum import ContextCurriculum
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import Adafactor, AdamW, Muon, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported
from baby_whale_v4.typing import OptimizerKind, SchedulerKind, assert_never, ensure_in

type DatasetBuilder = Callable[[int], TensorPairDataset]

type _Optimizer = AdamW | Adafactor | Muon


@dataclass
class PretrainConfig:
    optimizer: OptimizerKind = "adamw"
    scheduler: SchedulerKind = "constant"
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 1
    batch_size: int = 4
    max_steps: int = 50
    warmup_steps: int = 0
    log_every: int = 10
    save_every: int = 0
    seed: int = 0
    device: str = "mlx"

    def __post_init__(self) -> None:
        ensure_in("optimizer", self.optimizer, ("adamw", "adafactor", "muon"))
        ensure_in("scheduler", self.scheduler, ("constant", "cosine"))
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.save_every < 0:
            raise ValueError("save_every must be >= 0")
        if self.device != "mlx":
            raise ValueError("device must be 'mlx'; runtime is selected separately")


def _resolve_device(requested: str) -> str:
    if requested != "mlx":
        raise ValueError("device must be 'mlx'; runtime is selected separately")
    return "mlx"


def _build_optimizer(ptcfg: PretrainConfig) -> _Optimizer:
    match ptcfg.optimizer:
        case "adamw":
            return AdamW(
                learning_rate=ptcfg.lr,
                betas=ptcfg.betas,
                weight_decay=ptcfg.weight_decay,
            )
        case "adafactor":
            return Adafactor(
                learning_rate=ptcfg.lr,
                beta2=ptcfg.betas[1],
                weight_decay=ptcfg.weight_decay,
            )
        case "muon":
            return Muon(
                learning_rate=ptcfg.lr,
                momentum=ptcfg.betas[0],
                weight_decay=ptcfg.weight_decay,
            )
        case _:
            assert_never(ptcfg.optimizer)


def _lr_at_step(step: int, ptcfg: PretrainConfig) -> float:
    if step < ptcfg.warmup_steps:
        return ptcfg.lr * (step + 1) / max(1, ptcfg.warmup_steps)
    match ptcfg.scheduler:
        case "constant":
            return ptcfg.lr
        case "cosine":
            progress = (step - ptcfg.warmup_steps) / max(1, ptcfg.max_steps - ptcfg.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            floor = ptcfg.lr * ptcfg.min_lr_ratio
            return floor + 0.5 * (ptcfg.lr - floor) * (1.0 + math.cos(math.pi * progress))
        case _:
            assert_never(ptcfg.scheduler)


def _set_lr(opt: _Optimizer, lr: float) -> None:
    opt.learning_rate = lr


@dataclass
class _CosineSchedulerState:
    step: int = 0


class _CosineScheduler:
    def __init__(self, optimizer: _Optimizer, ptcfg: PretrainConfig):
        self.optimizer = optimizer
        self.ptcfg = ptcfg
        self.state = _CosineSchedulerState()

    def step(self) -> None:
        lr = _lr_at_step(self.state.step, self.ptcfg)
        _set_lr(self.optimizer, lr)
        self.state.step += 1

    def state_dict(self) -> dict[str, object]:
        return {"step": self.state.step}

    def load_state_dict(self, sd: dict[str, object]) -> None:
        step = sd["step"]
        if type(step) is not int:
            raise TypeError("scheduler step must be an integer")
        self.state.step = step

    @property
    def current_step(self) -> int:
        return self.state.step


def pretrain(
    *,
    config: BabyWhaleV4Config,
    pretrain_config: PretrainConfig,
    train_dataset: TensorPairDataset,
    out_dir: Path | str,
    eval_dataset: TensorPairDataset | None = None,
    model: BabyWhaleV4Model | None = None,
    resume_from: Checkpoint | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> BabyWhaleV4Model:
    ensure_training_precision_supported(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(pretrain_config.seed)

    if model is None:
        model = BabyWhaleV4Model(config)
    _resolve_device(pretrain_config.device)
    optimizer = _build_optimizer(pretrain_config)
    scheduler = _CosineScheduler(optimizer, pretrain_config)

    start_step = 0
    if resume_from is not None:
        if resume_from.config_hash != config.config_hash():
            raise ValueError(
                f"resume config hash mismatch: ckpt={resume_from.config_hash} vs config={config.config_hash()}"
            )
        model.load_state_dict(resume_from.model_state)
        if resume_from.optimizer_state is not None:
            optimizer.load_state_dict(resume_from.optimizer_state)
        if resume_from.scheduler_state is not None:
            scheduler.load_state_dict(resume_from.scheduler_state)
        if "mlx_seed" in resume_from.rng_state:
            mx.random.seed(resume_from.rng_state["mlx_seed"])
        start_step = resume_from.step

    def loss_fn(m: BabyWhaleV4Model, x: mx.array, y: mx.array) -> mx.array:
        out_obj = m(x, targets=y)
        if out_obj.loss is None:
            raise RuntimeError("model did not return a training loss")
        return out_obj.loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "metrics.jsonl")
    model.train()
    step = start_step
    epoch = 0
    tokens_seen = 0
    train_started = perf_counter()
    while step < pretrain_config.max_steps:
        epoch += 1
        for x, y in iter_batches(
            train_dataset,
            batch_size=pretrain_config.batch_size,
            shuffle=True,
            seed=pretrain_config.seed + epoch,
        ):
            scheduler.step()
            loss, grads = _accumulated_loss_and_grads(
                loss_and_grad,
                model,
                x,
                y,
                accum=pretrain_config.grad_accum,
            )
            if pretrain_config.grad_clip > 0:
                grads = clip_grad_norm(grads, pretrain_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            avg_loss = float(loss)
            tokens_seen += _target_token_count(y)
            elapsed = max(perf_counter() - train_started, 1e-9)
            step += 1

            if on_step is not None:
                on_step(step, avg_loss)
            if step % pretrain_config.log_every == 0:
                metrics.log(
                    {
                        "step": step,
                        "train_loss": avg_loss,
                        "lr": _lr_at_step(step, pretrain_config),
                        "tokens": tokens_seen,
                        "tokens_per_sec": tokens_seen / elapsed,
                    }
                )
            if pretrain_config.save_every and step % pretrain_config.save_every == 0:
                save_checkpoint(
                    out / f"step_{step}.bw4",
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    seed=pretrain_config.seed,
                )
            if step >= pretrain_config.max_steps:
                break

    if eval_dataset is not None:
        eval_loss = _eval_loss(model, eval_dataset, batch_size=pretrain_config.batch_size)
        metrics.log(
            {
                "step": step,
                "eval_loss": eval_loss,
                "eval_tokens": _dataset_token_count(eval_dataset),
            }
        )

    save_checkpoint(
        out / "final.bw4",
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        seed=pretrain_config.seed,
    )
    metrics.close()
    return model


def pretrain_with_curriculum(
    *,
    config: BabyWhaleV4Config,
    pretrain_config: PretrainConfig,
    curriculum: ContextCurriculum,
    build_dataset: DatasetBuilder,
    out_dir: Path | str,
    eval_dataset: TensorPairDataset | None = None,
    model: BabyWhaleV4Model | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> BabyWhaleV4Model:
    """V4-style native long-context pretrain via a context-length curriculum.

    The model is built once at ``curriculum.max_context_length`` so attention,
    RoPE, and the causal mask accommodate every phase. Each phase calls
    ``build_dataset(phase.context_length)`` to materialize a dataset packed at
    that block size, then trains until at least ``phase.n_tokens`` non-pad
    target tokens have been consumed in that phase. The optimizer and
    cosine-LR scheduler persist across phase boundaries so the global LR
    cooldown spans the full schedule — matches V4 §4.2.2 ("training starts
    with a sequence length of 4K, and we gradually extend ... to 16K, 64K,
    and 1M").

    ``pretrain_config.max_steps`` upper-bounds the *total* step count across
    all phases — set it generously; phase-token targets are the primary
    termination criterion.
    """

    if config.context_length != curriculum.max_context_length:
        raise ValueError(
            "config.context_length must equal curriculum.max_context_length "
            f"(got config={config.context_length}, "
            f"curriculum_max={curriculum.max_context_length})"
        )
    ensure_training_precision_supported(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(pretrain_config.seed)

    if model is None:
        model = BabyWhaleV4Model(config)
    _resolve_device(pretrain_config.device)
    optimizer = _build_optimizer(pretrain_config)
    scheduler = _CosineScheduler(optimizer, pretrain_config)

    def loss_fn(m: BabyWhaleV4Model, x: mx.array, y: mx.array) -> mx.array:
        out_obj = m(x, targets=y)
        if out_obj.loss is None:
            raise RuntimeError("model did not return a training loss")
        return out_obj.loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "metrics.jsonl")
    model.train()
    train_started = perf_counter()
    global_step = 0
    tokens_seen_total = 0

    for phase_idx, phase in enumerate(curriculum.phases):
        phase_dataset = build_dataset(phase.context_length)
        tokens_in_phase = 0
        epoch_in_phase = 0
        phase_done = False
        while not phase_done and global_step < pretrain_config.max_steps:
            epoch_in_phase += 1
            for x, y in iter_batches(
                phase_dataset,
                batch_size=pretrain_config.batch_size,
                shuffle=True,
                seed=pretrain_config.seed + 1000 * phase_idx + epoch_in_phase,
            ):
                scheduler.step()
                loss, grads = _accumulated_loss_and_grads(
                    loss_and_grad,
                    model,
                    x,
                    y,
                    accum=pretrain_config.grad_accum,
                )
                if pretrain_config.grad_clip > 0:
                    grads = clip_grad_norm(grads, pretrain_config.grad_clip)
                model.update(optimizer.step(model.parameters(), grads))
                mx.eval(model.parameters())
                avg_loss = float(loss)
                batch_tokens = _target_token_count(y)
                tokens_in_phase += batch_tokens
                tokens_seen_total += batch_tokens
                global_step += 1

                if on_step is not None:
                    on_step(global_step, avg_loss)
                if global_step % pretrain_config.log_every == 0:
                    elapsed = max(perf_counter() - train_started, 1e-9)
                    metrics.log(
                        {
                            "step": global_step,
                            "phase": phase_idx,
                            "phase_context_length": phase.context_length,
                            "phase_tokens": tokens_in_phase,
                            "phase_tokens_target": phase.n_tokens,
                            "tokens_total": tokens_seen_total,
                            "train_loss": avg_loss,
                            "lr": _lr_at_step(global_step, pretrain_config),
                            "tokens_per_sec": tokens_seen_total / elapsed,
                        }
                    )
                if pretrain_config.save_every and global_step % pretrain_config.save_every == 0:
                    save_checkpoint(
                        out / f"step_{global_step}.bw4",
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        seed=pretrain_config.seed,
                    )
                if tokens_in_phase >= phase.n_tokens:
                    phase_done = True
                    break
                if global_step >= pretrain_config.max_steps:
                    break

    if eval_dataset is not None:
        eval_loss = _eval_loss(model, eval_dataset, batch_size=pretrain_config.batch_size)
        metrics.log(
            {
                "step": global_step,
                "eval_loss": eval_loss,
                "eval_tokens": _dataset_token_count(eval_dataset),
            }
        )

    save_checkpoint(
        out / "final.bw4",
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        seed=pretrain_config.seed,
    )
    metrics.close()
    return model


def _accumulated_loss_and_grads(
    loss_and_grad: Callable[
        [BabyWhaleV4Model, mx.array, mx.array], tuple[mx.array, dict[str, object]]
    ],
    model: BabyWhaleV4Model,
    x: mx.array,
    y: mx.array,
    *,
    accum: int,
) -> tuple[mx.array, dict[str, object]]:
    if accum <= 0:
        raise ValueError("accum must be positive")
    micro_count = min(accum, int(x.shape[0]))
    microbatches = list(_split_microbatches(x, y, micro_count))
    token_counts = [_target_token_count(micro_y) for _micro_x, micro_y in microbatches]
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise ValueError("gradient accumulation requires at least one non-ignored target token")
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_grads: dict[str, object] | None = None
    for (micro_x, micro_y), token_count in zip(microbatches, token_counts, strict=True):
        loss, grads = loss_and_grad(model, micro_x, micro_y)
        scale = token_count / total_tokens
        scaled_grads = _scale_grad_tree(grads, scale)
        total_grads = (
            scaled_grads if total_grads is None else _add_grad_trees(total_grads, scaled_grads)
        )
        total_loss = total_loss + loss.astype(mx.float32) * scale
        mx.eval(total_loss, total_grads)
    if total_grads is None:
        raise RuntimeError("gradient accumulation produced no microbatches")
    return total_loss, total_grads


def _target_token_count(y: mx.array) -> int:
    if not isinstance(y, mx.array):
        raise TypeError("targets must be an MLX array")
    return int(mx.sum(mx.not_equal(y, -1)))


def _dataset_token_count(dataset: TensorPairDataset) -> int:
    total = 0
    for idx in range(len(dataset)):
        _x, y = dataset[idx]
        total += _target_token_count(y)
    return total


def _scale_grad_tree(grads: dict[str, object], scale: float) -> dict[str, object]:
    return tree_map(lambda grad: grad * scale if isinstance(grad, mx.array) else grad, grads)


def _add_grad_trees(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    def add(left_value: object, right_value: object) -> object:
        if isinstance(left_value, mx.array) and isinstance(right_value, mx.array):
            return left_value + right_value
        if isinstance(left_value, mx.array) or isinstance(right_value, mx.array):
            raise TypeError("gradient trees have mismatched MLX array leaves")
        return left_value

    return tree_map(add, left, right)


def _split_microbatches(
    x: mx.array, y: mx.array, accum: int
) -> Iterable[tuple[mx.array, mx.array]]:
    if accum == 1:
        yield x, y
        return
    n = int(x.shape[0])
    n_chunks = min(accum, n)
    base = n // n_chunks
    remainder = n % n_chunks
    start = 0
    for chunk_idx in range(n_chunks):
        size = base + (1 if chunk_idx < remainder else 0)
        end = start + size
        yield x[start:end], y[start:end]
        start = end


def _eval_loss(model: BabyWhaleV4Model, dataset: TensorPairDataset, *, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for x, y in iter_batches(dataset, batch_size=batch_size, shuffle=False):
        out = model(x, targets=y)
        if out.loss is None:
            raise RuntimeError("model did not return an eval loss")
        total += float(out.loss) * int(x.shape[0])
        n += int(x.shape[0])
    if was_training:
        model.train()
    if n == 0:
        return float("nan")
    return total / n
