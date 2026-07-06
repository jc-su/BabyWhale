from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.data.dataset import TensorPairDataset, iter_batches
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.training.metrics import JsonlMetrics
from baby_whale_v4.training.mlx_optim import AdamW, clip_grad_norm
from baby_whale_v4.training.precision import ensure_training_precision_supported
from baby_whale_v4.training.pretrain import _resolve_device


@dataclass
class SFTConfig:
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 4
    max_steps: int = 100
    log_every: int = 20
    seed: int = 0
    device: str = "mlx"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.device != "mlx":
            raise ValueError("device must be 'mlx'; runtime is selected separately")


def sft(
    *,
    config: BabyWhaleV4Config,
    sft_config: SFTConfig,
    train_dataset: TensorPairDataset,
    out_dir: Path | str,
    model: BabyWhaleV4Model | None = None,
) -> BabyWhaleV4Model:
    ensure_training_precision_supported(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(sft_config.seed)
    if model is None:
        model = BabyWhaleV4Model(config)
    _resolve_device(sft_config.device)
    optimizer = AdamW(
        learning_rate=sft_config.lr,
        weight_decay=sft_config.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(m: BabyWhaleV4Model, x: mx.array, y: mx.array) -> mx.array:
        out_obj = m(x, targets=y)
        if out_obj.loss is None:
            raise RuntimeError("model did not return SFT loss")
        return out_obj.loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics = JsonlMetrics(out / "sft_metrics.jsonl")
    model.train()
    step = 0
    epoch = 0
    while step < sft_config.max_steps:
        epoch += 1
        for x, y in iter_batches(
            train_dataset,
            batch_size=sft_config.batch_size,
            shuffle=True,
            seed=sft_config.seed + epoch,
        ):
            loss, grads = loss_and_grad(model, x, y)
            if not bool(mx.all(mx.isfinite(loss))):
                raise RuntimeError(f"non-finite SFT loss at step {step}")
            if sft_config.grad_clip > 0:
                grads = clip_grad_norm(grads, sft_config.grad_clip)
            model.update(optimizer.step(model.parameters(), grads))
            mx.eval(model.parameters())
            step += 1
            if step % sft_config.log_every == 0:
                metrics.log({"step": step, "sft_loss": float(loss)})
            if step >= sft_config.max_steps:
                break
    metrics.close()
    return model
