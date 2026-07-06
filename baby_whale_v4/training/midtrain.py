from dataclasses import dataclass
from pathlib import Path

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.data.dataset import TensorPairDataset
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.training.checkpoint import Checkpoint
from baby_whale_v4.training.pretrain import PretrainConfig, pretrain


@dataclass
class MidtrainConfig(PretrainConfig):
    stage_name: str = "midtrain"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_name != "midtrain":
            raise ValueError("stage_name must be 'midtrain'")


def midtrain(
    *,
    config: BabyWhaleV4Config,
    midtrain_config: MidtrainConfig,
    train_dataset: TensorPairDataset,
    out_dir: Path | str,
    eval_dataset: TensorPairDataset | None = None,
    model: BabyWhaleV4Model | None = None,
    resume_from: Checkpoint | None = None,
) -> BabyWhaleV4Model:
    return pretrain(
        config=config,
        pretrain_config=PretrainConfig(
            optimizer=midtrain_config.optimizer,
            scheduler=midtrain_config.scheduler,
            lr=midtrain_config.lr,
            min_lr_ratio=midtrain_config.min_lr_ratio,
            betas=midtrain_config.betas,
            weight_decay=midtrain_config.weight_decay,
            grad_clip=midtrain_config.grad_clip,
            grad_accum=midtrain_config.grad_accum,
            batch_size=midtrain_config.batch_size,
            max_steps=midtrain_config.max_steps,
            warmup_steps=midtrain_config.warmup_steps,
            log_every=midtrain_config.log_every,
            save_every=midtrain_config.save_every,
            seed=midtrain_config.seed,
            device=midtrain_config.device,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        out_dir=out_dir,
        model=model,
        resume_from=resume_from,
    )
