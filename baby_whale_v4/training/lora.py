from collections.abc import Iterator
from dataclasses import dataclass

from baby_whale_v4.layers import WhaleLinear
from baby_whale_v4.typing import LinearPlacement, ensure_in

_PLACEMENTS: tuple[LinearPlacement, ...] = (
    "general",
    "attention",
    "moe_expert",
    "router",
    "lm_head",
    "mtp",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    init_scale: float = 0.01
    placements: tuple[LinearPlacement, ...] = ("attention", "moe_expert")

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if self.init_scale <= 0:
            raise ValueError("LoRA init_scale must be positive")
        if not self.placements:
            raise ValueError("LoRA placements must be non-empty")
        for placement in self.placements:
            ensure_in("LoRA placement", placement, _PLACEMENTS)


@dataclass(frozen=True)
class LoRAAttachmentReport:
    attached: int
    placements: tuple[LinearPlacement, ...]

    def __post_init__(self) -> None:
        if self.attached <= 0:
            raise ValueError("LoRA attachment report must attach at least one layer")


def attach_lora_adapters(model: object, config: LoRAConfig) -> LoRAAttachmentReport:
    attached = 0
    wanted = set(config.placements)
    for layer in _iter_whale_linears(model):
        if layer.placement not in wanted:
            continue
        layer.enable_lora(rank=config.rank, alpha=config.alpha, scale=config.init_scale)
        attached += 1
    if attached == 0:
        raise ValueError(f"no WhaleLinear layers matched LoRA placements {sorted(wanted)}")
    return LoRAAttachmentReport(attached=attached, placements=config.placements)


def _iter_whale_linears(root: object) -> Iterator[WhaleLinear]:
    seen: set[int] = set()

    def walk(value: object) -> Iterator[WhaleLinear]:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        if isinstance(value, WhaleLinear):
            yield value
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from walk(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)
            return
        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                yield from walk(child)

    yield from walk(root)
