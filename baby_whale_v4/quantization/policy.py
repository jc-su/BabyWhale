from dataclasses import dataclass
from typing import get_args

from baby_whale_v4.typing import (
    LinearPlacement,
    QuantMode,
    ResolvedQuantMode,
    assert_never,
    ensure_in,
)

_QUANT_MODES: tuple[QuantMode, ...] = get_args(QuantMode)


@dataclass(frozen=True)
class QuantizedLinearPolicy:
    """Placement-aware quantization policy for DeepSeek-style precision layout."""

    requested: QuantMode

    def __post_init__(self) -> None:
        ensure_in("quant policy mode", self.requested, _QUANT_MODES)

    def for_placement(self, placement: LinearPlacement) -> ResolvedQuantMode:
        match self.requested:
            case "none":
                return "none"
            case "int8-weight" | "int4-weight":
                return self.requested
            case "fp4-expert":
                match placement:
                    case "moe_expert":
                        return "fp4-native"
                    case "general" | "attention" | "router" | "lm_head" | "mtp":
                        return "none"
                    case _:
                        assert_never(placement)
            case "fp4-native":
                return self.requested
            case _:
                assert_never(self.requested)


def quant_mode_for_placement(mode: QuantMode, placement: LinearPlacement) -> ResolvedQuantMode:
    return QuantizedLinearPolicy(mode).for_placement(placement)
