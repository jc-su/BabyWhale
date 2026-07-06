from collections.abc import Iterator
from typing import TYPE_CHECKING

from baby_whale_v4.quantization.policy import quant_mode_for_placement
from baby_whale_v4.typing import QuantMode, assert_never

if TYPE_CHECKING:
    from baby_whale_v4.layers import WhaleLinear


def apply_weight_quantization(model: object, mode: QuantMode) -> int:
    """Apply MLX weight quantization to every Baby Whale linear layer.

    Every active mode runs through real MLX kernels: ``int8-weight`` and
    ``int4-weight`` use ``mx.quantized_matmul(mode="affine")``, ``fp4-native``
    uses ``mode="mxfp4"``/``"nvfp4"``, ``fp4-expert`` is the DeepSeek-style
    placement policy that flips only MoE expert linears to native FP4. The
    layer packs lazily on first forward, so this function only sets the mode
    and clears any stale packed cache.
    """

    if mode == "none":
        return 0
    n = 0
    for module in _iter_whale_linears(model):
        target_mode = quant_mode_for_placement(mode, module.placement)
        match target_mode:
            case "none":
                module.clear_quant_cache()
                module.quant_mode = "none"
            case "int8-weight" | "int4-weight" | "fp4-native":
                module.clear_quant_cache()
                module.quant_mode = target_mode
                n += 1
            case _:
                assert_never(target_mode)
    return n


def apply_fp4_expert_export(model: object) -> int:
    """Apply DeepSeek-style FP4 only to MoE expert linears."""

    return apply_weight_quantization(model, "fp4-expert")


def _iter_whale_linears(root: object) -> Iterator[WhaleLinear]:
    # Local import: weights.py and layers.py have a runtime cycle (layers.py
    # needs the policy resolver, this module needs WhaleLinear). The cycle is
    # benign because we only need the class at iteration time, not import time.
    from baby_whale_v4.layers import WhaleLinear

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
