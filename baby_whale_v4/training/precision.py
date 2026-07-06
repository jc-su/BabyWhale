from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.training.fp4_native import require_native_fp4_training


def ensure_training_precision_supported(config: BabyWhaleV4Config) -> None:
    """Reject modes that are inference-only in the current MLX educational stack."""

    if config.quant_mode == "fp4-expert":
        raise RuntimeError("fp4-expert is inference/export only; train in bf16 and export experts")
    if config.quant_mode == "fp4-native":
        require_native_fp4_training()
