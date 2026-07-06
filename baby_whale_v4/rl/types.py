"""Typed records that flow between rollout, reward, buffer, and trainer."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.tools.schema import ToolCall


@dataclass(frozen=True)
class ToolSpec:
    """Lightweight handle the rollout engine can pass to a tool runner.

    The tool registry (``tools/local.py``) holds the actual executors; this
    record names which tools are visible for a given rollout.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must be non-empty")


@dataclass(frozen=True)
class RolloutRequest:
    prompt_ids: tuple[int, ...]
    options: GenerationOptions
    tools: tuple[ToolSpec, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("RolloutRequest.prompt_ids must be non-empty")
        if not isinstance(self.options, GenerationOptions):
            raise TypeError("RolloutRequest.options must be a GenerationOptions")


@dataclass(frozen=True)
class RolloutSample:
    """A single completion produced by a rollout engine.

    ``log_probs`` is the rollout-time log π_old(token | prefix) for each token
    in ``response_ids``, captured at sample time. Trainers should treat it as
    the old-policy reference and never re-derive it from a post-update model.
    """

    request: RolloutRequest
    response_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    finished: bool
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.log_probs):
            raise ValueError(
                f"response_ids ({len(self.response_ids)}) and log_probs "
                f"({len(self.log_probs)}) must have matching lengths"
            )
        if not isinstance(self.request, RolloutRequest):
            raise TypeError("RolloutSample.request must be a RolloutRequest")
        for value in self.log_probs:
            if not isinstance(value, float):
                raise TypeError("log_probs must be Python floats")


@dataclass(frozen=True)
class ScoredSample:
    """A rollout sample with its scalar reward attached."""

    sample: RolloutSample
    reward: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample, RolloutSample):
            raise TypeError("ScoredSample.sample must be a RolloutSample")
        if not isinstance(self.reward, float):
            raise TypeError("ScoredSample.reward must be a Python float")
