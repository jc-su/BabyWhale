from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from baby_whale_v4.rl.rewards import ToolRewardBreakdown, ToolUseTask, score_tool_response
from baby_whale_v4.tools.local import ToolRegistry


class TextPolicy(Protocol):
    def sample(self, prompt: str, *, group_size: int, max_new_tokens: int) -> list[str]: ...


@dataclass(frozen=True)
class ToolRolloutConfig:
    group_size: int = 4
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class RolloutRecord:
    prompt: str
    response: str
    reward: ToolRewardBreakdown

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("rollout prompt must be non-empty")
        if not isinstance(self.response, str):
            raise TypeError("rollout response must be a string")
        if not isinstance(self.reward, ToolRewardBreakdown):
            raise TypeError("rollout reward must be a ToolRewardBreakdown")


class ToolRolloutRunner:
    """Single-process rollout/reward runner for GRPO-style tool-use experiments.

    This mirrors the important separation used by larger RL stacks: policy
    rollout, environment/tool execution, reward scoring, and trainer update are
    separate surfaces. The update still belongs to the MLX GRPO trainer.
    """

    def __init__(self, *, policy: TextPolicy, registry: ToolRegistry, config: ToolRolloutConfig):
        self.policy = policy
        self.registry = registry
        self.config = config

    def collect(self, tasks: Sequence[ToolUseTask]) -> list[RolloutRecord]:
        records: list[RolloutRecord] = []
        for task in tasks:
            responses = self.policy.sample(
                task.prompt,
                group_size=self.config.group_size,
                max_new_tokens=self.config.max_new_tokens,
            )
            if len(responses) != self.config.group_size:
                raise ValueError("policy returned the wrong number of samples")
            for response in responses:
                reward = score_tool_response(response, task, self.registry)
                records.append(RolloutRecord(prompt=task.prompt, response=response, reward=reward))
        return records

    @staticmethod
    def mean_reward(records: Sequence[RolloutRecord]) -> float:
        if not records:
            raise ValueError("cannot average empty rollout records")
        return sum(record.reward.total for record in records) / len(records)
