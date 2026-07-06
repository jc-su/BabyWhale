"""Reward host that turns generated code into a verifiable scalar.

Given a :class:`RolloutSample`, the host:

1. Looks up the problem by ``sample.request.metadata["problem_id"]``.
2. Decodes ``response_ids`` to text via the user-supplied tokenizer.
3. Extracts a fenced Python code block, or uses the raw text if no fence is present.
4. Runs the code + the problem's tests in the subprocess sandbox.
5. Returns ``1.0`` if all tests pass, ``0.0`` otherwise. With
   ``partial_credit=True`` returns ``n_passed / n_tests``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from baby_whale_v4.rl.code_exec import execute_python_with_tests
from baby_whale_v4.rl.code_tasks import CodeProblem
from baby_whale_v4.rl.types import RolloutSample
from baby_whale_v4.typing import ProblemId

_CODE_FENCE_PY = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Pull a Python code block out of an assistant response.

    If a fenced Python block is present, return its contents. If a plain fence
    is present, return that. Otherwise return the raw text trimmed.
    """

    match = _CODE_FENCE_PY.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass(frozen=True)
class CodeRewardConfig:
    timeout_sec: float = 5.0
    partial_credit: bool = False
    fail_reward: float = 0.0
    pass_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self.pass_reward <= self.fail_reward:
            raise ValueError("pass_reward must be greater than fail_reward")


class CodeRewardHost:
    """Reward host backed by a subprocess Python sandbox.

    ``problems`` is a dict keyed by ``problem_id`` (use
    :func:`baby_whale_v4.rl.code_tasks.problem_index` to build it). ``decode``
    is a callable that converts ``response_ids`` to text — typically a thin
    wrapper around your tokenizer's ``decode`` method.
    """

    def __init__(
        self,
        *,
        problems: Mapping[ProblemId, CodeProblem],
        decode: Callable[[list[int]], str],
        config: CodeRewardConfig | None = None,
    ) -> None:
        if not callable(decode):
            raise TypeError("decode must be a callable")
        if not problems:
            raise ValueError("problems must be non-empty")
        self.problems: dict[ProblemId, CodeProblem] = dict(problems)
        self.decode = decode
        self.config = config or CodeRewardConfig()

    def score(self, sample: RolloutSample) -> float:
        if not isinstance(sample, RolloutSample):
            raise TypeError("CodeRewardHost.score expects a RolloutSample")
        raw_id = sample.request.metadata.get("problem_id")
        if not raw_id:
            return float(self.config.fail_reward)
        problem_id = ProblemId(raw_id)
        if problem_id not in self.problems:
            return float(self.config.fail_reward)
        problem = self.problems[problem_id]
        text = self.decode(list(sample.response_ids))
        code = extract_python_code(text)
        if not code:
            return float(self.config.fail_reward)
        try:
            result = execute_python_with_tests(
                solution_code=code,
                tests=list(problem.tests),
                timeout_sec=self.config.timeout_sec,
            )
        except ValueError, TypeError:
            return float(self.config.fail_reward)
        if self.config.partial_credit:
            fraction = result.n_passed / max(1, result.n_tests)
            span = self.config.pass_reward - self.config.fail_reward
            return float(self.config.fail_reward + fraction * span)
        return float(self.config.pass_reward if result.passed else self.config.fail_reward)

    def score_batch(self, samples: Sequence[RolloutSample]) -> list[float]:
        return [self.score(sample) for sample in samples]
