"""GRPO with verifiable code-execution rewards (MBPP / HumanEval-style).

Wraps the generic :func:`grpo` loop with the wiring a coding-agent run needs:

* Each problem's prompt is encoded with the user's tokenizer.
* Each ``RolloutRequest`` carries ``metadata={"problem_id": <id>}`` so the
  reward host can look up the right tests.
* A :class:`CodeRewardHost` is built automatically from the problem list and
  the tokenizer's ``decode``.

This is the canonical entry point for "train a coding agent on verifiable
unit tests". It is intentionally a thin wrapper — the gradient step itself is
just :func:`grpo_step`, the algorithm is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from baby_whale_v4.data.tokenizer import Tokenizer
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.rl.code_reward import CodeRewardConfig, CodeRewardHost
from baby_whale_v4.rl.code_tasks import CodeProblem, problem_index
from baby_whale_v4.rl.rollout import RolloutEngine
from baby_whale_v4.training.grpo import GRPOConfig, grpo


def code_grpo(
    *,
    model: BabyWhaleV4Model,
    problems: Sequence[CodeProblem],
    tokenizer: Tokenizer,
    grpo_config: GRPOConfig,
    out_dir: Path | str,
    rollout_engine: RolloutEngine | None = None,
    reward_config: CodeRewardConfig | None = None,
) -> BabyWhaleV4Model:
    """Run GRPO on a list of `CodeProblem`s with code-execution as the reward.

    Each problem's ``prompt`` is tokenized once; the rollout engine produces
    ``group_size`` samples per problem; the :class:`CodeRewardHost` runs the
    samples in a subprocess sandbox against ``problem.tests`` and returns
    pass/fail (or partial credit) as the scalar reward.
    """

    if not problems:
        raise ValueError("code_grpo requires at least one CodeProblem")
    idx = problem_index(problems)

    prompt_arrays: list[mx.array] = []
    metadata: list[dict[str, str]] = []
    for problem in problems:
        prompt_ids = tokenizer.encode(problem.prompt)
        if not prompt_ids:
            raise ValueError(f"problem {problem.problem_id!r} encoded to an empty prompt")
        prompt_arrays.append(mx.array(prompt_ids, dtype=mx.int32))
        metadata.append({"problem_id": problem.problem_id})

    host = CodeRewardHost(
        problems=idx,
        decode=lambda response_ids: tokenizer.decode(list(response_ids)),
        config=reward_config or CodeRewardConfig(),
    )

    return grpo(
        model=model,
        prompts=prompt_arrays,
        grpo_config=grpo_config,
        out_dir=out_dir,
        rollout_engine=rollout_engine,
        reward_host=host,
        request_metadata=metadata,
    )
