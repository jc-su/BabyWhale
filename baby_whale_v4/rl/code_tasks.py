"""MBPP / HumanEval / generic-code-problem loaders.

Each problem ends up as a :class:`CodeProblem` with a prompt for the model,
one or more test snippets, and (optionally) a canonical reference solution.
The loaders accept a ``limit`` so a Mac run can stay within minutes.

For training, attach a :class:`CodeProblem` to a :class:`RolloutRequest` via
``metadata={"problem_id": problem.problem_id}``; the matching reward host in
``code_reward.py`` looks up the tests by id at score time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

from baby_whale_v4.typing import ProblemId


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, dict) and all(isinstance(k, str) for k in value)


def _is_str_list(value: object) -> TypeIs[list[str]]:
    return isinstance(value, list) and all(isinstance(t, str) for t in value)


@dataclass(frozen=True)
class CodeProblem:
    """A code-generation task with attached unit tests."""

    problem_id: ProblemId
    prompt: str
    tests: tuple[str, ...]
    canonical_solution: str | None = None
    entry_point: str | None = None

    def __post_init__(self) -> None:
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if not self.tests:
            raise ValueError("tests must be non-empty")
        for test in self.tests:
            if not isinstance(test, str):
                raise TypeError("each test must be a string")
        if self.canonical_solution is not None and not isinstance(self.canonical_solution, str):
            raise TypeError("canonical_solution must be a string when present")
        if self.entry_point is not None and not isinstance(self.entry_point, str):
            raise TypeError("entry_point must be a string when present")


def to_jsonl_record(problem: CodeProblem) -> dict[str, object]:
    return {
        "problem_id": problem.problem_id,
        "prompt": problem.prompt,
        "tests": list(problem.tests),
        "canonical_solution": problem.canonical_solution,
        "entry_point": problem.entry_point,
    }


def from_jsonl_record(data: object) -> CodeProblem:
    if not _is_str_mapping(data):
        raise TypeError("CodeProblem JSONL record must be an object with string keys")
    tests = data.get("tests")
    if not _is_str_list(tests):
        raise TypeError("CodeProblem.tests must be a list of strings")
    canonical_raw = data.get("canonical_solution")
    entry_point_raw = data.get("entry_point")
    return CodeProblem(
        problem_id=ProblemId(str(data["problem_id"])),
        prompt=str(data["prompt"]),
        tests=tuple(tests),
        canonical_solution=None if canonical_raw is None else str(canonical_raw),
        entry_point=None if entry_point_raw is None else str(entry_point_raw),
    )


def save_problems_to_jsonl(problems: Iterable[CodeProblem], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for problem in problems:
            f.write(json.dumps(to_jsonl_record(problem), sort_keys=True) + "\n")
    return out


def load_problems_from_jsonl(path: Path | str) -> list[CodeProblem]:
    out: list[CodeProblem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(from_jsonl_record(json.loads(line)))
    return out


def problem_index(problems: Sequence[CodeProblem]) -> dict[ProblemId, CodeProblem]:
    """Index a list of problems by ``problem_id``. Rejects duplicates."""

    out: dict[ProblemId, CodeProblem] = {}
    for problem in problems:
        if problem.problem_id in out:
            raise ValueError(f"duplicate problem_id: {problem.problem_id}")
        out[problem.problem_id] = problem
    return out


def load_mbpp_from_hf(*, split: str = "test", limit: int | None = None) -> list[CodeProblem]:
    """Load MBPP problems via the ``datasets`` library.

    MBPP fields used:
      * ``task_id`` — int
      * ``text`` — natural language prompt ("Write a function...")
      * ``test_list`` — list[str] of assertions
      * ``code`` — canonical solution (used as ``canonical_solution``)

    The model prompt is the natural-language text plus an opening Python code
    fence, so the model is expected to emit a code block that we can parse.
    """

    from datasets import load_dataset

    ds = load_dataset("mbpp", split=split)
    problems: list[CodeProblem] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        problems.append(
            CodeProblem(
                problem_id=ProblemId(f"mbpp/{row['task_id']}"),
                prompt=f"{row['text']}\n\n```python\n",
                tests=tuple(row["test_list"]),
                canonical_solution=row.get("code"),
                entry_point=None,
            )
        )
    return problems


def load_humaneval_from_hf(*, limit: int | None = None) -> list[CodeProblem]:
    """Load OpenAI HumanEval problems via the ``datasets`` library.

    HumanEval's test field defines a ``check(candidate)`` function and we add
    a final ``check(<entry_point>)`` call so the runner triggers it.
    """

    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    problems: list[CodeProblem] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        test_block = row["test"] + f"\ncheck({row['entry_point']})\n"
        problems.append(
            CodeProblem(
                problem_id=ProblemId(row["task_id"]),
                prompt=row["prompt"],
                tests=(test_block,),
                canonical_solution=row.get("canonical_solution"),
                entry_point=row["entry_point"],
            )
        )
    return problems
