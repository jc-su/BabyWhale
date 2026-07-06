"""Context-length curriculum for V4-style native long-context pretrain.

DeepSeek-V4 grows the training sequence length 4K → 16K → 64K → 1M *inside*
pretraining rather than rescaling RoPE in a post-hoc stage. This module
provides the typed schedule and a thin re-entrant pretrain loop that
implements that idea.

The types are deliberately small and value-typed so the schedule is easy to
construct, validate, and pass around:

- ``CurriculumPhase``  : one (context_length, n_tokens) pair, frozen.
- ``ContextCurriculum``: ordered tuple of phases, monotone non-decreasing in
  ``context_length``. Constructable from a CLI string via :meth:`parse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

_HUMAN_SUFFIXES: dict[str, int] = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
}


def _parse_count(literal: str) -> int:
    """Accept ``"384"``, ``"80M"``, or ``"1.5B"``. Fail-fast on anything else."""
    text = literal.strip()
    if not text:
        raise ValueError("empty count literal")
    suffix = text[-1]
    if suffix in _HUMAN_SUFFIXES:
        mantissa = float(text[:-1])
        if mantissa <= 0:
            raise ValueError(f"count must be positive: {literal!r}")
        return int(mantissa * _HUMAN_SUFFIXES[suffix])
    value = int(text)
    if value <= 0:
        raise ValueError(f"count must be positive: {literal!r}")
    return value


@dataclass(frozen=True)
class CurriculumPhase:
    """One phase of a context-length curriculum.

    ``n_tokens`` is the *target token count* to train through at this phase's
    ``context_length`` before advancing. The loop counts non-pad target
    tokens (same notion of "token" as the metrics ``tokens`` field).
    """

    context_length: int
    n_tokens: int

    def __post_init__(self) -> None:
        if self.context_length <= 1:
            raise ValueError("CurriculumPhase.context_length must be > 1")
        if self.n_tokens <= 0:
            raise ValueError("CurriculumPhase.n_tokens must be positive")


@dataclass(frozen=True)
class ContextCurriculum:
    """Ordered sequence of curriculum phases.

    Phases must be non-decreasing in ``context_length`` so the model never
    sees a shorter window after a longer one — matches V4's 4K→16K→64K→1M
    ramp.
    """

    phases: tuple[CurriculumPhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("ContextCurriculum must contain at least one phase")
        for prev, cur in zip(self.phases[:-1], self.phases[1:], strict=True):
            if cur.context_length < prev.context_length:
                raise ValueError(
                    "phases must be non-decreasing in context_length; "
                    f"got {prev.context_length} -> {cur.context_length}"
                )

    @classmethod
    def parse(cls, spec: str) -> Self:
        """Parse a CLI string of the form ``"len:tokens,len:tokens,..."``.

        Example::

            ContextCurriculum.parse("384:80M,768:80M,1536:80M")
        """
        if not spec or not spec.strip():
            raise ValueError("ContextCurriculum.parse: empty schedule")
        phases: list[CurriculumPhase] = []
        for raw in spec.split(","):
            entry = raw.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(f"ContextCurriculum.parse: phase missing ':' separator: {raw!r}")
            len_str, tok_str = entry.split(":", 1)
            phases.append(
                CurriculumPhase(
                    context_length=_parse_count(len_str),
                    n_tokens=_parse_count(tok_str),
                )
            )
        return cls(phases=tuple(phases))

    @property
    def max_context_length(self) -> int:
        return max(p.context_length for p in self.phases)

    @property
    def total_tokens(self) -> int:
        return sum(p.n_tokens for p in self.phases)
