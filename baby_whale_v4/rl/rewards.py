from dataclasses import dataclass

from baby_whale_v4.tools.local import ToolRegistry
from baby_whale_v4.tools.schema import ToolCall, parse_tool_call_text, text_after_tool_call


@dataclass(frozen=True)
class ToolUseTask:
    prompt: str
    expected_call: ToolCall
    expected_answer: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("tool-use task prompt must be non-empty")


@dataclass(frozen=True)
class ToolRewardBreakdown:
    total: float
    valid_json: bool
    tool_exists: bool
    args_match: bool
    tool_exec_ok: bool
    answer_match: bool
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("valid_json", "tool_exists", "args_match", "tool_exec_ok", "answer_match"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.error is not None and not self.error:
            raise ValueError("reward error must be non-empty when present")

    def to_metrics(self) -> dict[str, float]:
        return {
            "reward": self.total,
            "valid_json": float(self.valid_json),
            "tool_exists": float(self.tool_exists),
            "args_match": float(self.args_match),
            "tool_exec_ok": float(self.tool_exec_ok),
            "answer_match": float(self.answer_match),
        }


def score_tool_response(
    response: str,
    task: ToolUseTask,
    registry: ToolRegistry,
) -> ToolRewardBreakdown:
    total = 0.0
    try:
        call = parse_tool_call_text(response)
    except ValueError as exc:
        return ToolRewardBreakdown(
            total=-1.0,
            valid_json=False,
            tool_exists=False,
            args_match=False,
            tool_exec_ok=False,
            answer_match=False,
            error=str(exc),
        )
    total += 1.0

    tool_exists = call.name in registry.names
    if tool_exists:
        total += 1.0

    args_match = call == task.expected_call
    if args_match:
        total += 1.0

    result = registry.execute(call)
    tool_exec_ok = result.ok
    if tool_exec_ok:
        total += 1.0

    answer_match = False
    if task.expected_answer is None:
        answer_match = True
    else:
        final_text = text_after_tool_call(response)
        answer_match = task.expected_answer.strip() in final_text
    if answer_match:
        total += 1.0

    return ToolRewardBreakdown(
        total=total,
        valid_json=True,
        tool_exists=tool_exists,
        args_match=args_match,
        tool_exec_ok=tool_exec_ok,
        answer_match=answer_match,
        error=None if result.ok else result.error,
    )
