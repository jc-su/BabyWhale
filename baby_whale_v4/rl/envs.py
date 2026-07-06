from dataclasses import dataclass

from baby_whale_v4.rl.rewards import ToolUseTask
from baby_whale_v4.tools.schema import ToolCall


@dataclass(frozen=True)
class ArithmeticTask:
    a: int
    b: int
    op: str

    def __post_init__(self) -> None:
        if self.op not in ("add", "subtract", "multiply"):
            raise ValueError(f"unsupported arithmetic op {self.op!r}")

    @property
    def answer(self) -> int:
        match self.op:
            case "add":
                return self.a + self.b
            case "subtract":
                return self.a - self.b
            case "multiply":
                return self.a * self.b
            case _:
                raise ValueError(f"unsupported arithmetic op {self.op!r}")

    def to_tool_task(self) -> ToolUseTask:
        tool_name = f"calculator.{self.op}"
        symbol = {"add": "+", "subtract": "-", "multiply": "*"}[self.op]
        return ToolUseTask(
            prompt=f"What is {self.a} {symbol} {self.b}? Use a tool call.",
            expected_call=ToolCall(tool_name, {"a": self.a, "b": self.b}),
            expected_answer=str(self.answer),
        )


def make_arithmetic_tool_tasks(limit: int = 12) -> list[ToolUseTask]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    tasks: list[ToolUseTask] = []
    ops = ("add", "subtract", "multiply")
    for idx in range(limit):
        op = ops[idx % len(ops)]
        a = 2 + idx
        b = 3 + (idx % 5)
        tasks.append(ArithmeticTask(a=a, b=b, op=op).to_tool_task())
    return tasks
