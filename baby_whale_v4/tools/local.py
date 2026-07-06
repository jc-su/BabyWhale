from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

from baby_whale_v4.tools.schema import JsonValue, ToolCall, ToolParameter, ToolResult, ToolSpec

type ToolFn = Callable[[Mapping[str, JsonValue]], JsonValue]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    fn: ToolFn

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ToolSpec):
            raise TypeError("registered tool spec must be a ToolSpec")
        if not callable(self.fn):
            raise TypeError("registered tool fn must be callable")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name!r}")
        self._tools[spec.name] = RegisteredTool(spec=spec, fn=fn)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}")
        return self._tools[name]

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.get(call.name)
            _validate_arguments(tool.spec, call.arguments)
            return ToolResult(name=call.name, ok=True, content=tool.fn(call.arguments))
        except Exception as exc:
            return ToolResult(name=call.name, ok=False, content=None, error=str(exc))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    _register_calculator(registry)
    _register_string(registry)
    _register_calendar(registry)
    return registry


def _validate_arguments(spec: ToolSpec, args: Mapping[str, JsonValue]) -> None:
    unknown = set(args) - spec.parameter_names
    if unknown:
        raise ValueError(f"unknown arguments for {spec.name}: {sorted(unknown)}")
    missing = spec.required_names - set(args)
    if missing:
        raise ValueError(f"missing required arguments for {spec.name}: {sorted(missing)}")
    by_name = {param.name: param for param in spec.parameters}
    for name, value in args.items():
        _validate_arg_type(spec.name, by_name[name], value)


def _validate_arg_type(tool_name: str, param: ToolParameter, value: JsonValue) -> None:
    match param.kind:
        case "string":
            ok = isinstance(value, str)
        case "integer":
            ok = type(value) is int
        case "number":
            ok = type(value) in (int, float)
        case "boolean":
            ok = type(value) is bool
        case "array":
            ok = isinstance(value, list)
        case "object":
            ok = isinstance(value, dict)
        case _:
            raise ValueError(f"unsupported tool parameter kind {param.kind!r}")
    if not ok:
        raise TypeError(f"{tool_name}.{param.name} must be {param.kind}")


def _number(args: Mapping[str, JsonValue], key: str) -> float:
    value = args[key]
    if type(value) is int:
        return float(value)
    if type(value) is float:
        return value
    raise TypeError(f"{key} must be a number")


def _str(args: Mapping[str, JsonValue], key: str) -> str:
    value = args[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _binary_number_spec(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=(
            ToolParameter("a", "number", description="left operand"),
            ToolParameter("b", "number", description="right operand"),
        ),
    )


def _register_calculator(registry: ToolRegistry) -> None:
    registry.register(_binary_number_spec("calculator.add", "Add two numbers"), _calc_add)
    registry.register(_binary_number_spec("calculator.subtract", "Subtract two numbers"), _calc_sub)
    registry.register(_binary_number_spec("calculator.multiply", "Multiply two numbers"), _calc_mul)
    registry.register(_binary_number_spec("calculator.divide", "Divide two numbers"), _calc_div)


def _calc_add(args: Mapping[str, JsonValue]) -> JsonValue:
    return _number(args, "a") + _number(args, "b")


def _calc_sub(args: Mapping[str, JsonValue]) -> JsonValue:
    return _number(args, "a") - _number(args, "b")


def _calc_mul(args: Mapping[str, JsonValue]) -> JsonValue:
    return _number(args, "a") * _number(args, "b")


def _calc_div(args: Mapping[str, JsonValue]) -> JsonValue:
    b = _number(args, "b")
    if b == 0:
        raise ValueError("division by zero")
    return _number(args, "a") / b


def _register_string(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="string.count",
            description="Count non-overlapping occurrences of a pattern in text",
            parameters=(
                ToolParameter("text", "string"),
                ToolParameter("pattern", "string"),
            ),
        ),
        lambda args: _str(args, "text").count(_str(args, "pattern")),
    )
    registry.register(
        ToolSpec(
            name="string.replace",
            description="Replace all occurrences of a substring",
            parameters=(
                ToolParameter("text", "string"),
                ToolParameter("old", "string"),
                ToolParameter("new", "string"),
            ),
        ),
        lambda args: _str(args, "text").replace(_str(args, "old"), _str(args, "new")),
    )


def _register_calendar(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="calendar.day_of_week",
            description="Return the weekday name for an ISO date",
            parameters=(ToolParameter("date", "string", description="YYYY-MM-DD date"),),
        ),
        _day_of_week,
    )


def _day_of_week(args: Mapping[str, JsonValue]) -> JsonValue:
    return date.fromisoformat(_str(args, "date")).strftime("%A")
