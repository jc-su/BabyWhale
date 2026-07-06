import json
import re
from dataclasses import dataclass
from typing import Literal, TypeIs, cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[object] | dict[str, object]
type ToolParamType = Literal["string", "integer", "number", "boolean", "array", "object"]

_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


def is_json_value(value: object) -> TypeIs[JsonValue]:
    if value is None or isinstance(value, str):
        return True
    if type(value) in (int, float, bool):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def ensure_tool_name(name: str) -> None:
    if not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid tool name {name!r}")


@dataclass(frozen=True)
class ToolParameter:
    name: str
    kind: ToolParamType
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool parameter name must be non-empty")
        if self.kind not in ("string", "integer", "number", "boolean", "array", "object"):
            raise ValueError(f"unsupported tool parameter kind {self.kind!r}")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...]

    def __post_init__(self) -> None:
        ensure_tool_name(self.name)
        seen: set[str] = set()
        for param in self.parameters:
            if param.name in seen:
                raise ValueError(f"duplicate parameter {param.name!r} for tool {self.name}")
            seen.add(param.name)

    @property
    def required_names(self) -> set[str]:
        return {param.name for param in self.parameters if param.required}

    @property
    def parameter_names(self) -> set[str]:
        return {param.name for param in self.parameters}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        ensure_tool_name(self.name)
        if not all(isinstance(key, str) for key in self.arguments):
            raise TypeError("tool call argument keys must be strings")
        for key, value in self.arguments.items():
            if not is_json_value(value):
                raise TypeError(f"tool call argument {key!r} is not JSON-serializable")

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: JsonValue
    error: str | None = None

    def __post_init__(self) -> None:
        ensure_tool_name(self.name)
        if not is_json_value(self.content):
            raise TypeError("tool result content must be JSON-serializable")
        if self.ok and self.error is not None:
            raise ValueError("successful tool result cannot include error")
        if not self.ok and not self.error:
            raise ValueError("failed tool result must include error")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"ok": self.ok, "content": self.content}
        if self.error is not None:
            payload["error"] = self.error
        return payload


_RESULT_OPEN_TAG = "<tool_result>"
_RESULT_CLOSE_TAG = "</tool_result>"


def render_tool_call(call: ToolCall) -> str:
    body = json.dumps(call.to_payload(), sort_keys=True, separators=(",", ":"))
    return f"{_OPEN_TAG}{body}{_CLOSE_TAG}"


def render_tool_result(result: ToolResult) -> str:
    body = json.dumps(result.to_payload(), sort_keys=True, separators=(",", ":"))
    return f"{_RESULT_OPEN_TAG}{body}{_RESULT_CLOSE_TAG}"


TOOL_CALL_CLOSE_TAG: str = _CLOSE_TAG


def parse_tool_call_text(text: str) -> ToolCall:
    if text.count(_OPEN_TAG) != 1 or text.count(_CLOSE_TAG) != 1:
        raise ValueError("assistant text must contain exactly one tool_call block")
    start = text.index(_OPEN_TAG) + len(_OPEN_TAG)
    end = text.index(_CLOSE_TAG)
    if end <= start:
        raise ValueError("tool_call block is empty")
    raw = text[start:end].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool_call JSON is invalid: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool_call payload must be a JSON object")
    unknown = set(payload) - {"name", "arguments"}
    if unknown:
        raise ValueError(f"unknown tool_call keys: {sorted(unknown)}")
    name = payload.get("name")
    args = payload.get("arguments")
    if not isinstance(name, str):
        raise ValueError("tool_call.name must be a string")
    if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
        raise ValueError("tool_call.arguments must be an object with string keys")
    arguments = cast(dict[str, object], args)
    if not all(is_json_value(value) for value in arguments.values()):
        raise ValueError("tool_call.arguments must be JSON values")
    return ToolCall(name=name, arguments=cast(dict[str, JsonValue], dict(arguments)))


def text_after_tool_call(text: str) -> str:
    if _CLOSE_TAG not in text:
        return ""
    return text.split(_CLOSE_TAG, 1)[1].strip()
