import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsIndex, TypeIs, cast

import mlx.core as mx

from baby_whale_v4.data.dataset import TensorPair
from baby_whale_v4.data.tokenizer import Tokenizer
from baby_whale_v4.typing import ChatRole, ThinkMode, ensure_in

_CHAT_ROLES: tuple[ChatRole, ...] = ("system", "user", "assistant", "tool")

EOT_TAG = "<|eot|>"


@dataclass(frozen=True)
class Message:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        ensure_in("message.role", self.role, _CHAT_ROLES)
        if not isinstance(self.content, str):
            raise TypeError("message.content must be a string")


def role_tag(role: ChatRole, think: ThinkMode = "none") -> str:
    if role == "assistant" and think == "thinking":
        return "<|assistant_thinking|>"
    return f"<|{role}|>"


def render_chat_prompt(messages: Sequence[Message] | Sequence[Mapping[str, str]]) -> str:
    """Render a chat history as the raw prompt string for generation.

    Emits ``<|role|>content<|eot|>`` per turn and appends ``<|assistant|>`` as
    the open-ended completion target. Accepts either ``Message`` instances or
    plain ``{"role","content"}`` mappings so HTTP request bodies and Python
    callers share the same template.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, Message):
            role, content = msg.role, msg.content
        else:
            role = str(msg["role"])
            content = str(msg["content"])
        parts.append(f"<|{role}|>{content}{EOT_TAG}")
    parts.append("<|assistant|>")
    return "".join(parts)


def format_chat(
    messages: Sequence[Message],
    tokenizer: Tokenizer,
    *,
    train_on_role_tag: bool = False,
    think: ThinkMode = "none",
) -> tuple[list[int], list[int]]:
    """Render a chat into byte token IDs and an assistant-only mask.

    Returned mask is 1 for tokens that should contribute to SFT loss, 0 otherwise.
    """
    ids: list[int] = []
    mask: list[int] = []
    for msg in messages:
        tag = role_tag(msg.role, think=think if msg.role == "assistant" else "none")
        tag_ids = tokenizer.encode(tag)
        body_ids = tokenizer.encode(msg.content + EOT_TAG)
        ids.extend(tag_ids)
        ids.extend(body_ids)
        if msg.role == "assistant":
            mask.extend([1 if train_on_role_tag else 0] * len(tag_ids))
            mask.extend([1] * len(body_ids))
        else:
            mask.extend([0] * (len(tag_ids) + len(body_ids)))
    return ids, mask


@dataclass
class ChatExample:
    messages: list[Message]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("chat example must contain at least one message")


@dataclass
class SFTDataset:
    examples: list[ChatExample]
    tokenizer: Tokenizer
    block_size: int

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("SFTDataset examples must be non-empty")
        if self.block_size <= 1:
            raise ValueError("block_size must be > 1")
        xs: list[mx.array] = []
        ys: list[mx.array] = []
        for ex in self.examples:
            ids, mask = format_chat(ex.messages, self.tokenizer)
            ids = ids[: self.block_size + 1]
            mask = mask[: self.block_size + 1]
            if len(ids) < self.block_size + 1:
                pad_n = self.block_size + 1 - len(ids)
                ids = ids + [self.tokenizer.pad_id] * pad_n
                mask = mask + [0] * pad_n
            x = mx.array(ids[:-1], dtype=mx.int32)
            y = mx.array(ids[1:], dtype=mx.int32)
            mask_y = mx.array(mask[1:], dtype=mx.int32)
            y = mx.where(mask_y == 0, mx.full(y.shape, -1, dtype=mx.int32), y)
            xs.append(x)
            ys.append(y)
        self._x = mx.stack(xs, axis=0)
        self._y = mx.stack(ys, axis=0)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: SupportsIndex) -> TensorPair:
        index = operator.index(index)
        return self._x[index], self._y[index]


def chat_examples_from_jsonl(path: Path | str) -> list[ChatExample]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    examples: list[ChatExample] = []
    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{line_no}: invalid JSONL row: {exc}") from exc
            examples.append(_chat_example_from_record(raw, source=f"{src}:{line_no}"))
    if not examples:
        raise ValueError(f"{src} produced zero chat examples")
    return examples


def sft_dataset_from_jsonl(
    path: Path | str, tokenizer: Tokenizer, *, block_size: int
) -> SFTDataset:
    return SFTDataset(
        examples=chat_examples_from_jsonl(path),
        tokenizer=tokenizer,
        block_size=block_size,
    )


def _chat_example_from_record(raw: object, *, source: str) -> ChatExample:
    if not _is_str_mapping(raw):
        raise TypeError(f"{source} must be an object with string keys")
    kind = raw.get("kind")
    if kind not in ("chat", "tool_trace"):
        raise ValueError(f"{source}.kind must be 'chat' or 'tool_trace'")
    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list):
        raise TypeError(f"{source}.messages must be a list")
    messages: list[Message] = []
    for idx, item in enumerate(messages_raw):
        if not _is_str_mapping(item):
            raise TypeError(f"{source}.messages[{idx}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in _CHAT_ROLES:
            raise ValueError(f"{source}.messages[{idx}].role is unsupported")
        if not isinstance(content, str):
            raise TypeError(f"{source}.messages[{idx}].content must be a string")
        messages.append(Message(cast(ChatRole, role), content))
    return ChatExample(messages)


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)
