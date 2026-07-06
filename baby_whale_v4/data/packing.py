import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeIs, cast

import numpy as np

from baby_whale_v4.data.dataset import PackedDataset, PackedTokenDataset
from baby_whale_v4.data.tokenizer import Tokenizer

type NormalizedKind = Literal["pretrain", "chat", "preference", "tool_trace"]


@dataclass(frozen=True)
class DatasetMixtureSource:
    path: Path
    repeat: int = 1
    limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.repeat <= 0:
            raise ValueError("mixture source repeat must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("mixture source limit must be positive when set")


@dataclass(frozen=True)
class PackedTokenFile:
    path: Path
    manifest_path: Path
    n_tokens: int
    n_blocks: int
    block_size: int
    tokenizer_hash: str

    def __post_init__(self) -> None:
        if self.n_tokens <= 0:
            raise ValueError("packed token file must contain tokens")
        if self.n_blocks <= 0:
            raise ValueError("packed token file must contain blocks")
        if self.block_size <= 1:
            raise ValueError("block_size must be > 1")
        if not self.tokenizer_hash:
            raise ValueError("tokenizer_hash must be non-empty")


def read_normalized_texts(path: Path | str, *, limit: int | None = None) -> list[str]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when set")
    texts: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(texts) >= limit:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{line_no}: invalid JSONL row: {exc}") from exc
            texts.append(normalized_record_to_text(raw, source=f"{src}:{line_no}"))
    if not texts:
        raise ValueError(f"{src} produced zero normalized texts")
    return texts


def normalized_record_to_text(raw: object, *, source: str = "record") -> str:
    if not _is_str_mapping(raw):
        raise TypeError(f"{source} must be an object with string keys")
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise TypeError(f"{source}.kind must be a string")
    match cast(NormalizedKind, kind):
        case "pretrain":
            return _read_nonempty_str(raw, "text", source)
        case "chat":
            return _messages_to_text(_read_messages(raw, "messages", source))
        case "preference":
            prompt = _read_nonempty_str(raw, "prompt", source)
            chosen = _read_nonempty_str(raw, "chosen", source)
            rejected = _read_nonempty_str(raw, "rejected", source)
            return f"<|user|>{prompt}<|chosen|>{chosen}<|rejected|>{rejected}"
        case "tool_trace":
            messages = _messages_to_text(_read_messages(raw, "messages", source))
            tools = raw.get("tools", [])
            if not isinstance(tools, list):
                raise TypeError(f"{source}.tools must be a list")
            return f"{messages}<|tools|>{json.dumps(tools, sort_keys=True, ensure_ascii=False)}"
        case _:
            raise ValueError(f"unsupported normalized record kind {kind!r}")


def pack_normalized_jsonl(
    path: Path | str,
    *,
    tokenizer: Tokenizer,
    block_size: int,
    limit: int | None = None,
) -> PackedDataset:
    texts = read_normalized_texts(path, limit=limit)
    docs = [tokenizer.encode(text) for text in texts]
    return PackedDataset(
        documents=docs,
        block_size=block_size,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
    )


def pack_mixture_jsonl(
    sources: Sequence[DatasetMixtureSource],
    *,
    tokenizer: Tokenizer,
    block_size: int,
) -> PackedDataset:
    if not sources:
        raise ValueError("dataset mixture sources must be non-empty")
    docs: list[list[int]] = []
    for source in sources:
        texts = read_normalized_texts(source.path, limit=source.limit)
        encoded = [tokenizer.encode(text) for text in texts]
        for _ in range(source.repeat):
            docs.extend(encoded)
    return PackedDataset(
        documents=docs,
        block_size=block_size,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
    )


def save_packed_token_file(
    dataset: PackedDataset | PackedTokenDataset,
    path: Path | str,
    *,
    tokenizer_hash: str,
    sources: Sequence[Path | str],
) -> PackedTokenFile:
    if not tokenizer_hash:
        raise ValueError("tokenizer_hash must be non-empty")
    out = Path(path)
    # np.savez_compressed appends ".npz" when the name lacks it; mirror that
    # here so _sha256_file, the manifest, and the returned path all reference
    # the file numpy actually writes. Without this, `--out foo` writes foo.npz
    # but the sha256 read targets the nonexistent `foo` and crashes.
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.array(dataset.packed_tokens.tolist(), dtype=np.int32)
    np.savez_compressed(
        out, tokens=tokens, block_size=np.array([dataset.block_size], dtype=np.int32)
    )
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest = {
        "format": "baby_whale_v4_packed_tokens_v1",
        "path": str(out),
        "n_tokens": int(tokens.shape[0]),
        "n_blocks": len(dataset),
        "block_size": dataset.block_size,
        "tokenizer_hash": tokenizer_hash,
        "sources": [str(Path(source)) for source in sources],
        "tokens_sha256": _sha256_file(out),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return PackedTokenFile(
        path=out,
        manifest_path=manifest_path,
        n_tokens=int(tokens.shape[0]),
        n_blocks=len(dataset),
        block_size=dataset.block_size,
        tokenizer_hash=tokenizer_hash,
    )


def load_packed_token_file(
    path: Path | str,
    *,
    expected_tokenizer_hash: str,
    pad_id: int,
) -> PackedTokenDataset:
    src = Path(path)
    # Mirror the ".npz" normalization applied at save time so load(path) finds
    # the file even when the caller omits the extension.
    if src.suffix != ".npz":
        src = src.with_name(src.name + ".npz")
    manifest_path = src.with_suffix(src.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _is_str_mapping(manifest):
        raise TypeError("packed token manifest must be an object")
    tokenizer_hash = manifest.get("tokenizer_hash")
    if tokenizer_hash != expected_tokenizer_hash:
        raise ValueError(
            f"packed token tokenizer hash mismatch: got {tokenizer_hash!r}, "
            f"expected {expected_tokenizer_hash!r}"
        )
    block_size = manifest.get("block_size")
    if type(block_size) is not int:
        raise TypeError("packed token manifest block_size must be an integer")
    with np.load(src) as data:
        if "tokens" not in data:
            raise ValueError("packed token file missing tokens array")
        tokens = data["tokens"]
    if tokens.ndim != 1:
        raise ValueError("packed token file tokens must be 1D")
    return PackedTokenDataset(
        tokens=[int(token) for token in tokens.tolist()],
        block_size=block_size,
        pad_id=pad_id,
    )


def _messages_to_text(messages: Sequence[Mapping[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>{msg['content']}<|eot|>")
    return "".join(parts)


def _read_messages(raw: Mapping[str, object], key: str, source: str) -> list[Mapping[str, str]]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{source}.{key} must be a list")
    messages: list[Mapping[str, str]] = []
    for idx, item in enumerate(value):
        if not _is_str_mapping(item):
            raise TypeError(f"{source}.{key}[{idx}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{source}.{key}[{idx}].role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError(f"{source}.{key}[{idx}].content must be a string")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError(f"{source}.{key} must be non-empty")
    return messages


def _read_nonempty_str(raw: Mapping[str, object], key: str, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{key} must be a non-empty string")
    return value


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
