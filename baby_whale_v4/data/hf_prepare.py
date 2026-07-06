import hashlib
import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeIs, cast

type DatasetKind = Literal["pretrain", "chat", "preference", "tool_trace"]
type NormalizedRecord = dict[str, object]


class _LoadDatasetFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ShuffleDatasetFn(Protocol):
    def __call__(self, *, seed: int, buffer_size: int) -> object: ...


@dataclass(frozen=True)
class HFSource:
    dataset_id: str
    split: str
    kind: DatasetKind
    subset: str | None = None
    text_field: str = "text"
    messages_field: str = "messages"
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    tools_field: str = "tools"
    limit: int = 1000
    seed: int = 0
    license_note: str = "check upstream dataset card"
    streaming: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        if self.kind not in ("pretrain", "chat", "preference", "tool_trace"):
            raise ValueError(f"unsupported dataset kind {self.kind!r}")
        if self.limit <= 0:
            raise ValueError("limit must be positive")

    @property
    def slug(self) -> str:
        raw = f"{self.dataset_id}:{self.subset or 'default'}:{self.split}:{self.kind}:{self.limit}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class MaterializedDataset:
    path: Path
    manifest_path: Path
    rows: int
    source: HFSource

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("materialized dataset path must be a Path")
        if not isinstance(self.manifest_path, Path):
            raise TypeError("materialized dataset manifest_path must be a Path")
        if self.rows <= 0:
            raise ValueError("materialized dataset rows must be positive")


def materialize_hf_source(source: HFSource, out_dir: Path | str) -> MaterializedDataset:
    datasets_mod = importlib.import_module("datasets")
    load_dataset = cast(_LoadDatasetFn, datasets_mod.load_dataset)
    kwargs: dict[str, object] = {
        "split": source.split,
        "streaming": source.streaming,
    }
    if source.subset is None:
        ds = load_dataset(source.dataset_id, **kwargs)
    else:
        ds = load_dataset(source.dataset_id, source.subset, **kwargs)
    if source.seed is not None:
        shuffle = cast(_ShuffleDatasetFn, object.__getattribute__(ds, "shuffle"))
        ds = shuffle(seed=source.seed, buffer_size=max(100, min(10_000, source.limit * 10)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / f"{source.slug}.{source.kind}.jsonl"
    rows = write_normalized_jsonl(_iter_rows(ds), source=source, path=data_path)
    manifest_path = data_path.with_suffix(".manifest.json")
    manifest = {
        "dataset_id": source.dataset_id,
        "subset": source.subset,
        "split": source.split,
        "kind": source.kind,
        "rows": rows,
        "source": asdict(source),
        "data_sha256": _sha256_file(data_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return MaterializedDataset(
        path=data_path, manifest_path=manifest_path, rows=rows, source=source
    )


def write_normalized_jsonl(rows: Iterable[object], *, source: HFSource, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            if n >= source.limit:
                break
            record = normalize_hf_row(row, source=source, source_index=index)
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise ValueError("dataset materialization produced zero rows")
    return n


def normalize_hf_row(row: object, *, source: HFSource, source_index: int) -> NormalizedRecord:
    if not _is_str_mapping(row):
        raise TypeError("dataset rows must be objects with string keys")
    match source.kind:
        case "pretrain":
            text = _read_nonempty_str(row, source.text_field)
            return _with_source(source, source_index, {"kind": "pretrain", "text": text})
        case "chat":
            messages = _read_messages(row, source.messages_field)
            return _with_source(source, source_index, {"kind": "chat", "messages": messages})
        case "preference":
            prompt = _read_nonempty_str(row, source.prompt_field)
            chosen = _read_nonempty_str(row, source.chosen_field)
            rejected = _read_nonempty_str(row, source.rejected_field)
            return _with_source(
                source,
                source_index,
                {"kind": "preference", "prompt": prompt, "chosen": chosen, "rejected": rejected},
            )
        case "tool_trace":
            messages = _read_messages(row, source.messages_field)
            tools = row.get(source.tools_field, [])
            if not isinstance(tools, list):
                raise TypeError(f"row.{source.tools_field} must be a list")
            return _with_source(
                source,
                source_index,
                {"kind": "tool_trace", "messages": messages, "tools": tools},
            )


def _with_source(source: HFSource, source_index: int, record: NormalizedRecord) -> NormalizedRecord:
    record["source"] = {
        "dataset_id": source.dataset_id,
        "subset": source.subset,
        "split": source.split,
        "index": source_index,
        "license_note": source.license_note,
    }
    return record


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _read_nonempty_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row.{key} must be a non-empty string")
    return value


def _read_messages(row: Mapping[str, object], key: str) -> list[dict[str, str]]:
    value = row.get(key)
    if not isinstance(value, list):
        raise TypeError(f"row.{key} must be a list")
    messages: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not _is_str_mapping(item):
            raise TypeError(f"row.{key}[{idx}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"row.{key}[{idx}].role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError(f"row.{key}[{idx}].content must be a string")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError(f"row.{key} must not be empty")
    return messages


def _iter_rows(ds: object) -> Iterable[object]:
    if not isinstance(ds, Iterable):
        raise TypeError("loaded dataset must be iterable")
    return ds


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
