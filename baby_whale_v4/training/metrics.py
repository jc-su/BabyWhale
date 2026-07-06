import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO

type MetricValue = str | int | float | bool | None
type MetricRecord = Mapping[str, MetricValue]


def _format_line(record: MetricRecord) -> str:
    """One-line human-readable rendering of a metric record.

    Ordering: ``step`` first, then any ``*_loss`` field, then the rest
    alphabetically. Floats are rounded to 4 decimals; ``None`` values dropped.
    """

    def sort_key(item: tuple[str, MetricValue]) -> tuple[int, str]:
        k = item[0]
        if k == "step":
            return (0, k)
        if k.endswith("_loss"):
            return (1, k)
        return (2, k)

    parts: list[str] = []
    for k, v in sorted(record.items(), key=sort_key):
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


@dataclass
class JsonlMetrics:
    path: Path
    echo: bool = True
    _file: IO[str] | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", buffering=1)

    def log(self, record: MetricRecord) -> None:
        if self._file is None:
            raise RuntimeError("metrics file is closed")
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        if self.echo:
            tag = self.path.stem.replace("_metrics", "")
            print(f"[{tag}] {_format_line(record)}", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlMetrics:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
