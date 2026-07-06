import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Protocol, TypeIs, cast

from baby_whale_v4.typing import TokenizerHash, TokenizerKind


class Tokenizer(Protocol):
    kind: TokenizerKind
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def hash_signature(self) -> TokenizerHash: ...


@dataclass
class ByteTokenizer:
    kind: TokenizerKind = "byte"
    bos_id: int = 256
    eos_id: int = 257
    pad_id: int = 258
    _special_strings: tuple[str, ...] = ("<|bos|>", "<|eos|>", "<|pad|>")

    def __post_init__(self) -> None:
        if self.kind != "byte":
            raise ValueError("ByteTokenizer kind must be 'byte'")
        ids = (self.bos_id, self.eos_id, self.pad_id)
        if len(set(ids)) != len(ids):
            raise ValueError("special token ids must be distinct")
        if any(token_id < 256 for token_id in ids):
            raise ValueError("special token ids must be >= 256")
        if len(self._special_strings) != 3:
            raise ValueError("_special_strings must contain BOS/EOS/PAD strings")

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        bytes_ = bytearray()
        for tid in ids:
            if tid < 256:
                bytes_.append(tid)
            elif tid == self.bos_id:
                bytes_.extend(self._special_strings[0].encode("utf-8"))
            elif tid == self.eos_id:
                bytes_.extend(self._special_strings[1].encode("utf-8"))
            elif tid == self.pad_id:
                bytes_.extend(self._special_strings[2].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {tid}")
        return bytes_.decode("utf-8", errors="replace")

    def hash_signature(self) -> TokenizerHash:
        return TokenizerHash(f"byte-v1-{self.vocab_size}")


_BPE_BASE_VOCAB = 259
_TOKENIZER_KEYS = frozenset({"kind", "bos_id", "eos_id", "pad_id", "merges"})


@dataclass(frozen=True)
class ByteBPETokenizer:
    kind: TokenizerKind = "byte_bpe"
    merges: tuple[tuple[int, int], ...] = ()
    bos_id: int = 256
    eos_id: int = 257
    pad_id: int = 258
    _special_strings: tuple[str, ...] = ("<|bos|>", "<|eos|>", "<|pad|>")

    def __post_init__(self) -> None:
        if self.kind != "byte_bpe":
            raise ValueError("ByteBPETokenizer kind must be 'byte_bpe'")
        ids = (self.bos_id, self.eos_id, self.pad_id)
        if ids != (256, 257, 258):
            raise ValueError("ByteBPETokenizer special ids must be 256/257/258")
        if len(self._special_strings) != 3:
            raise ValueError("_special_strings must contain BOS/EOS/PAD strings")
        normalized: list[tuple[int, int]] = []
        for idx, item in enumerate(self.merges):
            if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
                raise TypeError(f"merge {idx} must be a pair of token ids")
            left, right = item
            if type(left) is not int or type(right) is not int:
                raise TypeError(f"merge {idx} token ids must be integers")
            max_existing = _BPE_BASE_VOCAB + idx
            if left < 0 or right < 0 or left >= max_existing or right >= max_existing:
                raise ValueError(
                    f"merge {idx} may only reference tokens < {max_existing}; got {(left, right)}"
                )
            normalized.append((left, right))
        object.__setattr__(self, "merges", tuple(normalized))

    @property
    def vocab_size(self) -> int:
        return _BPE_BASE_VOCAB + len(self.merges)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = _bpe_encode(list(text.encode("utf-8")), _merge_ranks(self.merges))
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        token_bytes = _build_bpe_token_bytes(self.merges)
        out = bytearray()
        for tid in ids:
            if tid in token_bytes:
                out.extend(token_bytes[tid])
            elif tid == self.bos_id:
                out.extend(self._special_strings[0].encode("utf-8"))
            elif tid == self.eos_id:
                out.extend(self._special_strings[1].encode("utf-8"))
            elif tid == self.pad_id:
                out.extend(self._special_strings[2].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {tid}")
        return out.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def from_dict(cls, data: object) -> ByteBPETokenizer:
        if not _is_str_mapping(data):
            raise TypeError("tokenizer payload must be an object with string keys")
        unknown = set(data) - _TOKENIZER_KEYS
        if unknown:
            raise ValueError(f"unknown tokenizer keys: {sorted(unknown)}")
        missing = _TOKENIZER_KEYS - set(data)
        if missing:
            raise ValueError(f"missing tokenizer keys: {sorted(missing)}")
        kind = data["kind"]
        if kind != "byte_bpe":
            raise ValueError(f"unsupported tokenizer kind {kind!r}")
        return cls(
            kind=cast(TokenizerKind, kind),
            bos_id=_read_int(data, "bos_id"),
            eos_id=_read_int(data, "eos_id"),
            pad_id=_read_int(data, "pad_id"),
            merges=_read_merges(data["merges"]),
        )

    @classmethod
    def load(cls, path: Path | str) -> ByteBPETokenizer:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tokenizer JSON: {exc}") from exc
        return cls.from_dict(raw)

    def hash_signature(self) -> TokenizerHash:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return TokenizerHash(f"byte-bpe-v1-{self.vocab_size}-{digest}")


def train_byte_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    min_pair_count: int = 2,
) -> ByteBPETokenizer:
    if vocab_size < _BPE_BASE_VOCAB:
        raise ValueError(f"vocab_size must be >= {_BPE_BASE_VOCAB}")
    if min_pair_count < 2:
        raise ValueError("min_pair_count must be >= 2")
    docs: list[list[int]] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{idx}] must be a string")
        encoded = list(text.encode("utf-8"))
        if encoded:
            docs.append(encoded)
    if not docs:
        raise ValueError("cannot train tokenizer on zero non-empty texts")

    merges: list[tuple[int, int]] = []
    while _BPE_BASE_VOCAB + len(merges) < vocab_size:
        counts = _pair_counts(docs)
        if not counts:
            break
        best_count = max(counts.values())
        if best_count < min_pair_count:
            break
        best_pair = min(pair for pair, count in counts.items() if count == best_count)
        token_id = _BPE_BASE_VOCAB + len(merges)
        docs = [_merge_tokens(doc, best_pair, token_id) for doc in docs]
        merges.append(best_pair)
    return ByteBPETokenizer(merges=tuple(merges))


def load_tokenizer(path: Path | str | None) -> Tokenizer:
    if path is None:
        return ByteTokenizer()
    return ByteBPETokenizer.load(path)


def _pair_counts(docs: Sequence[Sequence[int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for doc in docs:
        for pair in pairwise(doc):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


@lru_cache(maxsize=8)
def _merge_ranks(merges: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    """Map each learned merge pair to its rank (merge index). Cached per tokenizer."""
    return {pair: idx for idx, pair in enumerate(merges)}


def _bpe_encode(ids: list[int], ranks: dict[tuple[int, int], int]) -> list[int]:
    """BPE-encode a byte sequence in O(len x log len) via a heap of adjacent-pair ranks.

    Output is identical to applying the learned merges in training order: a merge
    only creates a token referenced by strictly-later merges, so merging the
    globally-lowest-rank pair (one occurrence at a time, left-to-right) visits the
    merges in the same order as the old global rescan — but without its
    O(len x n_merges) per-line cost that hung on long lines. Positions never move
    (a doubly-linked list tracks neighbours); stale heap entries are skipped.
    """
    n = len(ids)
    if n < 2:
        return list(ids)
    tokens = list(ids)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))  # nxt[n-1] == n is the past-the-end sentinel
    alive = [True] * n
    heap: list[tuple[int, int]] = [
        (rank, i)
        for i in range(n - 1)
        if (rank := ranks.get((tokens[i], tokens[i + 1]))) is not None
    ]
    heapq.heapify(heap)
    while heap:
        rank, i = heapq.heappop(heap)
        j = nxt[i]
        if not alive[i] or j >= n or not alive[j]:
            continue
        if ranks.get((tokens[i], tokens[j])) != rank:
            continue  # stale entry — the pair at this position has since changed
        tokens[i] = _BPE_BASE_VOCAB + rank
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k < n:
            prev[k] = i
        left = prev[i]
        if left >= 0 and (r := ranks.get((tokens[left], tokens[i]))) is not None:
            heapq.heappush(heap, (r, left))
        if k < n and (r := ranks.get((tokens[i], tokens[k]))) is not None:
            heapq.heappush(heap, (r, i))
    return [tokens[idx] for idx in range(n) if alive[idx]]


def _merge_tokens(tokens: Sequence[int], pair: tuple[int, int], new_token: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_token)
            i += 2
        else:
            out.append(int(tokens[i]))
            i += 1
    return out


def _build_bpe_token_bytes(merges: Sequence[tuple[int, int]]) -> dict[int, bytes]:
    table: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for idx, (left, right) in enumerate(merges):
        token_id = _BPE_BASE_VOCAB + idx
        if left not in table or right not in table:
            raise ValueError(f"merge {idx} references unknown token {(left, right)}")
        table[token_id] = table[left] + table[right]
    return table


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _read_int(data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if type(value) is not int:
        raise TypeError(f"tokenizer.{key} must be an integer")
    return value


def _read_merges(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise TypeError("tokenizer.merges must be a list")
    merges: list[tuple[int, int]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(f"tokenizer.merges[{idx}] must be a pair")
        left, right = item
        if type(left) is not int or type(right) is not int:
            raise TypeError(f"tokenizer.merges[{idx}] values must be integers")
        merges.append((left, right))
    return tuple(merges)
