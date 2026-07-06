import operator
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, SupportsIndex

import mlx.core as mx
import numpy as np

type TensorPair = tuple[mx.array, mx.array]


@dataclass
class PackedDataset:
    documents: Sequence[Sequence[int]]
    block_size: int
    bos_id: int
    eos_id: int
    pad_id: int
    drop_last: bool = True

    _packed: mx.array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_size <= 1:
            raise ValueError("block_size must be > 1")
        flat: list[int] = []
        for doc in self.documents:
            if not doc:
                continue
            flat.append(self.bos_id)
            flat.extend(int(t) for t in doc)
            flat.append(self.eos_id)
        usable = (len(flat) - 1) // self.block_size
        if usable <= 0:
            if self.drop_last:
                raise ValueError(
                    f"not enough tokens to form one block of {self.block_size + 1}; got {len(flat)}"
                )
            pad_n = (self.block_size + 1) - len(flat)
            flat = flat + [self.pad_id] * pad_n
            usable = 1
        flat = flat[: usable * self.block_size + 1]
        self._packed = mx.array(flat, dtype=mx.int32)

    @property
    def packed_tokens(self) -> mx.array:
        return self._packed

    def __len__(self) -> int:
        return (int(self._packed.shape[0]) - 1) // self.block_size

    def __getitem__(self, index: SupportsIndex) -> TensorPair:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.block_size
        end = start + self.block_size
        x = self._packed[start:end]
        y = self._packed[start + 1 : end + 1]
        return x, y


@dataclass
class PackedTokenDataset:
    tokens: Sequence[int]
    block_size: int
    pad_id: int
    drop_last: bool = True

    _packed: mx.array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_size <= 1:
            raise ValueError("block_size must be > 1")
        flat = [int(t) for t in self.tokens]
        if any(token_id < 0 for token_id in flat):
            raise ValueError("packed tokens must be non-negative")
        usable = (len(flat) - 1) // self.block_size
        if usable <= 0:
            if self.drop_last:
                raise ValueError(
                    f"not enough tokens to form one block of {self.block_size + 1}; got {len(flat)}"
                )
            pad_n = (self.block_size + 1) - len(flat)
            flat = flat + [self.pad_id] * pad_n
            usable = 1
        flat = flat[: usable * self.block_size + 1]
        self._packed = mx.array(flat, dtype=mx.int32)

    @property
    def packed_tokens(self) -> mx.array:
        return self._packed

    def __len__(self) -> int:
        return (int(self._packed.shape[0]) - 1) // self.block_size

    def __getitem__(self, index: SupportsIndex) -> TensorPair:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.block_size
        end = start + self.block_size
        x = self._packed[start:end]
        y = self._packed[start + 1 : end + 1]
        return x, y


@dataclass
class SyntheticCopyDataset:
    n_samples: int
    seq_len: int
    vocab_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.seq_len < 4 or self.seq_len % 2 != 0:
            raise ValueError("seq_len must be even and >= 4")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be >= 4")
        rng = np.random.default_rng(self.seed)
        half = self.seq_len // 2
        prompt = rng.integers(0, self.vocab_size, size=(self.n_samples, half), dtype=np.int32)
        full = np.concatenate((prompt, prompt.copy()), axis=1)
        self._x = mx.array(full[:, :-1], dtype=mx.int32)
        self._y = mx.array(full[:, 1:], dtype=mx.int32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: SupportsIndex) -> TensorPair:
        index = operator.index(index)
        return self._x[index], self._y[index]


@dataclass
class SyntheticNeedleDataset:
    n_samples: int
    seq_len: int
    vocab_size: int
    needle_id: int = 1
    answer_id: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        if max(self.needle_id, self.answer_id) >= self.vocab_size:
            raise ValueError("special token ids must be < vocab_size")
        rng = np.random.default_rng(self.seed)
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for _ in range(self.n_samples):
            base = rng.integers(3, self.vocab_size, size=(self.seq_len + 1,), dtype=np.int32)
            insert_pos = int(rng.integers(2, self.seq_len - 4))
            base[insert_pos] = self.needle_id
            base[insert_pos + 1] = self.answer_id
            base[-2] = self.needle_id
            base[-1] = self.answer_id
            x_list.append(base[:-1].copy())
            y_list.append(base[1:].copy())
        self._x = mx.array(np.stack(x_list), dtype=mx.int32)
        self._y = mx.array(np.stack(y_list), dtype=mx.int32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: SupportsIndex) -> TensorPair:
        index = operator.index(index)
        return self._x[index], self._y[index]


class TensorPairDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: SupportsIndex) -> TensorPair: ...


def iter_batches(
    dataset: TensorPairDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> Iterator[TensorPair]:
    n = len(dataset)
    if n <= 0:
        raise ValueError("dataset must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, n, batch_size):
        chunk = indices[start : start + batch_size]
        if not chunk:
            continue
        xs = []
        ys = []
        for i in chunk:
            x, y = dataset[i]
            xs.append(x)
            ys.append(y)
        yield mx.stack(xs, axis=0), mx.stack(ys, axis=0)
