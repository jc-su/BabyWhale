"""Rollout buffer — the boundary between rollout/reward producers and trainer consumer.

The synchronous variant is a typed list. The asynchronous variant runs a
producer thread that pushes scored samples into a thread-safe deque while the
trainer consumes from the other side. Both implement the same Protocol so the
trainer code is unchanged.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

from baby_whale_v4.rl.types import ScoredSample


class RolloutBuffer(Protocol):
    def add(self, scored: ScoredSample) -> None: ...
    def add_many(self, scored: Iterable[ScoredSample]) -> None: ...
    def drain(self) -> list[ScoredSample]: ...
    def __len__(self) -> int: ...


class SyncRolloutBuffer:
    """List-backed buffer. ``drain()`` returns the contents and clears."""

    def __init__(self) -> None:
        self._items: list[ScoredSample] = []

    def add(self, scored: ScoredSample) -> None:
        if not isinstance(scored, ScoredSample):
            raise TypeError("SyncRolloutBuffer.add expects a ScoredSample")
        self._items.append(scored)

    def add_many(self, scored: Iterable[ScoredSample]) -> None:
        for item in scored:
            self.add(item)

    def drain(self) -> list[ScoredSample]:
        items, self._items = self._items, []
        return items

    def peek(self) -> Sequence[ScoredSample]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ScoredSample]:
        return iter(tuple(self._items))


class AsyncRolloutBuffer:
    """Bounded thread-safe buffer driven by a producer callable.

    Spawn the producer with ``start(producer_fn)``; ``producer_fn`` is invoked
    with this buffer and is expected to call ``add(...)`` until done, then
    return. ``drain()`` blocks until at least ``min_count`` scored samples are
    available, then returns the full current snapshot.
    """

    def __init__(self, *, max_size: int = 1024) -> None:
        if max_size <= 0:
            raise ValueError("AsyncRolloutBuffer.max_size must be positive")
        self.max_size = max_size
        self._items: deque[ScoredSample] = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._producer_done = threading.Event()
        self._producer_error: BaseException | None = None

    def add(self, scored: ScoredSample) -> None:
        if not isinstance(scored, ScoredSample):
            raise TypeError("AsyncRolloutBuffer.add expects a ScoredSample")
        with self._not_full:
            while len(self._items) >= self.max_size:
                self._not_full.wait()
            self._items.append(scored)
            self._not_empty.notify_all()

    def add_many(self, scored: Iterable[ScoredSample]) -> None:
        for item in scored:
            self.add(item)

    def drain(self, *, min_count: int = 1, timeout: float | None = None) -> list[ScoredSample]:
        if min_count < 0:
            raise ValueError("min_count must be non-negative")
        with self._not_empty:
            ok = self._not_empty.wait_for(
                lambda: len(self._items) >= min_count or self._producer_done.is_set(),
                timeout=timeout,
            )
            if not ok:
                raise TimeoutError("AsyncRolloutBuffer.drain timed out")
            self._raise_if_producer_failed()
            items = list(self._items)
            self._items.clear()
            self._not_full.notify_all()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def start(self, producer: Callable[[AsyncRolloutBuffer], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncRolloutBuffer producer already running")
        self._producer_done.clear()
        self._producer_error = None

        def _run() -> None:
            try:
                producer(self)
            except BaseException as exc:
                self._producer_error = exc
            finally:
                with self._lock:
                    self._producer_done.set()
                    self._not_empty.notify_all()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)
        self._raise_if_producer_failed()

    def _raise_if_producer_failed(self) -> None:
        if self._producer_error is not None:
            raise self._producer_error
