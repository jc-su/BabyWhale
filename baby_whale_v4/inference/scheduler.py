from collections import deque
from dataclasses import dataclass

from baby_whale_v4.cache import DynamicKVCache
from baby_whale_v4.inference.batched import decode_group_batched, decode_ragged_batched
from baby_whale_v4.inference.engine import Engine, GenerationOptions, RequestState
from baby_whale_v4.typing import RequestId


def _sampling_sig(options: GenerationOptions) -> tuple[object, ...]:
    """Sampling params that must match for two requests to share a batched forward.

    Finish conditions (eos_id, max_new_tokens) are handled per-request, so they
    are deliberately excluded — requests wanting different lengths still batch.
    """
    return (options.mode, options.temperature, options.top_k, options.top_p, options.min_p)


@dataclass
class _Stats:
    prefill_steps: int = 0
    decode_steps: int = 0
    completed: int = 0


class RequestScheduler:
    """Round-robin scheduler with chunked prefill.

    Per tick:
      1. Run one decode step for each currently-decoding request (anti-starvation).
      2. Run one prefill chunk for the head of the prefill queue.
    """

    def __init__(self, engine: Engine, *, prefill_chunk: int = 4, ragged: bool = False):
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        self.engine = engine
        self.prefill_chunk = prefill_chunk
        # Ragged batching (one forward for a *mixed-length* cohort) needs an
        # all-sliding_mqa model; otherwise we fall back to same-length cohorts.
        self._use_ragged = ragged and all(
            kind == "sliding_mqa" for kind in engine.model.config.effective_layer_schedule
        )
        self._prefill_queue: deque[RequestState] = deque()
        self._decode_queue: list[RequestState] = []
        self._completed: list[RequestState] = []
        self.stats = _Stats()

    def submit(
        self, request_id: str, prompt_ids: list[int], options: GenerationOptions
    ) -> RequestState:
        state = self.engine.new_request(RequestId(request_id), prompt_ids, options)
        if state.remaining_prefill > 0:
            self._prefill_queue.append(state)
        else:
            self._decode_queue.append(state)
        return state

    def has_work(self) -> bool:
        return bool(self._prefill_queue) or bool(self._decode_queue)

    def tick(self) -> None:
        self._decode_step()
        self._prefill_step()

    def _prefill_step(self) -> None:
        # Prefill every queued request by one chunk this tick (dropping cancelled
        # ones) so requests admitted together stay length-aligned and can batch in
        # the decode phase. Decode still runs every tick, so this keeps the
        # anti-starvation property while enabling cohort batching. Completed
        # prefills move to the decode queue.
        still_prefilling: deque[RequestState] = deque()
        while self._prefill_queue:
            state = self._prefill_queue.popleft()
            if state.cancelled:
                self._retire(state)
                continue
            self.engine.prefill_chunk(state, self.prefill_chunk)
            self.stats.prefill_steps += 1
            if state.remaining_prefill == 0:
                self.engine.commit_prefix_cache(state)
                self._decode_queue.append(state)
            else:
                still_prefilling.append(state)
        self._prefill_queue = still_prefilling

    def _decode_step(self) -> None:
        # Group decoding requests by (sequence length, sampling params) and run
        # one batched forward per cohort of >= 2 — continuous batching. Uniform
        # length ⇒ identical positions/mask across the batch, so no model change
        # is needed; singletons (and paged/speculative) decode per-request.
        active: list[RequestState] = []
        for state in self._decode_queue:
            if state.cancelled or state.finished or state.remaining_decode == 0:
                self._retire(state)
            else:
                active.append(state)

        groups: dict[object, list[RequestState]] = {}
        for state in active:
            if self._use_ragged:
                key: object = _sampling_sig(state.options)  # length-agnostic cohort
            else:
                length = state.cache.max_sequence_length() if state.cache is not None else 0
                key = (length, _sampling_sig(state.options))
            groups.setdefault(key, []).append(state)

        next_decode: list[RequestState] = []
        for group in groups.values():
            self._advance(group)
            for state in group:
                if state.cancelled or state.finished or state.remaining_decode == 0:
                    self._retire(state)
                else:
                    next_decode.append(state)
        self._decode_queue = next_decode

    def _advance(self, group: list[RequestState]) -> None:
        if (
            len(group) >= 2
            and group[0].options.mode != "speculative"
            and isinstance(group[0].cache, DynamicKVCache)
        ):
            if self._use_ragged:
                decode_ragged_batched(self.engine.model, group)
            else:
                decode_group_batched(self.engine.model, group)
            self.stats.decode_steps += len(group)
        else:
            for state in group:
                self.engine.decode_step(state)
                self.stats.decode_steps += 1

    def _retire(self, state: RequestState) -> None:
        state.finished = True
        self._completed.append(state)
        self.stats.completed += 1

    def run_until_done(self, max_ticks: int = 10000) -> list[RequestState]:
        ticks = 0
        while self.has_work():
            if ticks >= max_ticks:
                raise RuntimeError("scheduler exceeded max_ticks; possible deadlock")
            self.tick()
            ticks += 1
        return list(self._completed)
