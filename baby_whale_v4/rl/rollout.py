"""Rollout engines: the boundary between the trainer and the inference path.

`InProcessRolloutEngine` wraps `inference.Engine` so GRPO-style training reuses
the production inference stack — chunked prefill, prefix cache, and per-token
log-prob capture come for free. `HTTPRolloutEngine` talks to `inference.server`
in another process, mirroring the verl/SLIME split between training cluster
and rollout cluster.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

import mlx.core as mx

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.data.tokenizer import Tokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions, RequestState
from baby_whale_v4.inference.prefix_cache import PrefixCache
from baby_whale_v4.inference.scheduler import RequestScheduler
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.rl.types import RolloutRequest, RolloutSample
from baby_whale_v4.tools.local import ToolRegistry
from baby_whale_v4.tools.schema import (
    TOOL_CALL_CLOSE_TAG,
    ToolCall,
    parse_tool_call_text,
    render_tool_result,
)
from baby_whale_v4.typing import RequestId, TokenizerHash


class RolloutEngine(Protocol):
    def generate_batch(self, requests: Sequence[RolloutRequest]) -> list[RolloutSample]: ...

    def sync_weights(self, model: BabyWhaleV4Model) -> None: ...


class InProcessRolloutEngine:
    """Drives `inference.Engine` with chunked prefill + prefix-cache reuse.

    Submits all requests to a `RequestScheduler` so a system prompt that's shared
    across a GRPO group only pays prefill once.
    """

    def __init__(
        self,
        *,
        model: BabyWhaleV4Model,
        config: BabyWhaleV4Config,
        tokenizer_hash: TokenizerHash,
        prefix_cache_capacity: int = 64,
        prefill_chunk: int = 32,
    ):
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        if prefix_cache_capacity <= 0:
            raise ValueError("prefix_cache_capacity must be positive")
        self.model = model
        self.config = config
        self.tokenizer_hash = tokenizer_hash
        self.prefill_chunk = prefill_chunk
        self.prefix_cache = PrefixCache(capacity=prefix_cache_capacity)
        self.engine = Engine(
            model=model,
            config=config,
            tokenizer_hash=tokenizer_hash,
            prefix_cache=self.prefix_cache,
        )

    def generate_batch(self, requests: Sequence[RolloutRequest]) -> list[RolloutSample]:
        if not requests:
            return []
        was_training = self.model.training
        self.model.eval()
        try:
            scheduler = RequestScheduler(self.engine, prefill_chunk=self.prefill_chunk)
            states: list[tuple[RolloutRequest, RequestState] | None] = [None] * len(requests)
            groups: dict[tuple[int, ...], list[tuple[int, RolloutRequest]]] = {}
            group_order: list[tuple[int, ...]] = []
            for i, request in enumerate(requests):
                key = tuple(request.prompt_ids)
                if key not in groups:
                    groups[key] = []
                    group_order.append(key)
                groups[key].append((i, request))

            for key in group_order:
                group = groups[key]
                first_i, first_request = group[0]
                state = scheduler.submit(
                    request_id=f"rollout-{first_i}",
                    prompt_ids=list(first_request.prompt_ids),
                    options=first_request.options,
                )
                states[first_i] = (first_request, state)
                while state.remaining_prefill > 0:
                    scheduler.tick()
                for i, request in group[1:]:
                    warmed = scheduler.submit(
                        request_id=f"rollout-{i}",
                        prompt_ids=list(request.prompt_ids),
                        options=request.options,
                    )
                    states[i] = (request, warmed)
            scheduler.run_until_done()
            samples: list[RolloutSample] = []
            for item in states:
                if item is None:
                    raise RuntimeError("rollout state construction left an empty slot")
                req, state = item
                samples.append(_build_sample(req, state))
            return samples
        finally:
            if was_training:
                self.model.train()

    def sync_weights(self, model: BabyWhaleV4Model) -> None:
        # Same Python object — but the underlying weights changed. The KV/prefix
        # caches we hold were computed against the old weights, so they must be
        # invalidated before the next rollout.
        if model is not self.model:
            raise ValueError(
                "InProcessRolloutEngine.sync_weights must be called with the same "
                "model instance the engine was constructed against"
            )
        self.prefix_cache.clear()

    def generate_with_tools(
        self,
        request: RolloutRequest,
        *,
        tokenizer: Tokenizer,
        registry: ToolRegistry,
        max_turns: int = 4,
    ) -> RolloutSample:
        """Multi-turn rollout: model emits a ``<tool_call>...</tool_call>`` block,
        engine executes it via ``registry``, and the rendered ``<tool_result>``
        is fed back through the model's KV cache so generation can resume.

        Returns a single :class:`RolloutSample` whose ``response_ids`` is the
        full transcript (model output + injected tool results) and whose
        ``log_probs`` covers the model-generated positions only — tool result
        tokens are not on-policy and carry zero log-prob.
        """

        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        was_training = self.model.training
        self.model.eval()
        try:
            state = self.engine.new_request(
                RequestId("rollout-tool"), list(request.prompt_ids), request.options
            )
            while state.remaining_prefill > 0:
                self.engine.prefill_chunk(state, self.prefill_chunk)
            self.engine.commit_prefix_cache(state)

            response_ids: list[int] = []
            log_probs: list[float] = []
            tool_calls: list[ToolCall] = []
            turn = 0
            budget = request.options.max_new_tokens
            while turn < max_turns and len(response_ids) < budget:
                turn += 1
                turn_start = len(response_ids)
                while not state.finished and len(response_ids) < budget:
                    before_decode = state.total_emitted
                    self.engine.decode_step(state)
                    if state.total_emitted == before_decode:
                        break
                    response_ids.append(state.generated[-1])
                    log_probs.append(state.captured_log_probs[-1])
                    decoded = tokenizer.decode(response_ids[turn_start:])
                    if TOOL_CALL_CLOSE_TAG in decoded:
                        break
                turn_text = tokenizer.decode(response_ids[turn_start:])
                if TOOL_CALL_CLOSE_TAG not in turn_text:
                    break
                try:
                    call = parse_tool_call_text(turn_text)
                except ValueError:
                    break
                tool_calls.append(call)
                result = registry.execute(call)
                result_ids = tokenizer.encode(render_tool_result(result))
                if not result_ids:
                    break
                if (
                    state.prefilled + len(state.generated) + len(result_ids)
                    > self.config.context_length
                ):
                    break
                result_arr = mx.array([result_ids], dtype=mx.int32)
                out = self.model(result_arr, cache=state.cache)
                state.last_logits = out.logits[:, -1, :]
                response_ids.extend(int(t) for t in result_ids)
                log_probs.extend(0.0 for _ in result_ids)
                state.finished = False
            return RolloutSample(
                request=request,
                response_ids=tuple(response_ids),
                log_probs=tuple(log_probs),
                finished=state.finished,
                tool_calls=tuple(tool_calls),
            )
        finally:
            if was_training:
                self.model.train()


class HTTPRolloutEngine:
    """Rollout engine that POSTs to ``inference/server.py`` /generate.

    The protocol on the wire is intentionally narrow:

      Request:  {"prompt_ids": [int, ...], "options": {...}}
      Response: {"response_ids": [int, ...], "log_probs": [float, ...],
                 "finished": bool}

    Multi-turn / tool-call rollouts compose by issuing follow-up requests with
    the previous turn's tokens appended; this engine does not attempt to drive
    the loop itself.
    """

    def __init__(self, url: str, *, timeout_sec: float = 60.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("HTTPRolloutEngine.url must be http:// or https://")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.url = url.rstrip("/")
        self.timeout_sec = timeout_sec

    def generate_batch(self, requests: Sequence[RolloutRequest]) -> list[RolloutSample]:
        return [self._generate_one(request) for request in requests]

    def sync_weights(self, model: BabyWhaleV4Model) -> None:
        """Push the model's current weights to the rollout server.

        Saves a weight-only checkpoint to a temporary path, then POSTs the path
        to ``{url}/sync_weights``. The server validates ``config_hash`` against
        its own config and rejects a mismatched architecture. After a successful
        load the server clears its prefix cache so the next rollout reflects the
        updated policy. The temp file is removed once the server confirms.
        """

        import tempfile
        from pathlib import Path

        from baby_whale_v4.training.checkpoint import save_checkpoint

        with tempfile.NamedTemporaryFile(
            prefix="bwv4_sync_", suffix=".ckpt", delete=False
        ) as handle:
            checkpoint_path = Path(handle.name)
        try:
            save_checkpoint(
                checkpoint_path,
                config=model.config,
                model=model,
                optimizer=None,
                scheduler=None,
                step=0,
            )
            body = json.dumps({"checkpoint_path": str(checkpoint_path)}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.url}/sync_weights",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                finally:
                    exc.close()
                raise RuntimeError(
                    f"sync_weights request failed: HTTP {exc.code}: {detail[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"sync_weights request failed: {exc}") from exc
            if not isinstance(payload, dict) or payload.get("loaded") is not True:
                raise RuntimeError(f"sync_weights server response unexpected: {payload!r}")
        finally:
            with contextlib.suppress(FileNotFoundError):
                checkpoint_path.unlink()

    def _generate_one(self, request: RolloutRequest) -> RolloutSample:
        body = json.dumps(
            {
                "prompt_ids": list(request.prompt_ids),
                "options": _encode_options(request.options),
            }
        ).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.url}/rollout",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTPRolloutEngine request failed: {exc}") from exc
        return _decode_sample(request, payload)


def _build_sample(request: RolloutRequest, state: RequestState) -> RolloutSample:
    return RolloutSample(
        request=request,
        response_ids=tuple(int(t) for t in state.generated),
        log_probs=tuple(float(p) for p in state.captured_log_probs),
        finished=state.finished,
    )


def _encode_options(options: GenerationOptions) -> dict:
    return {
        "max_new_tokens": options.max_new_tokens,
        "mode": options.mode,
        "temperature": options.temperature,
        "top_k": options.top_k,
        "eos_id": options.eos_id,
    }


def _decode_sample(request: RolloutRequest, payload: dict) -> RolloutSample:
    response_ids = payload.get("response_ids")
    log_probs = payload.get("log_probs")
    finished = payload.get("finished")
    if not isinstance(response_ids, list) or not all(isinstance(t, int) for t in response_ids):
        raise TypeError("rollout payload response_ids must be a list[int]")
    if not isinstance(log_probs, list) or not all(isinstance(p, (int, float)) for p in log_probs):
        raise TypeError("rollout payload log_probs must be a list[float]")
    if not isinstance(finished, bool):
        raise TypeError("rollout payload finished must be a bool")
    return RolloutSample(
        request=request,
        response_ids=tuple(response_ids),
        log_probs=tuple(float(p) for p in log_probs),
        finished=finished,
    )
