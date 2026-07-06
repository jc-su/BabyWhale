"""Continuous-batching serving loop.

The dense :class:`~baby_whale_v4.inference.scheduler.RequestScheduler` already
interleaves decode across every in-flight request (no head-of-line blocking),
but the simple HTTP server drives it one request at a time. This module runs the
scheduler on a **single background thread** — the only thread that touches the
engine/model, since MLX and the model are not thread-safe — while HTTP handler
threads submit work and wait on per-request signals.

That is continuous batching at the serving layer: a freshly-arrived request
immediately starts sharing decode steps with the ones already running instead of
queueing behind them. (Decode is still one B=1 forward per request per tick — the
interleaving/scheduling is the educational point; true ragged kernel batching
would need a paged-attention kernel we deliberately don't build.)

Threading contract:
  * handler threads → :meth:`BatchingServer.submit` (enqueue) / :meth:`run_control`.
  * the loop thread → the *only* caller of the scheduler / engine / model.
  * results flow back via a per-request :class:`RequestHandle` (a token queue for
    streaming plus a ``done`` event for the final state).
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from baby_whale_v4.inference.engine import Engine, GenerationOptions, RequestState
from baby_whale_v4.inference.scheduler import RequestScheduler

_TOKEN_SENTINEL = object()  # end-of-stream marker pushed onto a handle's queue


class RequestHandle:
    """Client-side handle for one submitted request.

    The loop thread pushes decoded token ids onto ``_tokens`` as they are
    produced and sets ``done`` when the request finishes (or fails, recording
    ``error``). Handler threads consume via :meth:`next_token` (streaming) or
    :meth:`result` (whole response).
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.error: Exception | None = None
        self.state: RequestState | None = None
        self._tokens: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._emitted = 0
        self._cancelled = False

    # ---- loop-thread side -------------------------------------------------
    def _bind(self, state: RequestState) -> None:
        self.state = state

    def _pump(self) -> None:
        """Push any newly-generated tokens onto the queue (idempotent)."""
        if self.state is None:
            return
        generated = self.state.generated
        while self._emitted < len(generated):
            self._tokens.put(generated[self._emitted])
            self._emitted += 1

    def _finish(self) -> None:
        self._pump()
        self._tokens.put(_TOKEN_SENTINEL)
        self._done.set()

    def _fail(self, exc: Exception) -> None:
        self.error = exc
        self._tokens.put(_TOKEN_SENTINEL)
        self._done.set()

    # ---- handler-thread side ---------------------------------------------
    def next_token(self, timeout: float) -> int | None:
        """Return the next decoded token id, or ``None`` at end-of-stream.

        After ``None`` is returned, check :attr:`error` for a failure. Raises
        ``queue.Empty`` (via timeout) if the loop stalls.
        """
        item = self._tokens.get(timeout=timeout)
        return None if item is _TOKEN_SENTINEL else item

    def result(self, timeout: float) -> RequestState:
        """Block until the request completes and return its final state."""
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("request did not complete within timeout")
        if self.error is not None:
            raise self.error
        if self.state is None:
            raise RuntimeError("request produced no state")
        return self.state

    def cancel(self) -> None:
        """Ask the serving loop to stop generating (e.g. the SSE client hung up).

        Sets a flag the loop propagates to the request's ``cancelled`` state on
        its next tick (the scheduler then drops it); safe to call from any thread.
        """
        self._cancelled = True
        state = self.state
        if state is not None:
            state.cancel()


@dataclass
class _Submission:
    prompt_ids: list[int]
    options: GenerationOptions
    handle: RequestHandle


@dataclass
class _Control:
    fn: Callable[[Engine], object]
    done: threading.Event
    box: dict[str, object]


class BatchingServer:
    """Runs a :class:`RequestScheduler` on a background thread for concurrent HTTP serving."""

    def __init__(
        self, engine: Engine, *, prefill_chunk: int = 4, idle_timeout: float = 0.05
    ) -> None:
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self._engine = engine
        self._scheduler = RequestScheduler(engine, prefill_chunk=prefill_chunk)
        self._pending: queue.Queue[_Submission] = queue.Queue()
        self._control: queue.Queue[_Control] = queue.Queue()
        self._tracked: list[RequestHandle] = []
        self._wake = threading.Event()
        self._idle_timeout = idle_timeout
        self._running = False
        self._thread: threading.Thread | None = None
        self._ids = itertools.count()
        self._id_lock = threading.Lock()

    @property
    def engine(self) -> Engine:
        return self._engine

    def start(self) -> None:
        if self._running:
            return
        self._warmup()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="bwv4-serving-loop", daemon=True)
        self._thread.start()

    def _warmup(self) -> None:
        # Force MLX/Metal initialization on the calling (main) thread. The loop
        # thread's first model forward can otherwise hang if it is the process's
        # very first Metal dispatch — warming here makes background forwards safe.
        out = self._engine.model(mx.array([[0]], dtype=mx.int32))
        mx.eval(out.logits)

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None

    def _next_id(self, prefix: str) -> str:
        with self._id_lock:
            return f"{prefix}-{next(self._ids)}"

    def submit(
        self,
        prompt_ids: list[int],
        options: GenerationOptions,
        *,
        request_id: str | None = None,
    ) -> RequestHandle:
        if not self._running:
            raise RuntimeError("BatchingServer is not running; call start() first")
        handle = RequestHandle(request_id or self._next_id("req"))
        self._pending.put(_Submission(list(prompt_ids), options, handle))
        self._wake.set()
        return handle

    def run_control(self, fn: Callable[[Engine], object], *, timeout: float = 30.0) -> object:
        """Run ``fn(engine)`` on the loop thread (exclusive model access).

        Used for weight sync and any other operation that must not race a model
        forward. Returns the callable's result, or re-raises its exception.
        """
        control = _Control(fn=fn, done=threading.Event(), box={})
        self._control.put(control)
        self._wake.set()
        if not control.done.wait(timeout=timeout):
            raise TimeoutError("control action did not run within timeout")
        if "error" in control.box:
            error = control.box["error"]
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(f"control action failed: {error!r}")
        return control.box.get("result")

    # ---- loop-thread internals -------------------------------------------
    def _drain_control(self) -> None:
        while True:
            try:
                control = self._control.get_nowait()
            except queue.Empty:
                return
            try:
                control.box["result"] = control.fn(self._engine)
            except Exception as exc:  # surfaced to the calling thread
                control.box["error"] = exc
            finally:
                control.done.set()

    def _drain_pending(self) -> bool:
        drained = False
        while True:
            try:
                sub = self._pending.get_nowait()
            except queue.Empty:
                return drained
            drained = True
            if sub.options.mode == "speculative":
                # spec_decode is a self-contained verify loop, not a stepwise
                # decoder — run it inline (not through the scheduler tick).
                try:
                    state = self._engine.generate_with_state(
                        sub.prompt_ids, sub.options, request_id=sub.handle.request_id
                    )
                except (ValueError, TypeError) as exc:
                    sub.handle._fail(exc)
                    continue
                sub.handle._bind(state)
                sub.handle._finish()
                continue
            try:
                state = self._scheduler.submit(sub.handle.request_id, sub.prompt_ids, sub.options)
            except (ValueError, TypeError) as exc:
                sub.handle._fail(exc)
                continue
            sub.handle._bind(state)
            if sub.handle._cancelled:
                state.cancel()  # cancelled before we bound it
            self._tracked.append(sub.handle)

    def _loop(self) -> None:
        # Only this thread runs model ops (MLX isn't thread-safe), so the default
        # stream is fine; MLX/Metal was already initialized by start()'s warmup.
        while self._running:
            self._drain_control()
            drained = self._drain_pending()
            if self._scheduler.has_work():
                self._scheduler.tick()
                still: list[RequestHandle] = []
                for handle in self._tracked:
                    handle._pump()
                    if handle.state is not None and handle.state.finished:
                        handle._finish()
                    else:
                        still.append(handle)
                self._tracked = still
            elif not drained and self._control.empty():
                self._wake.wait(timeout=self._idle_timeout)
                self._wake.clear()
        # Shutdown: fail anything still in flight so waiters don't hang.
        for handle in self._tracked:
            handle._fail(RuntimeError("server stopped before request completed"))
        self._tracked = []
