"""Reward computation as a typed boundary.

A reward host turns a `RolloutSample` into a scalar. Local hosts wrap a Python
callable; HTTP hosts forward the sample to a separate verifier service.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Protocol

from baby_whale_v4.rl.types import RolloutSample


class RewardHost(Protocol):
    def score(self, sample: RolloutSample) -> float: ...

    def score_batch(self, samples: Sequence[RolloutSample]) -> list[float]: ...


class LocalRewardHost:
    """Reward host that runs a synchronous Python callable in-process."""

    def __init__(self, fn: Callable[[RolloutSample], float]) -> None:
        if not callable(fn):
            raise TypeError("LocalRewardHost requires a callable reward function")
        self._fn = fn

    def score(self, sample: RolloutSample) -> float:
        if not isinstance(sample, RolloutSample):
            raise TypeError("LocalRewardHost.score expects a RolloutSample")
        value = self._fn(sample)
        if not isinstance(value, float):
            raise TypeError(
                f"reward function must return a Python float; got {type(value).__name__}"
            )
        return value

    def score_batch(self, samples: Sequence[RolloutSample]) -> list[float]:
        return [self.score(sample) for sample in samples]


class HTTPRewardHost:
    """Reward host that POSTs samples to a verifier HTTP endpoint.

    Request body shape:
        {"response_ids": [int, ...], "metadata": {str: str, ...}}
    Expected response:
        {"reward": float}
    """

    def __init__(self, url: str, *, timeout_sec: float = 10.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("HTTPRewardHost.url must be http:// or https://")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.url = url
        self.timeout_sec = timeout_sec

    def score(self, sample: RolloutSample) -> float:
        body = json.dumps(
            {
                "response_ids": list(sample.response_ids),
                "metadata": dict(sample.request.metadata),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTPRewardHost failed: {exc}") from exc
        reward = payload.get("reward")
        if not isinstance(reward, (int, float)):
            raise TypeError(f"HTTPRewardHost expected float reward, got {payload!r}")
        return float(reward)

    def score_batch(self, samples: Sequence[RolloutSample]) -> list[float]:
        return [self.score(sample) for sample in samples]
