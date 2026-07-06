"""Tests for MLX runtime validation."""

from __future__ import annotations

import unittest
from typing import cast, get_args
from unittest.mock import patch

import mlx.core as mx

from baby_whale_v4.device import (
    active_runtime,
    available_runtimes,
    ensure_runtime_matches,
    is_cuda_runtime,
    is_metal_runtime,
)
from baby_whale_v4.typing import Backend, MLXRuntime


class _RuntimeProbe:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class TestBackendLiteral(unittest.TestCase):
    def test_backend_literal_is_mlx_only(self) -> None:
        self.assertEqual(get_args(Backend), ("mlx",))

    def test_runtime_literal_keeps_metal_and_cuda(self) -> None:
        self.assertEqual(get_args(MLXRuntime), ("mlx-metal", "mlx-cuda"))


class TestRuntimeDetection(unittest.TestCase):
    def test_active_runtime_is_available(self) -> None:
        runtime = active_runtime()
        self.assertIn(runtime, available_runtimes())
        self.assertEqual(ensure_runtime_matches("mlx"), runtime)

    def test_current_explicit_runtime_passes(self) -> None:
        runtime = active_runtime()
        self.assertEqual(ensure_runtime_matches("mlx", runtime), runtime)

    def test_ensure_rejects_non_mlx_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported backend"):
            ensure_runtime_matches(cast(Backend, "cuda"), active_runtime())

    def test_metal_only_runtime(self) -> None:
        with (
            patch.object(mx, "metal", _RuntimeProbe(True), create=True),
            patch.object(mx, "cuda", _RuntimeProbe(False), create=True),
        ):
            self.assertTrue(is_metal_runtime())
            self.assertFalse(is_cuda_runtime())
            self.assertEqual(available_runtimes(), ("mlx-metal",))
            self.assertEqual(active_runtime(), "mlx-metal")
            self.assertEqual(ensure_runtime_matches("mlx", "mlx-metal"), "mlx-metal")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                ensure_runtime_matches("mlx", "mlx-cuda")

    def test_cuda_only_runtime(self) -> None:
        with (
            patch.object(mx, "metal", _RuntimeProbe(False), create=True),
            patch.object(mx, "cuda", _RuntimeProbe(True), create=True),
        ):
            self.assertFalse(is_metal_runtime())
            self.assertTrue(is_cuda_runtime())
            self.assertEqual(available_runtimes(), ("mlx-cuda",))
            self.assertEqual(active_runtime(), "mlx-cuda")
            self.assertEqual(ensure_runtime_matches("mlx", "mlx-cuda"), "mlx-cuda")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                ensure_runtime_matches("mlx", "mlx-metal")

    def test_missing_runtime_fails_fast(self) -> None:
        with (
            patch.object(mx, "metal", _RuntimeProbe(False), create=True),
            patch.object(mx, "cuda", _RuntimeProbe(False), create=True),
        ):
            self.assertEqual(available_runtimes(), ())
            with self.assertRaisesRegex(RuntimeError, "MLX Metal or MLX CUDA"):
                active_runtime()
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                ensure_runtime_matches("mlx", "mlx-cuda")

    def test_multiple_runtimes_require_explicit_selection(self) -> None:
        with (
            patch.object(mx, "metal", _RuntimeProbe(True), create=True),
            patch.object(mx, "cuda", _RuntimeProbe(True), create=True),
        ):
            self.assertEqual(available_runtimes(), ("mlx-metal", "mlx-cuda"))
            with self.assertRaisesRegex(RuntimeError, "multiple MLX runtimes"):
                active_runtime()
            self.assertEqual(ensure_runtime_matches("mlx", "mlx-metal"), "mlx-metal")
            self.assertEqual(ensure_runtime_matches("mlx", "mlx-cuda"), "mlx-cuda")

    def test_rejects_unknown_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported runtime"):
            ensure_runtime_matches("mlx", cast(MLXRuntime, "cuda"))


if __name__ == "__main__":
    unittest.main()
