"""Tests for the per-stage evaluation subcommands and shared RL telemetry."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.cli import eval as eval_cli
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.training.checkpoint import save_checkpoint
from baby_whale_v4.training.rl_telemetry import policy_telemetry


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _run_capturing_json(
    fn: Callable[[argparse.Namespace], None], args: argparse.Namespace
) -> dict[str, Any]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(args)
    parsed = json.loads(buf.getvalue())
    if not isinstance(parsed, dict):
        raise TypeError("captured JSON output must be an object")
    return cast(dict[str, Any], parsed)


def _write_tiny_ckpt(tmp: Path, ctx: int = 64) -> Path:
    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=ctx)
    model = BabyWhaleV4Model(cfg)
    out = tmp / "tiny.bw4"
    save_checkpoint(out, config=cfg, model=model, optimizer=None, scheduler=None, step=0)
    return out


class TestEvalCli(unittest.TestCase):
    def test_eval_tokenizer_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jsonl = tmp / "data.jsonl"
            jsonl.write_text(
                "\n".join(
                    json.dumps({"text": t})
                    for t in ["hello world", "the quick brown fox", "abc def ghi jkl"]
                )
                + "\n",
                encoding="utf-8",
            )
            out = _run_capturing_json(
                eval_cli._eval_tokenizer,
                _ns(
                    tokenizer_path=None,
                    input_jsonl=str(jsonl),
                    text_field="text",
                    limit=None,
                ),
            )
            self.assertEqual(out["n_lines"], 3)
            self.assertGreater(out["bytes_per_token"], 0)
            self.assertGreater(out["fertility_tokens_per_word"], 0)

    def test_eval_bpb_returns_finite_numbers(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _write_tiny_ckpt(tmp, ctx=32)
            jsonl = tmp / "valid.jsonl"
            jsonl.write_text(
                json.dumps({"text": "a small block of text to score"}) + "\n",
                encoding="utf-8",
            )
            out = _run_capturing_json(
                eval_cli._eval_bpb,
                _ns(
                    from_checkpoint=str(ckpt),
                    tokenizer_path=None,
                    input_jsonl=str(jsonl),
                    text_field="text",
                    block_size=8,
                    limit_blocks=None,
                ),
            )
            self.assertGreater(out["n_blocks"], 0)
            self.assertGreater(out["mean_loss_nats"], 0)
            self.assertGreater(out["bits_per_byte"], 0)
            self.assertGreater(out["perplexity"], 1.0)

    def test_eval_code_pass_at_1_on_trivial_problem(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _write_tiny_ckpt(tmp, ctx=64)
            problems_path = tmp / "problems.jsonl"
            problems_path.write_text(
                json.dumps(
                    {
                        "problem_id": "trivial-0",
                        "prompt": "def add(a, b):",
                        "tests": ["assert True"],
                        "canonical_solution": "def add(a,b):\n    return a+b",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = _run_capturing_json(
                eval_cli._eval_code,
                _ns(
                    from_checkpoint=str(ckpt),
                    tokenizer_path=None,
                    problems_jsonl=str(problems_path),
                    max_new_tokens=8,
                    chat_template=False,
                    timeout_sec=2.0,
                    limit=None,
                    show_details=True,
                ),
            )
            self.assertEqual(out["n_total"], 1)
            self.assertIn("pass_at_1", out)
            self.assertGreaterEqual(out["pass_at_1"], 0.0)
            self.assertLessEqual(out["pass_at_1"], 1.0)

    def test_eval_ifeval_runs_and_returns_accuracy(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _write_tiny_ckpt(tmp, ctx=128)
            out = _run_capturing_json(
                eval_cli._eval_ifeval,
                _ns(
                    from_checkpoint=str(ckpt),
                    tokenizer_path=None,
                    max_new_tokens=16,
                    show_details=False,
                ),
            )
            self.assertEqual(out["n_total"], 8)
            self.assertGreaterEqual(out["strict_accuracy"], 0.0)
            self.assertLessEqual(out["strict_accuracy"], 1.0)

    def test_eval_dpo_accuracy_in_range(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _write_tiny_ckpt(tmp, ctx=64)
            jsonl = tmp / "pref.jsonl"
            jsonl.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "kind": "preference",
                            "prompt": p,
                            "chosen": c,
                            "rejected": r,
                        }
                    )
                    for p, c, r in [
                        ("Hello", "world good", "world bad"),
                        ("Pick one", "yes", "no"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = _run_capturing_json(
                eval_cli._eval_dpo,
                _ns(
                    from_checkpoint=str(ckpt),
                    ref_checkpoint=None,
                    tokenizer_path=None,
                    input_jsonl=str(jsonl),
                    beta=0.1,
                    max_prompt_tokens=8,
                    max_response_tokens=8,
                    limit=None,
                ),
            )
            self.assertEqual(out["n_total"], 2)
            # Reference == model copy, so reward_accuracy must be 0 (margins are all 0).
            self.assertEqual(out["reward_accuracy"], 0.0)

    def test_eval_rl_health_detects_violations(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mpath = tmp / "metrics.jsonl"
            rows = [
                {
                    "step": 1,
                    "reward_mean": 0.1,
                    "reward_std": 0.01,
                    "kl_mean": 1.0,
                    "entropy_mean": 4.0,
                },
                {
                    "step": 2,
                    "reward_mean": 0.1,
                    "reward_std": 0.01,
                    "kl_mean": 1.5,
                    "entropy_mean": 3.5,
                },
                {
                    "step": 3,
                    "reward_mean": 0.1,
                    "reward_std": 0.005,
                    "kl_mean": 50.0,
                    "entropy_mean": 0.01,
                },
                {
                    "step": 4,
                    "reward_mean": 0.05,
                    "reward_std": 0.0,
                    "kl_mean": 60.0,
                    "entropy_mean": 0.0,
                },
            ]
            mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out = _run_capturing_json(
                eval_cli._eval_rl_health,
                _ns(
                    metrics_jsonl=str(mpath),
                    max_kl=20.0,
                    min_entropy=0.1,
                    min_reward_std=1e-4,
                    min_reward_delta=0.0,
                ),
            )
            self.assertFalse(out["healthy"])
            warnings = " ".join(out["warnings"])
            self.assertIn("kl ceiling", warnings)
            self.assertIn("entropy collapsed", warnings)
            self.assertIn("reward_std collapse", warnings)
            self.assertIn("reward stagnation", warnings)

    def test_eval_rl_health_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mpath = tmp / "metrics.jsonl"
            rows = [
                {
                    "step": 1,
                    "reward_mean": 0.1,
                    "reward_std": 0.5,
                    "kl_mean": 1.0,
                    "entropy_mean": 4.0,
                },
                {
                    "step": 2,
                    "reward_mean": 0.2,
                    "reward_std": 0.5,
                    "kl_mean": 2.0,
                    "entropy_mean": 3.8,
                },
                {
                    "step": 3,
                    "reward_mean": 0.4,
                    "reward_std": 0.5,
                    "kl_mean": 3.0,
                    "entropy_mean": 3.5,
                },
                {
                    "step": 4,
                    "reward_mean": 0.6,
                    "reward_std": 0.5,
                    "kl_mean": 4.0,
                    "entropy_mean": 3.2,
                },
            ]
            mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out = _run_capturing_json(
                eval_cli._eval_rl_health,
                _ns(
                    metrics_jsonl=str(mpath),
                    max_kl=20.0,
                    min_entropy=0.1,
                    min_reward_std=1e-4,
                    min_reward_delta=0.0,
                ),
            )
            self.assertTrue(out["healthy"])
            self.assertEqual(out["warnings"], [])

    def test_eval_parity_matches(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _write_tiny_ckpt(tmp, ctx=32)
            out = _run_capturing_json(
                eval_cli._eval_parity,
                _ns(
                    from_checkpoint=str(ckpt),
                    tokenizer_path=None,
                    prompt="hi",
                    max_new_tokens=4,
                    with_prefix_cache=False,
                    seed=0,
                ),
            )
            self.assertTrue(out["parity_ok"])
            self.assertEqual(out["n_match"], out["n_total"])


class TestMetricsEcho(unittest.TestCase):
    def test_format_line_orders_step_first_then_loss(self) -> None:
        from baby_whale_v4.training.metrics import _format_line

        line = _format_line({"reward_mean": 0.5, "step": 10, "grpo_loss": 0.123, "kl_mean": 1.0})
        # step first, then grpo_loss, then alphabetical.
        self.assertTrue(line.startswith("step=10 "))
        self.assertIn("step=10 grpo_loss=0.1230", line)
        self.assertLess(line.index("grpo_loss"), line.index("kl_mean"))
        self.assertLess(line.index("kl_mean"), line.index("reward_mean"))

    def test_jsonl_metrics_echoes_to_stderr(self) -> None:
        from baby_whale_v4.training.metrics import JsonlMetrics

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "grpo_metrics.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf), JsonlMetrics(path) as m:
                m.log({"step": 1, "grpo_loss": 0.5, "reward_mean": 0.1})
            out = buf.getvalue()
            self.assertIn("[grpo] step=1", out)
            self.assertIn("grpo_loss=0.5000", out)
            self.assertIn("reward_mean=0.1000", out)

    def test_jsonl_metrics_echo_off(self) -> None:
        from baby_whale_v4.training.metrics import JsonlMetrics

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf), JsonlMetrics(path, echo=False) as m:
                m.log({"step": 1, "x": 0.5})
            self.assertEqual(buf.getvalue(), "")


class TestWatchMetrics(unittest.TestCase):
    def test_watch_metrics_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mpath = tmp / "metrics.jsonl"
            rows = [
                {
                    "step": 1,
                    "reward_mean": 0.1,
                    "kl_mean": 1.0,
                    "entropy_mean": 3.0,
                    "grpo_loss": 0.5,
                },
                {
                    "step": 2,
                    "reward_mean": 0.3,
                    "kl_mean": 1.5,
                    "entropy_mean": 2.8,
                    "grpo_loss": 0.4,
                },
                {
                    "step": 3,
                    "reward_mean": 0.5,
                    "kl_mean": 2.0,
                    "entropy_mean": 2.5,
                    "grpo_loss": 0.3,
                },
            ]
            mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out = _run_capturing_json(
                eval_cli._watch_metrics,
                _ns(metrics_jsonl=str(mpath), watch=False, interval=1.0),
            )
            self.assertEqual(out["n_steps"], 3)
            self.assertEqual(out["latest"]["step"], 3)
            self.assertEqual(out["reward_mean"]["first"], 0.1)
            self.assertEqual(out["reward_mean"]["last"], 0.5)
            self.assertEqual(out["kl_mean"]["max"], 2.0)
            self.assertIn("grpo_loss", out["losses"])
            self.assertEqual(out["losses"]["grpo_loss"]["last"], 0.3)


class TestRLTelemetry(unittest.TestCase):
    def test_policy_telemetry_returns_finite(self) -> None:
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=32)
        model = BabyWhaleV4Model(cfg)
        ref = BabyWhaleV4Model(cfg)
        ref.update(model.state_dict())  # identical, so KL ≈ 0
        prompt = mx.array([1, 2, 3], dtype=mx.int32)
        samples = mx.array([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=mx.int32)
        tele = policy_telemetry(model=model, ref=ref, prompt=prompt, samples=samples)
        self.assertAlmostEqual(tele["kl_mean"], 0.0, places=4)
        self.assertGreater(tele["entropy_mean"], 0.0)
        self.assertEqual(tele["response_len_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
