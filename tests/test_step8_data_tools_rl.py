import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from baby_whale_v4 import BabyWhaleV4Config
from baby_whale_v4.data.hf_prepare import HFSource, normalize_hf_row, write_normalized_jsonl
from baby_whale_v4.kernels.fp4_training import benchmark_metal_weight_grad
from baby_whale_v4.rl import (
    ToolRolloutConfig,
    ToolRolloutRunner,
    ToolUseTask,
    make_arithmetic_tool_tasks,
    score_tool_response,
)
from baby_whale_v4.tools import (
    ToolCall,
    build_default_registry,
    parse_tool_call_text,
    render_tool_call,
)
from baby_whale_v4.training.fp4_native import (
    benchmark_custom_vjp_fp4_training,
    benchmark_fp4_training_memory,
    probe_custom_vjp_fp4_training,
    probe_metal_vjp_fp4_training,
    probe_native_fp4_training,
)
from baby_whale_v4.training.precision import ensure_training_precision_supported


class FakePolicy:
    def __init__(self, responses: list[str]):
        self.responses = responses

    def sample(self, prompt: str, *, group_size: int, max_new_tokens: int) -> list[str]:
        if not prompt:
            raise ValueError("prompt must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        return self.responses[:group_size]


class TestStep8DataToolsRL(unittest.TestCase):
    def test_normalize_pretrain_row(self):
        src = HFSource(dataset_id="demo/pretrain", split="train", kind="pretrain", limit=2)
        row = normalize_hf_row({"text": "hello world"}, source=src, source_index=3)
        self.assertEqual(row["kind"], "pretrain")
        self.assertEqual(row["text"], "hello world")
        row_source = row["source"]
        self.assertIsInstance(row_source, dict)
        if not isinstance(row_source, dict):
            raise AssertionError("expected normalized source object")
        source = cast(dict[str, object], row_source)
        self.assertEqual(source["dataset_id"], "demo/pretrain")
        self.assertEqual(source["index"], 3)

    def test_normalize_tool_trace_row(self):
        src = HFSource(dataset_id="demo/tools", split="train", kind="tool_trace", limit=2)
        row = normalize_hf_row(
            {
                "messages": [
                    {"role": "user", "content": "multiply"},
                    {"role": "assistant", "content": "tool"},
                ],
                "tools": [{"name": "calculator.multiply"}],
            },
            source=src,
            source_index=0,
        )
        self.assertEqual(row["kind"], "tool_trace")
        messages = row["messages"]
        self.assertIsInstance(messages, list)
        if not isinstance(messages, list):
            raise AssertionError("expected normalized messages list")
        self.assertEqual(len(messages), 2)

    def test_write_normalized_jsonl(self):
        src = HFSource(dataset_id="demo/pretrain", split="train", kind="pretrain", limit=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            n = write_normalized_jsonl(
                [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                source=src,
                path=path,
            )
            self.assertEqual(n, 2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["text"] for row in rows], ["a", "b"])

    def test_tool_call_roundtrip_and_execution(self):
        call = ToolCall("calculator.multiply", {"a": 19, "b": 23})
        text = render_tool_call(call)
        parsed = parse_tool_call_text(text)
        self.assertEqual(parsed, call)
        result = build_default_registry().execute(parsed)
        self.assertTrue(result.ok)
        self.assertEqual(result.content, 437.0)

    def test_tool_registry_rejects_bad_args(self):
        result = build_default_registry().execute(ToolCall("calculator.multiply", {"a": 1}))
        self.assertFalse(result.ok)
        error = result.error
        self.assertIsNotNone(error)
        if error is None:
            raise AssertionError("expected tool error")
        self.assertIn("missing required arguments", error)

    def test_tool_reward_scores_valid_response(self):
        registry = build_default_registry()
        call = ToolCall("calculator.multiply", {"a": 19, "b": 23})
        task = ToolUseTask(prompt="What is 19 * 23?", expected_call=call, expected_answer="437")
        reward = score_tool_response(render_tool_call(call) + " 437", task, registry)
        self.assertEqual(reward.total, 5.0)
        self.assertTrue(reward.valid_json)
        self.assertTrue(reward.args_match)
        self.assertTrue(reward.answer_match)

    def test_tool_reward_penalizes_invalid_json(self):
        registry = build_default_registry()
        call = ToolCall("calculator.multiply", {"a": 19, "b": 23})
        task = ToolUseTask(prompt="What is 19 * 23?", expected_call=call)
        reward = score_tool_response("<tool_call>{bad</tool_call>", task, registry)
        self.assertEqual(reward.total, -1.0)
        self.assertFalse(reward.valid_json)

    def test_mac_rl_runner_collects_grouped_rollouts(self):
        registry = build_default_registry()
        call = ToolCall("calculator.multiply", {"a": 2, "b": 4})
        good = render_tool_call(call) + " 8"
        bad = "<tool_call>{bad</tool_call>"
        runner = ToolRolloutRunner(
            policy=FakePolicy([good, bad]),
            registry=registry,
            config=ToolRolloutConfig(group_size=2, max_new_tokens=32),
        )
        records = runner.collect(
            [ToolUseTask(prompt="What is 2 * 4?", expected_call=call, expected_answer="8")]
        )
        self.assertEqual(len(records), 2)
        self.assertGreater(records[0].reward.total, records[1].reward.total)
        self.assertGreater(ToolRolloutRunner.mean_reward(records), 0.0)

    def test_arithmetic_tool_environment_builds_verifiable_tasks(self):
        tasks = make_arithmetic_tool_tasks(limit=4)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(tasks[0].expected_call.name, "calculator.add")
        self.assertEqual(tasks[2].expected_call.name, "calculator.multiply")
        registry = build_default_registry()
        call = tasks[2].expected_call
        reward = score_tool_response(
            render_tool_call(call) + f" {tasks[2].expected_answer}", tasks[2], registry
        )
        self.assertEqual(reward.total, 5.0)

    def test_fp4_native_training_fails_fast(self):
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        cfg = BabyWhaleV4Config.from_dict({**base.to_dict(), "quant_mode": "fp4-native"})
        status = probe_native_fp4_training()
        self.assertFalse(status.supported)
        self.assertIn("QuantizedMatmul::vjp", status.reason)
        with self.assertRaisesRegex(RuntimeError, "native MLX FP4 training is not supported"):
            ensure_training_precision_supported(cfg)

    def test_fp4_expert_training_fails_fast(self):
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        cfg = BabyWhaleV4Config.from_dict({**base.to_dict(), "quant_mode": "fp4-expert"})
        with self.assertRaisesRegex(RuntimeError, "fp4-expert is inference/export only"):
            ensure_training_precision_supported(cfg)

    def test_fp4_native_train_custom_vjp_probe_is_supported(self):
        status = probe_custom_vjp_fp4_training()
        self.assertTrue(status.supported, status.reason)

    def test_fp4_native_train_benchmark_reports_ratio(self):
        bench = benchmark_custom_vjp_fp4_training()
        self.assertGreater(bench.dense_ms, 0.0)
        self.assertGreater(bench.fp4_ms, 0.0)
        self.assertGreater(bench.ratio, 0.0)

    def test_fp4_training_memory_benchmark_reports_peak_ratio(self):
        bench = benchmark_fp4_training_memory(
            batch=8,
            input_dims=128,
            output_dims=128,
            baseline="bf16",
            cache_policy="recompute",
            optimizer="adafactor",
            fp4_master_dtype="bf16",
        )
        self.assertGreater(bench.dense_fp32_peak_bytes, 0)
        self.assertGreater(bench.dense_fp32_active_bytes, 0)
        self.assertGreater(bench.dense_bf16_peak_bytes, 0)
        self.assertGreater(bench.dense_bf16_active_bytes, 0)
        self.assertGreater(bench.fp4_peak_bytes, 0)
        self.assertGreater(bench.peak_ratio, 0.0)
        self.assertGreater(bench.active_ratio, 0.0)
        self.assertEqual(bench.cache_policy, "recompute")
        self.assertEqual(bench.optimizer, "adafactor")
        self.assertEqual(bench.fp4_master_dtype, "bf16")

    def test_fp4_native_train_metal_probe_is_supported(self):
        status = probe_metal_vjp_fp4_training()
        self.assertTrue(status.supported, status.reason)

    def test_fp4_metal_weight_grad_benchmark_reports_ratio(self):
        bench = benchmark_metal_weight_grad(timed_steps=2)
        self.assertGreater(bench.dense_ms, 0.0)
        self.assertGreater(bench.metal_ms, 0.0)
        self.assertGreater(bench.ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
