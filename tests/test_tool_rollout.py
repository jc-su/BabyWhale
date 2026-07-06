import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.rl import InProcessRolloutEngine, RolloutRequest
from baby_whale_v4.tools.local import build_default_registry
from baby_whale_v4.tools.schema import (
    ToolCall,
    parse_tool_call_text,
    render_tool_call,
    render_tool_result,
)


class TestToolDrivenRollout(unittest.TestCase):
    def test_parse_render_round_trip(self):
        call = ToolCall(name="calculator.add", arguments={"a": 2, "b": 3})
        rendered = render_tool_call(call)
        parsed = parse_tool_call_text(rendered)
        self.assertEqual(parsed.name, call.name)
        self.assertEqual(parsed.arguments, call.arguments)

    def test_generate_with_tools_returns_sample_when_no_tool_call(self):
        # The randomly-initialized tiny model is extremely unlikely to emit a
        # well-formed <tool_call>...</tool_call> block. The rollout should
        # finish without invoking any tool and still produce a valid sample.
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=259, context_length=64)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        tok = ByteTokenizer()
        engine = InProcessRolloutEngine(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
        )
        registry = build_default_registry()
        request = RolloutRequest(
            prompt_ids=tuple(tok.encode("hi")),
            options=GenerationOptions(max_new_tokens=12, mode="sample"),
        )
        sample = engine.generate_with_tools(request, tokenizer=tok, registry=registry, max_turns=2)
        self.assertGreater(len(sample.response_ids), 0)
        self.assertEqual(len(sample.tool_calls), 0)
        # log_probs cover all model-emitted positions (no tool injection
        # happened).
        self.assertEqual(len(sample.log_probs), len(sample.response_ids))

    def test_render_tool_result_wraps_payload(self):
        from baby_whale_v4.tools.schema import ToolResult

        result = ToolResult(name="calculator.add", ok=True, content=5)
        text = render_tool_result(result)
        self.assertTrue(text.startswith("<tool_result>"))
        self.assertTrue(text.endswith("</tool_result>"))
        self.assertIn('"ok":true', text)
        self.assertIn('"content":5', text)


if __name__ == "__main__":
    unittest.main()
