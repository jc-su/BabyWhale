"""Green-gate for the course infrastructure: presets, journey, ablations, autograder.

The digit-prefixed module folders aren't importable, so their thin wrapper scripts
aren't tested here; the *logic* they call (in ``course/*.py``) is.
"""

from __future__ import annotations

import unittest

import mlx.core as mx
from course.ablations import (
    attention_cost_rows,
    kv_decode_rows,
    mla_kv_cache_rows,
    moe_params_rows,
    quant_memory_rows,
    spec_tokens_rows,
)
from course.journey import run_journey
from course.labs import (
    attention_reference,
    cross_entropy_reference,
    dpo_loss_reference,
    grade_attention,
    grade_cross_entropy,
    grade_dpo,
    grade_group_advantages,
    grade_kv_append,
    grade_mla_roundtrip,
    grade_moe_route,
    grade_mtp_head,
    grade_rms_norm,
    grade_rope,
    grade_spec_accept,
    grade_swiglu,
    grade_transformer_layer,
    group_advantages_reference,
    kv_append_reference,
    mla_roundtrip_reference,
    moe_route_reference,
    mtp_head_reference,
    rms_norm_reference,
    rope_reference,
    spec_accept_reference,
    swiglu_reference,
    transformer_layer_reference,
)
from course.milestones import FAST_MILESTONES
from course.presets import LADDER, load_preset
from course.systems import summarize

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model


class TestPresets(unittest.TestCase):
    def test_every_preset_builds_a_model(self) -> None:
        for name in LADDER:
            cfg = load_preset(name)
            self.assertIsInstance(cfg, BabyWhaleV4Config)
            BabyWhaleV4Model(cfg)  # construction validates schedule + fields
            self.assertEqual(len(cfg.effective_layer_schedule), cfg.n_layer)

    def test_minimal_and_full_forward(self) -> None:
        for name in ("gpt-minimal", "full"):
            cfg = load_preset(name)
            model = BabyWhaleV4Model(cfg)
            model.eval()
            out = model(mx.array([[1, 2, 3, 4]]))
            self.assertEqual(out.logits.shape[-1], cfg.vocab_size)

    def test_ladder_adds_one_thing_at_a_time(self) -> None:
        minimal = load_preset("gpt-minimal")
        self.assertNotIn("mla", minimal.effective_layer_schedule)
        self.assertIn("mla", load_preset("plus-mla").effective_layer_schedule)
        self.assertEqual(load_preset("gpt-minimal").mtp_heads, 0)
        self.assertEqual(load_preset("plus-mtp").mtp_heads, 2)

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            load_preset("nope")


class TestJourney(unittest.TestCase):
    def test_journey_runs_and_learns(self) -> None:
        result = run_journey(steps=20, seed=0)
        self.assertIsInstance(result["sample"], str)
        self.assertLess(result["loss_last"], result["loss_first"])


class TestAblation(unittest.TestCase):
    def test_mla_is_mqa_sized_and_smaller_than_mha(self) -> None:
        by = {row.label.split()[0]: row.bytes_per_token for row in mla_kv_cache_rows()}
        self.assertLess(by["MLA"], by["MHA"])
        self.assertEqual(by["MLA"], by["MQA"])


class TestLabAutograder(unittest.TestCase):
    def test_reference_passes(self) -> None:
        grade_mla_roundtrip(mla_roundtrip_reference)

    def test_wrong_implementation_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            grade_mla_roundtrip(lambda kv, w_down, w_up: (kv, kv))


class TestAlignmentLabs(unittest.TestCase):
    """Modeling/alignment labs (RoPE, DPO, GRPO) — the course is not only systems."""

    def test_rope_reference_passes(self) -> None:
        grade_rope(rope_reference)

    def test_dpo_reference_passes(self) -> None:
        grade_dpo(dpo_loss_reference)

    def test_grpo_reference_passes(self) -> None:
        grade_group_advantages(group_advantages_reference)

    def test_a_wrong_rope_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            grade_rope(lambda x, cos, sin: x)  # no rotation at all


class TestFoundationLabs(unittest.TestCase):
    """theory→code labs across the stack — each reference passes its own grader."""

    def test_rms_norm(self) -> None:
        grade_rms_norm(rms_norm_reference)

    def test_attention(self) -> None:
        grade_attention(attention_reference)

    def test_moe_route(self) -> None:
        grade_moe_route(moe_route_reference)

    def test_spec_accept(self) -> None:
        grade_spec_accept(spec_accept_reference)

    def test_cross_entropy(self) -> None:
        grade_cross_entropy(cross_entropy_reference)

    def test_kv_append(self) -> None:
        grade_kv_append(kv_append_reference)


class TestBuildLabs(unittest.TestCase):
    """Build-the-real-component labs — each graded against the actual module."""

    def test_swiglu(self) -> None:
        grade_swiglu(swiglu_reference)

    def test_mtp_head(self) -> None:
        grade_mtp_head(mtp_head_reference)

    def test_transformer_layer(self) -> None:
        grade_transformer_layer(transformer_layer_reference)


class TestRunnableAblations(unittest.TestCase):
    """Every 'measure' beat prints a real number — check the numbers are sane."""

    def test_attention_cost_grows_quadratically(self) -> None:
        rows = attention_cost_rows()
        self.assertLess(rows[0][3], rows[-1][3])  # full/windowed ratio grows with n
        self.assertGreaterEqual(rows[-1][3], 8)

    def test_moe_decouples_capacity_from_flops(self) -> None:
        dense, moe = moe_params_rows(n_expert=8, k=2)
        self.assertEqual(moe.params, 8 * dense.params)  # 8x params
        self.assertEqual(moe.active_units, 2 * dense.active_units)  # 2x FLOPs

    def test_kv_cache_saving_grows(self) -> None:
        rows = kv_decode_rows()
        self.assertLess(rows[0][3], rows[-1][3])

    def test_spec_tokens_increase_with_acceptance(self) -> None:
        rows = spec_tokens_rows(k=4)
        self.assertTrue(all(expected >= 1.0 for _, expected in rows))
        self.assertLess(rows[0][1], rows[-1][1])

    def test_quant_is_4x(self) -> None:
        for _params, _bf16, _fp4, ratio in quant_memory_rows():
            self.assertAlmostEqual(ratio, 4.0, places=1)


class TestSystemsLens(unittest.TestCase):
    def test_summary_numbers_make_sense(self) -> None:
        s = summarize(load_preset("full"))
        self.assertGreater(s.params, 0)
        self.assertLess(s.weight_mb_fp4, s.weight_mb_bf16)  # fp4 is smaller
        self.assertLess(s.attn_flops_windowed, s.attn_flops_full)  # windows save FLOPs
        self.assertLess(s.active_experts, s.total_experts)  # MoE is sparse


class TestMilestones(unittest.TestCase):
    def test_fast_milestones_pass(self) -> None:
        for milestone in FAST_MILESTONES:
            result = milestone()
            self.assertTrue(result.passed, f"{result.name}: {result.evidence}")


if __name__ == "__main__":
    unittest.main()
