"""Drift guard for the code shown in each module's "In the code" beat.

Every identifier the course *displays* must exist in the real source (so a rename
can't silently make the docs lie) and must actually appear in the module README
(so the map stays in sync with what's shown). This is what lets the course paste
real code without it drifting from the implementation.
"""

from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# module slug -> ((source file under baby_whale_v4/, identifiers shown), ...)
SHOWN: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "01-backbone": (("model.py", ("hc.consume", "self.ln_1", "self.attn", "hc.produce")),),
    "02-attention-basics": (
        (
            "attention.py",
            (
                "q_proj",
                "k_proj",
                "self.rope",
                "swapaxes(-2, -1)",
                "_masked_softmax",
                "sliding_window",
            ),
        ),
    ),
    "03-attention-mla": (("attention.py", ("kv_a_proj", "append_latent", "kv_b_proj")),),
    "04-attention-compressed": (
        ("attention.py", ("_block_mean_pool", "comp_allowed", "raw_allowed", "sliding_window")),
    ),
    "05-moe": (("moe.py", ("nn.softplus", "_bias_array", "mx.argsort", "take_along_axis")),),
    "06-hyperconnect": (
        ("mhc.py", ("input_logits", 'einsum("btkd,k->btd"', "sinkhorn", "write_logits")),
    ),
    "07-mtp": (("mtp.py", ("self.head", "nn.silu", "self.transform")),),
    "08-tokenizer-and-data": (
        ("data/tokenizer.py", ("heapq.heappop", "_BPE_BASE_VOCAB", "alive[j] = False")),
    ),
    "09-pretraining": (
        (
            "training/pretrain.py",
            ("_accumulated_loss_and_grads", "clip_grad_norm", "_target_token_count"),
        ),
        ("training/mlx_optim.py", ("m_hat", "v_hat", "weight_decay")),
    ),
    "10-midtraining": (
        ("training/midtrain.py", ("def midtrain", "PretrainConfig", "warmup_steps")),
    ),
    "11-sft": (("data/chat.py", ("format_chat", "mask_y", "mx.where")),),
    "12-dpo": (
        (
            "training/dpo.py",
            ("_logp_response", "_log_sigmoid", "ref_logratio", "_precompute_ref_logratios"),
        ),
    ),
    "13-rl-grpo": (
        ("training/grpo.py", ("_std(rewards)", "_kl_per_token", "advantages[:, None]")),
    ),
    "14-kv-cache": (("cache.py", ("old_key", "self.keys[layer_idx]", "axis=2")),),
    "15-paged-kv-offload": (
        (
            "inference/paged_kv.py",
            ("append_tokens", "table.blocks", "self.allocate()", "block_pos"),
        ),
    ),
    "16-speculative-decoding": (
        ("model.py", ("spec_decode", "verify_logits", "mx.array_equal", "n_drafts_accepted")),
    ),
    "17-continuous-batching": (
        ("inference/serving.py", ("_drain_control", "_drain_pending", "_pump", "has_work")),
    ),
    "18-quantization": (
        (
            "layers.py",
            ("quant_mode_for_placement", "_affine_matmul", "linear_mlx_fp4", "assert_never"),
        ),
    ),
    "19-evaluation": (
        ("cli/eval.py", ("mean_loss_nats", "tokens_per_byte", "math.log(2)")),
        ("eval/needle.py", ("marker_id", "answers")),
    ),
    "20-vision-vl2": (
        ("model.py", ("_prepend_vision", "block_ids", "pad_ids")),
        ("vision/connector.py", ("nn.gelu", "self.fc2")),
    ),
}


class TestCourseSnippets(unittest.TestCase):
    def test_every_content_module_is_mapped(self) -> None:
        # All 20 content modules (01-20) show code; each must be drift-guarded.
        mapped = set(SHOWN)
        expected = {
            p.name for p in (ROOT / "course").iterdir() if p.is_dir() and p.name[:2].isdigit()
        }
        expected -= {"00-the-map", "21-capstone"}  # no beat-3 in the framing modules
        self.assertEqual(mapped, expected)

    def test_shown_code_matches_source(self) -> None:
        for module, sources in SHOWN.items():
            readme = (ROOT / "course" / module / "README.md").read_text()
            for src, identifiers in sources:
                source = (ROOT / "baby_whale_v4" / src).read_text()
                for ident in identifiers:
                    self.assertIn(
                        ident, source, f"{module}: `{ident}` shown but not in {src} (renamed?)"
                    )
                    self.assertIn(
                        ident, readme, f"{module}: `{ident}` mapped but not shown in the README"
                    )


if __name__ == "__main__":
    unittest.main()
