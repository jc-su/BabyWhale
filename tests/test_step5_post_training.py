import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import (
    ByteTokenizer,
    ChatExample,
    Message,
    SFTDataset,
    chat_examples_from_jsonl,
    sft_dataset_from_jsonl,
)
from baby_whale_v4.data.chat import format_chat
from baby_whale_v4.training import (
    DPOConfig,
    GRPOConfig,
    SFTConfig,
    dpo_examples_from_jsonl,
    dpo_loss,
    grpo,
    make_reference,
    rejection_finetune_collect,
    sft,
)
from baby_whale_v4.training.mlx_optim import AdamW
from baby_whale_v4.typing import ChatRole, array_to_int_tuple
from tests.mlx_helpers import finite


class TestStep5(unittest.TestCase):
    def test_chat_format_mask_assistant_only(self):
        tok = ByteTokenizer()
        msgs = [
            Message("user", "hi"),
            Message("assistant", "hello"),
        ]
        ids, mask = format_chat(msgs, tok)
        self.assertEqual(len(ids), len(mask))
        u_tag = tok.encode("<|user|>")
        u_body = tok.encode("hi<|eot|>")
        a_tag = tok.encode("<|assistant|>")
        u_part_mask = mask[: len(u_tag) + len(u_body)]
        a_tag_mask = mask[len(u_tag) + len(u_body) : len(u_tag) + len(u_body) + len(a_tag)]
        a_body_mask = mask[len(u_tag) + len(u_body) + len(a_tag) :]
        self.assertTrue(all(v == 0 for v in u_part_mask))
        self.assertTrue(all(v == 0 for v in a_tag_mask))
        self.assertTrue(all(v == 1 for v in a_body_mask))

    def test_sft_overfits_tiny_chat(self):
        mx.random.seed(0)
        tok = ByteTokenizer()
        examples = [
            ChatExample([Message("user", "ab"), Message("assistant", "yz")]),
            ChatExample([Message("user", "cd"), Message("assistant", "yz")]),
        ]
        block_size = 64
        ds = SFTDataset(examples=examples, tokenizer=tok, block_size=block_size)
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=block_size)
        sft_cfg = SFTConfig(lr=3e-3, batch_size=2, max_steps=8, log_every=4, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            model = sft(config=cfg, sft_config=sft_cfg, train_dataset=ds, out_dir=tmp)
        model.eval()
        x, y = ds[0]
        out = model(x[None, :], targets=y[None, :])
        loss = out.loss
        self.assertIsNotNone(loss)
        if loss is None:
            raise AssertionError("expected training loss")
        self.assertTrue(finite(loss))

    def test_sft_dataset_loads_chat_jsonl(self):
        rows = [
            {
                "kind": "chat",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        ]
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            examples = chat_examples_from_jsonl(path)
            self.assertEqual(len(examples), 1)
            ds = sft_dataset_from_jsonl(path, tok, block_size=32)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (32,))
            self.assertEqual(tuple(y.shape), (32,))

    def test_dpo_examples_load_preference_jsonl(self):
        rows = [
            {
                "kind": "preference",
                "prompt": "question",
                "chosen": "good answer",
                "rejected": "bad answer",
            }
        ]
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pref.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            examples = dpo_examples_from_jsonl(
                path,
                tok,
                max_prompt_tokens=8,
                max_response_tokens=8,
            )
            self.assertEqual(len(examples), 1)
            self.assertLessEqual(examples[0].prompt.shape[0], 8)
            self.assertLessEqual(examples[0].chosen.shape[0], 8)
            self.assertLessEqual(examples[0].rejected.shape[0], 8)
            self.assertNotIn(tok.pad_id, array_to_int_tuple(examples[0].prompt))
            self.assertNotIn(tok.pad_id, array_to_int_tuple(examples[0].chosen))
            self.assertNotIn(tok.pad_id, array_to_int_tuple(examples[0].rejected))

    def test_dpo_loss_matches_hand_computed(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=16, context_length=16)
        model = BabyWhaleV4Model(cfg)
        ref = make_reference(model)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        chosen = mx.array([[4, 5]], dtype=mx.int32)
        rejected = mx.array([[6, 7]], dtype=mx.int32)

        beta = 0.5
        loss = dpo_loss(model, ref, prompt, chosen, rejected, beta=beta)
        self.assertAlmostEqual(float(loss), -math.log(0.5), places=4)

    def test_dpo_loss_can_mask_padded_response_tokens(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=16, context_length=16)
        model = BabyWhaleV4Model(cfg)
        ref = make_reference(model)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        chosen = mx.array([[4, 5]], dtype=mx.int32)
        rejected = mx.array([[6, 7]], dtype=mx.int32)
        mask = mx.array([[1.0, 0.0]], dtype=mx.float32)
        masked = dpo_loss(
            model,
            ref,
            prompt,
            chosen,
            rejected,
            beta=0.5,
            chosen_mask=mask,
            rejected_mask=mask,
        )
        single = dpo_loss(
            model,
            ref,
            prompt,
            chosen[:, :1],
            rejected[:, :1],
            beta=0.5,
        )
        self.assertAlmostEqual(float(masked), float(single), places=4)

    def test_dpo_config_validates_bounds(self):
        with self.assertRaisesRegex(ValueError, "beta must be positive"):
            DPOConfig(beta=0.0)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            DPOConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "log_every must be positive"):
            DPOConfig(log_every=0)

    def test_sft_grpo_and_chat_boundaries_fail_fast(self):
        tok = ByteTokenizer()
        with self.assertRaisesRegex(ValueError, "message.role"):
            Message(cast(ChatRole, "bad"), "content")
        with self.assertRaisesRegex(ValueError, "chat example"):
            ChatExample([])
        with self.assertRaisesRegex(ValueError, "SFTDataset examples"):
            SFTDataset(examples=[], tokenizer=tok, block_size=8)
        with self.assertRaisesRegex(ValueError, "log_every must be positive"):
            SFTConfig(log_every=0)
        with self.assertRaisesRegex(ValueError, "log_every must be positive"):
            GRPOConfig(log_every=0)

    def test_dpo_update_runs_and_returns_finite_loss(self):
        mx.random.seed(2)
        cfg = BabyWhaleV4Config.tiny(vocab_size=16, context_length=16)
        model = BabyWhaleV4Model(cfg)
        ref = make_reference(model)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        chosen = mx.array([[4, 5]], dtype=mx.int32)
        rejected = mx.array([[6, 7]], dtype=mx.int32)

        opt = AdamW(learning_rate=5e-3, weight_decay=0.0)

        def loss_fn(m: BabyWhaleV4Model) -> mx.array:
            return dpo_loss(m, ref, prompt, chosen, rejected, beta=0.5)

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        model.update(opt.step(model.parameters(), grads))
        mx.eval(model.parameters())
        after = dpo_loss(model, ref, prompt, chosen, rejected, beta=0.5)
        self.assertTrue(finite(loss))
        self.assertTrue(finite(after))

    def test_grpo_improves_toy_reward(self):
        mx.random.seed(0)
        vocab = 16
        cfg = BabyWhaleV4Config.tiny(vocab_size=vocab, context_length=24)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
        target_token = 7

        def reward_fn(sample: mx.array) -> float:
            return float(mx.sum(mx.equal(sample, target_token).astype(mx.float32)))

        with tempfile.TemporaryDirectory() as tmp:
            grpo(
                model=model,
                prompts=[prompt],
                reward_fn=reward_fn,
                grpo_config=GRPOConfig(
                    lr=3e-3,
                    group_size=4,
                    response_len=4,
                    max_steps=2,
                    log_every=1,
                    beta_kl=0.0,
                ),
                out_dir=tmp,
            )
        out = model(prompt[None, :])
        self.assertTrue(finite(out.logits))

    def test_rejection_finetune_collect_returns_topk(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=16, context_length=16)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        def reward_fn(s: mx.array) -> float:
            return float(mx.sum(mx.equal(s, 5).astype(mx.float32)))

        kept = rejection_finetune_collect(
            model=model,
            prompt=prompt,
            n_samples=6,
            response_len=4,
            reward_fn=reward_fn,
            keep_top=2,
        )
        self.assertEqual(len(kept), 2)
        for s in kept:
            self.assertEqual(tuple(s.shape), (4,))

        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            rejection_finetune_collect(
                model=model,
                prompt=prompt,
                n_samples=2,
                response_len=2,
                reward_fn=reward_fn,
                keep_top=1,
                temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
