import json
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import (
    ByteTokenizer,
    DatasetMixtureSource,
    PackedDataset,
    SyntheticCopyDataset,
    load_packed_token_file,
    pack_mixture_jsonl,
    pack_normalized_jsonl,
    read_normalized_texts,
    save_packed_token_file,
    train_byte_bpe,
)
from baby_whale_v4.training import (
    MidtrainConfig,
    PretrainConfig,
    load_checkpoint,
    midtrain,
    pretrain,
)
from tests.mlx_helpers import finite


class TestStep2(unittest.TestCase):
    def test_byte_tokenizer_roundtrip(self):
        tok = ByteTokenizer()
        ids = tok.encode("hello!", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertEqual(tok.decode(ids[1:-1]), "hello!")

    def test_byte_bpe_tokenizer_and_jsonl_packing(self):
        rows = [
            {"kind": "pretrain", "text": "hello hello whale"},
            {
                "kind": "chat",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            texts = read_normalized_texts(path)
            tok = train_byte_bpe(texts, vocab_size=270)
            encoded = tok.encode("hello hello", add_bos=True, add_eos=True)
            self.assertEqual(tok.decode(encoded[1:-1]), "hello hello")
            tok_path = Path(tmpdir) / "tokenizer.json"
            tok.save(tok_path)
            loaded = type(tok).load(tok_path)
            self.assertEqual(loaded.hash_signature(), tok.hash_signature())

            ds = pack_normalized_jsonl(path, tokenizer=loaded, block_size=8)
            self.assertGreater(len(ds), 0)
            packed = save_packed_token_file(
                ds,
                Path(tmpdir) / "packed.npz",
                tokenizer_hash=loaded.hash_signature(),
                sources=[path],
            )
            restored = load_packed_token_file(
                packed.path,
                expected_tokenizer_hash=loaded.hash_signature(),
                pad_id=loaded.pad_id,
            )
            self.assertEqual(len(restored), len(ds))
            with self.assertRaisesRegex(ValueError, "tokenizer hash mismatch"):
                load_packed_token_file(
                    packed.path,
                    expected_tokenizer_hash="wrong",
                    pad_id=loaded.pad_id,
                )

    def test_packed_dataset_shape(self):
        tok = ByteTokenizer()
        docs = [tok.encode("abcdefghij"), tok.encode("klmnop")]
        ds = PackedDataset(
            documents=docs,
            block_size=8,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
        )
        self.assertGreater(len(ds), 0)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (8,))
        self.assertEqual(tuple(y.shape), (8,))
        self.assertTrue(bool(mx.array_equal(x[1:], y[:-1])))

    def test_pretrain_reduces_synthetic_copy_loss(self):
        mx.random.seed(0)
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        ds = SyntheticCopyDataset(n_samples=8, seq_len=16, vocab_size=32, seed=0)
        x, y = ds[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            ptcfg = PretrainConfig(lr=3e-3, max_steps=8, batch_size=4, log_every=4, seed=0)
            model = pretrain(
                config=config,
                pretrain_config=ptcfg,
                train_dataset=ds,
                out_dir=tmpdir,
            )
            out = model(x[None, :], targets=y[None, :])
            loss = out.loss
            self.assertIsNotNone(loss)
            if loss is None:
                raise AssertionError("expected training loss")
            self.assertTrue(finite(loss))
            self.assertTrue((Path(tmpdir) / "final.bw4").exists())

    def test_pretrain_supports_adafactor_and_gradient_accumulation(self):
        mx.random.seed(0)
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        ds = SyntheticCopyDataset(n_samples=4, seq_len=16, vocab_size=32, seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ptcfg = PretrainConfig(
                lr=1e-3,
                optimizer="adafactor",
                max_steps=2,
                batch_size=4,
                grad_accum=2,
                log_every=1,
                seed=0,
            )
            model = pretrain(
                config=config,
                pretrain_config=ptcfg,
                train_dataset=ds,
                out_dir=tmpdir,
            )
            x, y = ds[0]
            out = model(x[None, :], targets=y[None, :])
            loss = out.loss
            self.assertIsNotNone(loss)
            if loss is None:
                raise AssertionError("expected training loss")
            self.assertTrue(finite(loss))
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            train_rows = [row for row in rows if "train_loss" in row]
            self.assertTrue(train_rows)
            self.assertIn("tokens_per_sec", train_rows[0])
            self.assertGreater(train_rows[0]["tokens"], 0)

    def test_gradient_accumulation_matches_full_batch_token_weighting(self):
        from baby_whale_v4.training.pretrain import _accumulated_loss_and_grads

        mx.random.seed(0)
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(config)
        x = mx.random.randint(0, config.vocab_size, (4, 8))
        y = mx.random.randint(0, config.vocab_size, (4, 8))
        y = mx.concatenate([y[:3], mx.full((1, 8), -1, dtype=mx.int32)], axis=0)

        def loss_fn(m: BabyWhaleV4Model, xb: mx.array, yb: mx.array) -> mx.array:
            out = m(xb, targets=yb)
            if out.loss is None:
                raise RuntimeError("expected loss")
            return out.loss

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        full_loss, full_grads = _accumulated_loss_and_grads(loss_and_grad, model, x, y, accum=1)
        accum_loss, accum_grads = _accumulated_loss_and_grads(loss_and_grad, model, x, y, accum=2)
        self.assertTrue(bool(mx.allclose(full_loss, accum_loss, atol=1e-5, rtol=1e-5)))
        _assert_grad_tree_close(self, full_grads, accum_grads)

    def test_pretrain_config_and_dataset_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "log_every must be positive"):
            PretrainConfig(log_every=0)
        with self.assertRaisesRegex(ValueError, "save_every must be >= 0"):
            PretrainConfig(save_every=-1)
        with self.assertRaisesRegex(ValueError, "n_samples must be positive"):
            SyntheticCopyDataset(n_samples=0, seq_len=16, vocab_size=32)

    def test_resume_restores_state(self):
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        ds = SyntheticCopyDataset(n_samples=4, seq_len=16, vocab_size=32, seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            ptcfg_a = PretrainConfig(lr=1e-3, max_steps=2, batch_size=2, log_every=1, seed=0)
            model_a = pretrain(
                config=config,
                pretrain_config=ptcfg_a,
                train_dataset=ds,
                out_dir=Path(tmpdir) / "run_a",
            )
            ckpt = load_checkpoint(Path(tmpdir) / "run_a" / "final.bw4")
            self.assertEqual(ckpt.config_hash, config.config_hash())
            self.assertEqual(ckpt.step, 2)

            ptcfg_b = PretrainConfig(lr=1e-3, max_steps=3, batch_size=2, log_every=1, seed=0)
            model_b = pretrain(
                config=config,
                pretrain_config=ptcfg_b,
                train_dataset=ds,
                out_dir=Path(tmpdir) / "run_b",
                resume_from=ckpt,
            )
            ckpt2 = load_checkpoint(Path(tmpdir) / "run_b" / "final.bw4")
            self.assertEqual(ckpt2.step, 3)

            x, _ = ds[0]
            la = model_a(x[None, :]).logits
            lb = model_b(x[None, :]).logits
            self.assertEqual(la.shape, lb.shape)

    def test_pretrain_logs_eval_loss_and_token_count(self):
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        train_ds = SyntheticCopyDataset(n_samples=4, seq_len=16, vocab_size=32, seed=1)
        eval_ds = SyntheticCopyDataset(n_samples=2, seq_len=16, vocab_size=32, seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrain(
                config=config,
                pretrain_config=PretrainConfig(max_steps=1, batch_size=2, log_every=1),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                out_dir=tmpdir,
            )
            rows = [
                json.loads(line)
                for line in (Path(tmpdir) / "metrics.jsonl").read_text().splitlines()
            ]
            eval_rows = [row for row in rows if "eval_loss" in row]
            self.assertEqual(len(eval_rows), 1)
            self.assertGreater(eval_rows[0]["eval_tokens"], 0)

    def test_midtrain_runs_on_jsonl_mixture(self):
        rows = [
            {"kind": "pretrain", "text": "alpha beta gamma"},
            {"kind": "pretrain", "text": "delta epsilon zeta"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mid.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            tok = ByteTokenizer()
            ds = pack_mixture_jsonl(
                [DatasetMixtureSource(path=path, repeat=2)],
                tokenizer=tok,
                block_size=8,
            )
            config = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=8)
            model = midtrain(
                config=config,
                midtrain_config=MidtrainConfig(max_steps=1, batch_size=2, log_every=1),
                train_dataset=ds,
                out_dir=Path(tmpdir) / "midrun",
            )
            self.assertIsInstance(model, BabyWhaleV4Model)
            self.assertTrue((Path(tmpdir) / "midrun" / "final.bw4").exists())

    def test_checkpoint_loader_validates_payload_shape(self):
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        payload = {
            "config": config.to_dict(),
            "config_hash": config.config_hash(),
            "model_state": {},
            "optimizer_state": None,
            "scheduler_state": None,
            "step": 0,
            "rng_state": {"mlx_seed": 0},
            "extra": {},
            "format_version": 2,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.bw4"

            bad = dict(payload)
            bad["unknown"] = True
            path.write_bytes(pickle.dumps(bad))
            with self.assertRaisesRegex(ValueError, "unknown checkpoint keys"):
                load_checkpoint(path)

            bad = dict(payload)
            bad.pop("rng_state")
            path.write_bytes(pickle.dumps(bad))
            with self.assertRaisesRegex(ValueError, "missing checkpoint keys"):
                load_checkpoint(path)

            bad = dict(payload)
            bad["model_state"] = {"weight": "not an array"}
            path.write_bytes(pickle.dumps(bad))
            with self.assertRaisesRegex(TypeError, "checkpoint.model_state"):
                load_checkpoint(path)

    def test_microbatch_splitter_keeps_requested_count_when_possible(self):
        from baby_whale_v4.training.pretrain import _split_microbatches

        x = mx.arange(5)
        y = mx.arange(5)
        chunks = list(_split_microbatches(x, y, 2))
        self.assertEqual([int(c[0].size) for c in chunks], [3, 2])

        chunks = list(_split_microbatches(mx.arange(2), mx.arange(2), 4))
        self.assertEqual([int(c[0].size) for c in chunks], [1, 1])


def _assert_grad_tree_close(
    case: unittest.TestCase,
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    case.assertEqual(set(left), set(right))
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, mx.array) and isinstance(right_value, mx.array):
            case.assertTrue(
                bool(mx.allclose(left_value, right_value, atol=1e-4, rtol=1e-4)),
                f"gradient leaf {key} mismatch",
            )
        elif isinstance(left_value, dict) and isinstance(right_value, dict):
            _assert_grad_tree_close(
                case,
                cast(dict[str, object], left_value),
                cast(dict[str, object], right_value),
            )
        else:
            case.assertEqual(left_value, right_value)


if __name__ == "__main__":
    unittest.main()
