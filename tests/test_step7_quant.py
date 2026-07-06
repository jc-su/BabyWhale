import tempfile
import unittest
from typing import cast

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import SyntheticCopyDataset
from baby_whale_v4.kernels.fp4_training import linear_weight_grad_metal
from baby_whale_v4.kernels.metal import probe_metal_kernel_runtime
from baby_whale_v4.layers import WhaleLinear
from baby_whale_v4.mlx_fp4 import (
    MLXFP4Mode,
    linear_mlx_fp4,
    linear_mlx_fp4_train,
    quantize_weight_mlx_fp4,
)
from baby_whale_v4.quantization import (
    apply_fp4_expert_export,
    apply_weight_quantization,
)
from baby_whale_v4.training import LoRAConfig, PretrainConfig, attach_lora_adapters, pretrain
from baby_whale_v4.training.mlx_optim import Muon
from tests.mlx_helpers import finite, max_abs


class TestStep7(unittest.TestCase):
    def test_int8_weight_linear_matches_dequantized_reference(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 64))
        x = mx.random.normal((2, 3, 64))
        layer = WhaleLinear(64, 5, bias=False, quant_mode="int8-weight")
        layer.inner.weight = w
        out = layer(x)
        packed, scales, biases = mx.quantize(w, group_size=64, bits=8, mode="affine")
        deq = mx.dequantize(
            packed, scales, biases, group_size=64, bits=8, mode="affine", dtype=w.dtype
        )
        ref = x @ deq.T
        mx.eval(out, ref)
        self.assertEqual(out.shape, (2, 3, 5))
        self.assertTrue(finite(out))
        self.assertLess(max_abs(out - ref), 1e-4)

    def test_int4_weight_linear_lossier_than_int8(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 64))
        x = mx.random.normal((4, 64))
        ref = x @ w.T
        layer8 = WhaleLinear(64, 5, bias=False, quant_mode="int8-weight")
        layer8.inner.weight = w
        layer4 = WhaleLinear(64, 5, bias=False, quant_mode="int4-weight")
        layer4.inner.weight = w
        err8 = max_abs(layer8(x) - ref)
        err4 = max_abs(layer4(x) - ref)
        self.assertGreater(err4, err8)

    def test_int8_weight_linear_rejects_bad_input_dim(self):
        layer = WhaleLinear(20, 5, bias=False, quant_mode="int8-weight")
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            layer(mx.random.normal((1, 20)))

    def test_mlx_fp4_rejects_bad_group_shape(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size=32"):
            quantize_weight_mlx_fp4(mx.random.normal((4, 20)), "mxfp4")

    def test_mlx_fp4_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unsupported MLX FP4 mode"):
            quantize_weight_mlx_fp4(
                mx.random.normal((4, 32)),
                cast(MLXFP4Mode, "bad-mode"),
            )

    def test_mlx_fp4_pack_and_dequantize_for_supported_modes(self):
        mx.random.seed(0)
        cases = (("mxfp4", 32), ("nvfp4", 16))
        for mode, group_size in cases:
            with self.subTest(mode=mode):
                w = mx.random.normal((5, group_size * 2))
                packed = quantize_weight_mlx_fp4(w, mode)
                dequantized = packed.dequantize()
                self.assertEqual(packed.shape, tuple(w.shape))
                self.assertEqual(packed.group_size, group_size)
                self.assertEqual(dequantized.shape, w.shape)
                self.assertTrue(finite(dequantized))

    def test_mlx_fp4_linear_matches_dequantized_reference(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 32))
        x = mx.random.normal((2, 3, 32))
        bias = mx.random.normal((5,))
        packed = quantize_weight_mlx_fp4(w, "mxfp4")
        out = linear_mlx_fp4(x, packed, bias)
        ref = x @ packed.dequantize().T + bias
        self.assertEqual(out.shape, (2, 3, 5))
        self.assertTrue(finite(out))
        self.assertLess(max_abs(out - ref), 1e-4)

    def test_mlx_fp4_train_custom_vjp_has_weight_gradient(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 32))
        x = mx.random.normal((2, 3, 32))
        y = mx.random.normal((2, 3, 5))

        def loss_fn(weight: mx.array) -> mx.array:
            out = linear_mlx_fp4_train(x, weight, mode="mxfp4")
            return mx.mean(mx.square(out - y))

        loss, grad = mx.value_and_grad(loss_fn)(w)
        mx.eval(loss, grad)
        self.assertTrue(finite(loss))
        self.assertEqual(grad.shape, w.shape)
        self.assertGreater(max_abs(grad), 0.0)

    def test_mlx_fp4_train_recompute_custom_vjp_has_weight_gradient(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 32))
        x = mx.random.normal((2, 3, 32))
        y = mx.random.normal((2, 3, 5))

        def loss_fn(weight: mx.array) -> mx.array:
            out = linear_mlx_fp4_train(x, weight, mode="mxfp4", cache_policy="recompute")
            return mx.mean(mx.square(out - y))

        loss, grad = mx.value_and_grad(loss_fn)(w)
        mx.eval(loss, grad)
        self.assertTrue(finite(loss))
        self.assertEqual(grad.shape, w.shape)
        self.assertGreater(max_abs(grad), 0.0)

    def test_custom_metal_kernel_weight_grad_matches_mlx(self):
        status = probe_metal_kernel_runtime()
        self.assertTrue(status.supported, status.reason)
        mx.random.seed(0)
        x = mx.random.normal((4, 32), dtype=mx.float32)
        dy = mx.random.normal((4, 5), dtype=mx.float32)
        out = linear_weight_grad_metal(dy, x)
        ref = dy.T @ x
        mx.eval(out, ref)
        self.assertLess(max_abs(out - ref), 1e-5)

    def test_mlx_fp4_train_metal_custom_vjp_has_weight_gradient(self):
        mx.random.seed(0)
        w = mx.random.normal((5, 32))
        x = mx.random.normal((2, 3, 32))
        y = mx.random.normal((2, 3, 5))

        def loss_fn(weight: mx.array) -> mx.array:
            out = linear_mlx_fp4_train(x, weight, mode="mxfp4", weight_grad="metal")
            return mx.mean(mx.square(out - y))

        loss, grad = mx.value_and_grad(loss_fn)(w)
        mx.eval(loss, grad)
        self.assertTrue(finite(loss))
        self.assertEqual(grad.shape, w.shape)
        self.assertGreater(max_abs(grad), 0.0)

    def test_apply_weight_quantization_keeps_logits_finite(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        x = mx.random.randint(0, 32, (1, 8))
        n = apply_weight_quantization(model, "int8-weight")
        self.assertGreater(n, 0)
        quantized = model(x).logits
        self.assertTrue(finite(quantized))

    def test_lora_adapters_attach_to_attention_and_change_after_update(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(cfg)
        report = attach_lora_adapters(
            model, LoRAConfig(rank=2, alpha=4.0, placements=("attention",))
        )
        self.assertGreater(report.attached, 0)
        block = next(iter(model.blocks.values()))
        self.assertIsNotNone(block.attn.q_proj.lora_a)
        x = mx.random.randint(0, 32, (1, 4))
        out = model(x).logits
        self.assertTrue(finite(out))

    def test_muon_optimizer_updates_matrix_params(self):
        mx.random.seed(0)
        opt = Muon(learning_rate=1e-2)
        params: dict[str, object] = {"w": mx.random.normal((4, 4))}
        grads: dict[str, object] = {"w": mx.ones((4, 4))}
        updated = opt.step(params, grads)
        self.assertFalse(bool(mx.array_equal(cast(mx.array, updated["w"]), params["w"])))

    def test_pretrain_accepts_muon_optimizer(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        ds = SyntheticCopyDataset(n_samples=4, seq_len=16, vocab_size=32, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            model = pretrain(
                config=cfg,
                pretrain_config=PretrainConfig(
                    optimizer="muon",
                    max_steps=1,
                    batch_size=2,
                    log_every=1,
                ),
                train_dataset=ds,
                out_dir=tmp,
            )
        self.assertTrue(finite(model(mx.random.randint(0, 32, (1, 4))).logits))

    def test_apply_fp4_native_switches_linear_layers(self):
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(cfg)
        n = apply_weight_quantization(model, "fp4-native")
        self.assertGreater(n, 0)
        out = model(mx.random.randint(0, 32, (1, 4))).logits
        self.assertTrue(finite(out))

    def test_apply_fp4_expert_export_only_switches_moe_experts(self):
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(cfg)
        n = apply_fp4_expert_export(model)
        expected = cfg.n_layer * (cfg.n_expert + cfg.n_shared_expert) * 3
        self.assertEqual(n, expected)

        block = next(iter(model.blocks.values()))
        self.assertEqual(block.attn.q_proj.quant_mode, "none")
        self.assertEqual(block.moe.router.quant_mode, "none")
        self.assertEqual(block.moe.shared_expert.w_gate.quant_mode, "fp4-native")
        self.assertEqual(block.moe.experts["expert_0"].w_down.quant_mode, "fp4-native")
        self.assertEqual(model.lm_head.quant_mode, "none")

        out = model(mx.random.randint(0, 32, (1, 4))).logits
        self.assertTrue(finite(out))

    def test_fp4_native_linear_reuses_packed_weight_until_weight_changes(self):
        mx.random.seed(0)
        layer = WhaleLinear(32, 5, quant_mode="fp4-native")
        x = mx.random.normal((2, 32))
        first = layer(x)
        cached = layer._fp4_cached_weight
        second = layer(x)
        mx.eval(first, second)
        self.assertIs(cached, layer._fp4_cached_weight)
        layer.inner.weight = layer.inner.weight + mx.array(0.01, dtype=layer.inner.weight.dtype)
        _ = layer(x)
        self.assertIsNot(cached, layer._fp4_cached_weight)


if __name__ == "__main__":
    unittest.main()
