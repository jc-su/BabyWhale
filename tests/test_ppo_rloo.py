import tempfile
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.rl import InProcessRolloutEngine, RolloutRequest
from baby_whale_v4.training.ppo import PPOConfig, ppo
from baby_whale_v4.training.rloo import RLOOConfig, _leave_one_out_advantage, rloo
from baby_whale_v4.typing import TokenizerHash, array_to_int_tuple


def _measure_target_reward(model: BabyWhaleV4Model, prompt: mx.array, target: int) -> float:
    mx.random.seed(99)
    engine = InProcessRolloutEngine(
        model=model,
        config=model.config,
        tokenizer_hash=TokenizerHash("probe"),
    )
    requests = [
        RolloutRequest(
            prompt_ids=array_to_int_tuple(prompt),
            options=GenerationOptions(max_new_tokens=8, mode="sample"),
        )
        for _ in range(8)
    ]
    samples = engine.generate_batch(requests)
    counts = [float(sum(1 for t in s.response_ids if t == target)) for s in samples]
    return sum(counts) / len(counts)


class TestPPO(unittest.TestCase):
    def test_ppo_improves_toy_reward(self):
        mx.random.seed(0)
        vocab = 16
        cfg = BabyWhaleV4Config.tiny(vocab_size=vocab, context_length=24)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
        target = 7

        def reward_fn(sample: mx.array) -> float:
            return float(mx.sum(mx.equal(sample, target)))

        before = _measure_target_reward(model, prompt, target)
        with tempfile.TemporaryDirectory() as tmp:
            ppo(
                model=model,
                prompts=[prompt],
                reward_fn=reward_fn,
                ppo_config=PPOConfig(
                    lr=5e-3,
                    clip_eps=0.2,
                    beta_kl=0.0,
                    group_size=8,
                    response_len=8,
                    max_steps=20,
                    log_every=10,
                ),
                out_dir=tmp,
            )
        after = _measure_target_reward(model, prompt, target)
        self.assertGreater(after, before)


class TestRLOO(unittest.TestCase):
    def test_leave_one_out_advantage_for_known_inputs(self):
        rewards = mx.array([1.0, 2.0, 3.0, 4.0])
        adv = _leave_one_out_advantage(rewards)
        # mean of others = (sum - r_i) / (G-1). For G=4, sum=10:
        # adv_0 = 1 - (10-1)/3 = 1 - 3 = -2
        # adv_1 = 2 - (10-2)/3 = 2 - 8/3 ≈ -0.667
        # adv_2 = 3 - (10-3)/3 = 3 - 7/3 ≈ 0.667
        # adv_3 = 4 - (10-4)/3 = 4 - 2 = 2
        expected = mx.array([-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0])
        self.assertLess(float(mx.max(mx.abs(adv - expected))), 1e-5)

    def test_leave_one_out_advantage_sums_to_zero(self):
        mx.random.seed(0)
        rewards = mx.random.normal((8,))
        adv = _leave_one_out_advantage(rewards)
        # Sum_i [r_i - mean(r_{j≠i})]:
        # = sum_i r_i - sum_i (sum_{j≠i} r_j)/(G-1)
        # The second term equals (G * sum r - sum r) / (G-1) = sum r.
        # So total = sum r - sum r = 0.
        self.assertLess(float(mx.abs(mx.sum(adv))), 1e-5)

    def test_rloo_improves_toy_reward(self):
        mx.random.seed(0)
        vocab = 16
        cfg = BabyWhaleV4Config.tiny(vocab_size=vocab, context_length=24)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
        target = 7

        def reward_fn(sample: mx.array) -> float:
            return float(mx.sum(mx.equal(sample, target)))

        before = _measure_target_reward(model, prompt, target)
        with tempfile.TemporaryDirectory() as tmp:
            rloo(
                model=model,
                prompts=[prompt],
                reward_fn=reward_fn,
                rloo_config=RLOOConfig(
                    lr=5e-3,
                    beta_kl=0.0,
                    group_size=8,
                    response_len=8,
                    max_steps=20,
                    log_every=10,
                ),
                out_dir=tmp,
            )
        after = _measure_target_reward(model, prompt, target)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
