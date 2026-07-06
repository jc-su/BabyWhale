import json
import queue
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.prefix_cache import PrefixCache
from baby_whale_v4.inference.server import ServeContext, make_handler
from baby_whale_v4.rl import (
    AsyncRolloutBuffer,
    HTTPRolloutEngine,
    InProcessRolloutEngine,
    LocalRewardHost,
    RolloutRequest,
    RolloutSample,
    ScoredSample,
    SyncRolloutBuffer,
)
from baby_whale_v4.training import GRPOConfig, grpo
from baby_whale_v4.typing import TokenizerHash, array_to_int_tuple
from tests.mlx_helpers import finite


def _model_and_engine() -> tuple[
    BabyWhaleV4Model, BabyWhaleV4Config, ByteTokenizer, InProcessRolloutEngine
]:
    mx.random.seed(0)
    cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=32)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    tok = ByteTokenizer()
    engine = InProcessRolloutEngine(
        model=model,
        config=cfg,
        tokenizer_hash=tok.hash_signature(),
        prefix_cache_capacity=8,
        prefill_chunk=4,
    )
    return model, cfg, tok, engine


class TestRolloutTypes(unittest.TestCase):
    def test_rollout_sample_rejects_mismatched_log_probs(self):
        req = RolloutRequest(
            prompt_ids=(1, 2, 3),
            options=GenerationOptions(max_new_tokens=2, mode="greedy"),
        )
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            RolloutSample(request=req, response_ids=(4, 5), log_probs=(-0.1,), finished=True)

    def test_rollout_request_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt_ids must be non-empty"):
            RolloutRequest(
                prompt_ids=(),
                options=GenerationOptions(max_new_tokens=2, mode="greedy"),
            )


class TestInProcessRollout(unittest.TestCase):
    def test_generate_batch_captures_log_probs(self):
        _, _, _, engine = _model_and_engine()
        request = RolloutRequest(
            prompt_ids=(1, 2, 3, 4),
            options=GenerationOptions(max_new_tokens=4, mode="sample"),
        )
        samples = engine.generate_batch([request, request])
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(len(sample.response_ids), 4)
            self.assertEqual(len(sample.log_probs), 4)
            self.assertTrue(all(p < 0.0 for p in sample.log_probs))

    def test_repeated_prompts_hit_prefix_cache(self):
        _, _, _, engine = _model_and_engine()
        prompt = (5, 6, 7, 8)
        opts = GenerationOptions(max_new_tokens=2, mode="greedy")
        first = engine.generate_batch([RolloutRequest(prompt_ids=prompt, options=opts)])
        before_hits = engine.prefix_cache.hits
        engine.generate_batch([RolloutRequest(prompt_ids=prompt, options=opts)])
        self.assertGreater(engine.prefix_cache.hits, before_hits)
        self.assertEqual(len(first), 1)

    def test_same_batch_repeated_prompts_share_prefill(self):
        _, _, _, engine = _model_and_engine()
        prompt = (5, 6, 7, 8)
        opts = GenerationOptions(max_new_tokens=1, mode="greedy")
        before_hits = engine.prefix_cache.hits
        requests = [RolloutRequest(prompt_ids=prompt, options=opts) for _ in range(3)]
        samples = engine.generate_batch(requests)
        self.assertEqual(len(samples), 3)
        self.assertGreaterEqual(engine.prefix_cache.hits - before_hits, 2)

    def test_sync_weights_invalidates_prefix_cache(self):
        model, _, _, engine = _model_and_engine()
        prompt = (5, 6, 7, 8)
        opts = GenerationOptions(max_new_tokens=2, mode="greedy")
        engine.generate_batch([RolloutRequest(prompt_ids=prompt, options=opts)])
        self.assertGreater(len(engine.prefix_cache), 0)
        engine.sync_weights(model)
        self.assertEqual(len(engine.prefix_cache), 0)

    def test_sync_weights_rejects_foreign_model(self):
        _, cfg, _, engine = _model_and_engine()
        other = BabyWhaleV4Model(cfg)
        with self.assertRaisesRegex(ValueError, "same model instance"):
            engine.sync_weights(other)

    def test_eval_mode_is_restored_after_generate(self):
        model, _, _, engine = _model_and_engine()
        model.train()
        engine.generate_batch(
            [
                RolloutRequest(
                    prompt_ids=(1, 2, 3),
                    options=GenerationOptions(max_new_tokens=2, mode="sample"),
                )
            ]
        )
        self.assertTrue(model.training)


class TestRewardHost(unittest.TestCase):
    def test_local_reward_host_passes_sample(self):
        seen: list[RolloutSample] = []

        def fn(sample: RolloutSample) -> float:
            seen.append(sample)
            return 1.5

        host = LocalRewardHost(fn)
        request = RolloutRequest(
            prompt_ids=(1, 2),
            options=GenerationOptions(max_new_tokens=1, mode="greedy"),
        )
        sample = RolloutSample(request=request, response_ids=(7,), log_probs=(-0.5,), finished=True)
        scores = host.score_batch([sample, sample])
        self.assertEqual(scores, [1.5, 1.5])
        self.assertEqual(len(seen), 2)

    def test_local_reward_host_rejects_non_float(self):
        host = LocalRewardHost(lambda _s: 1)  # int, not float
        sample = RolloutSample(
            request=RolloutRequest(
                prompt_ids=(1,),
                options=GenerationOptions(max_new_tokens=1, mode="greedy"),
            ),
            response_ids=(2,),
            log_probs=(-0.1,),
            finished=True,
        )
        with self.assertRaisesRegex(TypeError, "Python float"):
            host.score(sample)


class TestSyncBuffer(unittest.TestCase):
    def _scored(self, reward: float) -> ScoredSample:
        return ScoredSample(
            sample=RolloutSample(
                request=RolloutRequest(
                    prompt_ids=(1,),
                    options=GenerationOptions(max_new_tokens=1, mode="greedy"),
                ),
                response_ids=(2,),
                log_probs=(-0.1,),
                finished=True,
            ),
            reward=reward,
        )

    def test_add_drain_returns_in_order_and_clears(self):
        buf = SyncRolloutBuffer()
        buf.add_many([self._scored(0.0), self._scored(1.0), self._scored(2.0)])
        self.assertEqual(len(buf), 3)
        items = buf.drain()
        self.assertEqual([s.reward for s in items], [0.0, 1.0, 2.0])
        self.assertEqual(len(buf), 0)


class TestAsyncBuffer(unittest.TestCase):
    def _scored(self, reward: float) -> ScoredSample:
        return ScoredSample(
            sample=RolloutSample(
                request=RolloutRequest(
                    prompt_ids=(1,),
                    options=GenerationOptions(max_new_tokens=1, mode="greedy"),
                ),
                response_ids=(2,),
                log_probs=(-0.1,),
                finished=True,
            ),
            reward=reward,
        )

    def test_producer_drain_join(self):
        buf = AsyncRolloutBuffer(max_size=4)

        def producer(b: AsyncRolloutBuffer) -> None:
            for i in range(3):
                b.add(self._scored(float(i)))

        buf.start(producer)
        buf.join(timeout=5.0)
        items = buf.drain(min_count=3, timeout=5.0)
        self.assertEqual([s.reward for s in items], [0.0, 1.0, 2.0])

    def test_producer_failure_propagates(self):
        buf = AsyncRolloutBuffer(max_size=4)

        def producer(_b: AsyncRolloutBuffer) -> None:
            raise RuntimeError("boom")

        buf.start(producer)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            buf.drain(min_count=0, timeout=5.0)


class TestGRPOWithNewLayer(unittest.TestCase):
    def test_grpo_improves_toy_reward(self):
        mx.random.seed(0)
        vocab = 16
        cfg = BabyWhaleV4Config.tiny(vocab_size=vocab, context_length=24)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
        target_token = 7

        def reward_fn(sample: mx.array) -> float:
            return float(mx.sum(mx.equal(sample, target_token)))

        def measure(model: BabyWhaleV4Model) -> float:
            from baby_whale_v4.rl import (
                InProcessRolloutEngine,
                RolloutRequest,
            )

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
            counts = [float(sum(1 for t in s.response_ids if t == target_token)) for s in samples]
            return sum(counts) / len(counts)

        before = measure(model)
        with tempfile.TemporaryDirectory() as tmp:
            grpo(
                model=model,
                prompts=[prompt],
                reward_fn=reward_fn,
                grpo_config=GRPOConfig(
                    lr=5e-3,
                    group_size=8,
                    response_len=8,
                    max_steps=20,
                    log_every=10,
                    beta_kl=0.0,
                ),
                out_dir=tmp,
            )
        after = measure(model)
        self.assertGreater(after, before)


class TestHTTPRollout(unittest.TestCase):
    def _start_http_server(self) -> tuple[int, HTTPServer, threading.Thread]:
        """Construct the model + engine + server inside the server thread.

        MLX arrays are bound to the thread that allocated them, so building the
        model on the main thread and then handling requests on a server thread
        produces ``Stream(gpu, 0) not in current thread`` errors. This helper
        mirrors the working pattern from ``test_step6_inference``.
        """

        ready: queue.Queue[tuple[int, HTTPServer]] = queue.Queue()

        def run() -> None:
            mx.random.seed(0)
            tok = ByteTokenizer()
            cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
            model = BabyWhaleV4Model(cfg)
            model.eval()
            engine = Engine(
                model=model,
                config=cfg,
                tokenizer_hash=tok.hash_signature(),
                prefix_cache=PrefixCache(capacity=4),
            )
            ctx = ServeContext(engine=engine, tokenizer=tok, config=cfg)
            ctx.start()
            server = HTTPServer(("127.0.0.1", 0), make_handler(ctx))
            ready.put((server.server_address[1], server))
            server.serve_forever()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        port, server = ready.get(timeout=10)
        return port, server, thread

    def test_rollout_endpoint_returns_response_ids_and_log_probs(self):
        port, server, thread = self._start_http_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/rollout",
                data=json.dumps(
                    {
                        "prompt_ids": [1, 2, 3, 4],
                        "options": {"max_new_tokens": 4, "mode": "greedy"},
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            with resp:
                payload = json.loads(resp.read())
            self.assertEqual(len(payload["response_ids"]), 4)
            self.assertEqual(len(payload["log_probs"]), 4)
            self.assertTrue(all(isinstance(p, float) for p in payload["log_probs"]))
            self.assertIs(payload["finished"], True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_rollout_engine_matches_endpoint(self):
        port, server, thread = self._start_http_server()
        try:
            client = HTTPRolloutEngine(f"http://127.0.0.1:{port}")
            request = RolloutRequest(
                prompt_ids=(1, 2, 3, 4),
                options=GenerationOptions(max_new_tokens=4, mode="greedy"),
            )
            samples = client.generate_batch([request])
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0].response_ids), 4)
            self.assertEqual(len(samples[0].log_probs), 4)
            self.assertTrue(samples[0].finished)
            self.assertTrue(finite(mx.array(list(samples[0].log_probs))))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_sync_weights_pushes_checkpoint(self):
        port, server, thread = self._start_http_server()
        try:
            # Construct a fresh model with the SAME config as the server so the
            # config_hash matches and weights can be loaded. The ServeContext's
            # tiny config is BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size,
            # context_length=64); replicate that here.
            mx.random.seed(99)
            tok = ByteTokenizer()
            cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
            client_model = BabyWhaleV4Model(cfg)
            engine_client = HTTPRolloutEngine(f"http://127.0.0.1:{port}")
            engine_client.sync_weights(client_model)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_sync_weights_rejects_mismatched_config(self):
        port, server, thread = self._start_http_server()
        try:
            mx.random.seed(0)
            cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=32)
            client_model = BabyWhaleV4Model(cfg)
            engine_client = HTTPRolloutEngine(f"http://127.0.0.1:{port}")
            with self.assertRaisesRegex(RuntimeError, "sync_weights"):
                engine_client.sync_weights(client_model)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
