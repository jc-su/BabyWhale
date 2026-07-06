import json
import queue
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from typing import cast

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference import (
    Engine,
    GenerationOptions,
    PrefixCache,
    PrefixCacheKey,
    RequestScheduler,
    benchmark_scheduler,
    load_kv_cache_npz,
    save_kv_cache_npz,
)
from baby_whale_v4.inference.server import ServeContext, make_handler
from baby_whale_v4.typing import GenerationMode, TokenizerHash


class TestStep6(unittest.TestCase):
    def _engine(
        self, *, with_cache: bool = True
    ) -> tuple[Engine, BabyWhaleV4Model, BabyWhaleV4Config, ByteTokenizer]:
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        cache = PrefixCache(capacity=8) if with_cache else None
        engine = Engine(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
            prefix_cache=cache,
        )
        return engine, model, cfg, tok

    def test_generate_matches_model_generate(self):
        engine, model, _cfg, tok = self._engine(with_cache=False)
        prompt = tok.encode("hello")
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")
        out_engine = engine.generate(prompt, opts)
        seq = mx.array([prompt], dtype=mx.int32)
        for _ in range(5):
            logits = model(seq).logits[:, -1, :]
            nxt = mx.argmax(logits, axis=-1).reshape(1, 1)
            seq = mx.concatenate([seq, nxt], axis=1)
        ref = seq[0, len(prompt) :].tolist()
        self.assertEqual(out_engine, ref)

    def test_prefix_cache_reuse_skips_prefill(self):
        engine, _, _, tok = self._engine(with_cache=True)
        prompt = tok.encode("the quick brown fox")
        opts = GenerationOptions(max_new_tokens=4, mode="greedy")

        out1 = engine.generate(prompt, opts, request_id="req1")
        prefix_cache = engine.prefix_cache
        self.assertIsNotNone(prefix_cache)
        if prefix_cache is None:
            raise AssertionError("expected prefix cache")
        first_hits = prefix_cache.hits

        out2 = engine.generate(prompt, opts, request_id="req2")
        self.assertEqual(out1, out2)
        self.assertGreater(prefix_cache.hits, first_hits)

    def test_prefix_cache_key_includes_config_hash(self):
        cfg_a = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
        cfg_b = BabyWhaleV4Config.tiny(vocab_size=128, context_length=16)
        prefix = [1, 2, 3]
        key_a = PrefixCacheKey.build(
            prefix_ids=prefix, config=cfg_a, tokenizer_hash=TokenizerHash("t")
        )
        key_b = PrefixCacheKey.build(
            prefix_ids=prefix, config=cfg_b, tokenizer_hash=TokenizerHash("t")
        )
        self.assertNotEqual(key_a.config_hash, key_b.config_hash)
        self.assertNotEqual(key_a, key_b)

    def test_prefix_cache_key_includes_runtime(self):
        cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
        prefix = [1, 2, 3]
        key_metal = PrefixCacheKey.build(
            prefix_ids=prefix,
            config=cfg,
            tokenizer_hash=TokenizerHash("t"),
            runtime="mlx-metal",
        )
        key_cuda = PrefixCacheKey.build(
            prefix_ids=prefix,
            config=cfg,
            tokenizer_hash=TokenizerHash("t"),
            runtime="mlx-cuda",
        )
        self.assertNotEqual(key_metal, key_cuda)

    def test_prefix_cache_preserves_mla_latents(self):
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=tok.vocab_size, context_length=32)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        prefix_cache = PrefixCache(capacity=8)
        engine = Engine(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
            prefix_cache=prefix_cache,
        )
        prompt = tok.encode("latent")
        engine.generate(prompt, GenerationOptions(max_new_tokens=1, mode="greedy"))

        key = PrefixCacheKey.build(
            prefix_ids=prompt,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
        )
        hit = prefix_cache.get(key)
        self.assertIsNotNone(hit)
        if hit is None:
            raise AssertionError("expected prefix cache hit")
        _n, restored, _logits = hit
        self.assertEqual(restored.max_sequence_length(), len(prompt))
        self.assertIsNotNone((restored.latents or [None])[0])

    def test_kv_cache_npz_offload_roundtrip(self):
        _engine, model, cfg, _tok = self._engine(with_cache=False)
        cache = model.empty_cache()
        idx = mx.random.randint(0, cfg.vocab_size, (1, 4))
        model(idx, cache=cache)
        with tempfile.TemporaryDirectory() as tmp:
            report = save_kv_cache_npz(cache, f"{tmp}/cache.npz")
            restored = load_kv_cache_npz(report.path, expected_n_layer=cfg.n_layer)
            self.assertEqual(restored.max_sequence_length(), cache.max_sequence_length())
            self.assertGreater(report.bytes_written, 0)

    def test_kv_cache_npz_offload_roundtrip_preserves_mla_latents(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=64, context_length=16)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        cache = model.empty_cache()
        idx = mx.random.randint(0, cfg.vocab_size, (1, 4))
        model(idx, cache=cache)
        with tempfile.TemporaryDirectory() as tmp:
            report = save_kv_cache_npz(cache, f"{tmp}/mla-cache.npz")
            restored = load_kv_cache_npz(report.path, expected_n_layer=cfg.n_layer)
            self.assertEqual(restored.max_sequence_length(), cache.max_sequence_length())
            self.assertIsNotNone((restored.latents or [None])[0])
            self.assertGreater(restored.stats().bytes, 0)

    def test_generate_rejects_total_context_overflow(self):
        engine, _, cfg, _tok = self._engine(with_cache=False)
        prompt = [1] * (cfg.context_length - 1)
        with self.assertRaisesRegex(ValueError, "prompt plus generation exceeds"):
            engine.generate(prompt, GenerationOptions(max_new_tokens=2, mode="greedy"))

    def test_generation_options_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unsupported mode"):
            GenerationOptions(mode=cast(GenerationMode, "beam"))

    def test_chunked_prefill_two_concurrent_requests(self):
        engine, _, _, tok = self._engine(with_cache=False)
        sched = RequestScheduler(engine, prefill_chunk=2)
        opts = GenerationOptions(max_new_tokens=3, mode="greedy")

        long_prompt = tok.encode("this is a longer prompt for prefill")
        short_prompt = tok.encode("hi")

        s1 = sched.submit("long", long_prompt, opts)
        s2 = sched.submit("short", short_prompt, opts)
        completed = sched.run_until_done()
        self.assertEqual(len(completed), 2)
        self.assertTrue(s1.finished)
        self.assertTrue(s2.finished)
        self.assertEqual(s1.total_emitted, 3)
        self.assertEqual(s2.total_emitted, 3)
        self.assertGreater(sched.stats.prefill_steps, 1)
        self.assertGreater(sched.stats.decode_steps, 0)

    def test_scheduler_decode_runs_while_prefill_remains(self):
        engine, _, _, tok = self._engine(with_cache=False)
        sched = RequestScheduler(engine, prefill_chunk=1)
        opts = GenerationOptions(max_new_tokens=3, mode="greedy")
        ready = tok.encode("a")
        long = tok.encode("abcdefghij")
        sched.submit("ready", ready, opts)
        sched.submit("long", long, opts)
        sched.tick()
        sched.tick()
        decode_before = sched.stats.decode_steps
        sched.tick()
        self.assertGreater(sched.stats.decode_steps, decode_before)

    def test_http_server_generate_endpoint(self):
        port, server, thread, cfg = self._start_http_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"prompt": "hi", "max_new_tokens": 3, "mode": "greedy"}).encode(
                    "utf-8"
                ),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            with resp:
                payload = json.loads(resp.read())
            self.assertIn("completion", payload)
            self.assertEqual(payload["generated_tokens"], 3)
            health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            with health:
                hpayload = json.loads(health.read())
            self.assertEqual(hpayload["config_hash"], cfg.config_hash())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_server_openai_chat_completions_endpoint(self):
        port, server, thread, _cfg = self._start_http_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "baby-whale-v4-local",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "mode": "greedy",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            with resp:
                payload = json.loads(resp.read())
            self.assertEqual(payload["object"], "chat.completion")
            self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
            # max_tokens is an upper bound; the server now also stops on EOS and
            # reports a real finish_reason instead of a hardcoded "length".
            self.assertLessEqual(payload["usage"]["completion_tokens"], 2)
            self.assertGreaterEqual(payload["usage"]["completion_tokens"], 1)
            self.assertIn(payload["choices"][0]["finish_reason"], ("stop", "length"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_server_openai_chat_streaming_endpoint(self):
        port, server, thread, _cfg = self._start_http_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "baby-whale-v4-local",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "stream": True,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            with resp:
                content_type = resp.headers["Content-Type"]
                body = resp.read().decode("utf-8")
            self.assertEqual(content_type, "text/event-stream")
            self.assertIn("chat.completion.chunk", body)
            self.assertIn("data: [DONE]", body)
            # Real incremental streaming: one content chunk per decoded token
            # (finish_reason None) plus exactly one final finish_reason chunk —
            # not a single replayed blob.
            events = [
                json.loads(line[len("data: ") :])
                for line in body.splitlines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]
            content_chunks = [e for e in events if e["choices"][0]["finish_reason"] is None]
            final_chunks = [e for e in events if e["choices"][0]["finish_reason"] is not None]
            self.assertGreaterEqual(len(content_chunks), 1)
            self.assertEqual(len(final_chunks), 1)
            self.assertIn(final_chunks[0]["choices"][0]["finish_reason"], ("stop", "length"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_compare_inference_configs_reports_strategies(self):
        from baby_whale_v4.inference.bench import compare_inference_configs

        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=32)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        prompts = [tok.encode("shared one"), tok.encode("shared two"), tok.encode("shared one")]
        comparison = compare_inference_configs(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
            prompts=prompts,
            options=GenerationOptions(max_new_tokens=4, mode="greedy"),
            prefill_chunk=2,
            quant_modes=("int8-weight",),
        )
        names = [name for name, _ in comparison.rows]
        # KV strategies always present; tiny config is sliding_mqa so paged applies.
        self.assertEqual(names[:3], ["no-cache", "prefix-cache", "paged"])
        self.assertIn("quant:int8-weight", names)
        for _name, bench in comparison.rows:
            self.assertGreater(bench.decode_tokens_per_sec, 0)
            self.assertEqual(bench.completed, len(prompts))
        self.assertEqual(len(comparison.as_rows()), len(comparison.rows))
        self.assertIn(comparison.fastest(), names)

    def test_inference_benchmark_reports_scheduler_metrics(self):
        engine, _, _, tok = self._engine(with_cache=True)
        bench = benchmark_scheduler(
            engine=engine,
            prompts=[tok.encode("hello"), tok.encode("hello")],
            options=GenerationOptions(max_new_tokens=2, mode="greedy"),
            prefill_chunk=2,
        )
        self.assertEqual(bench.requests, 2)
        self.assertEqual(bench.completed, 2)
        self.assertEqual(bench.generated_tokens, 4)
        self.assertGreater(bench.decode_tokens_per_sec, 0)

    def test_http_server_rejects_invalid_generate_payload(self):
        port, server, thread, _cfg = self._start_http_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"prompt": "hi", "mode": "beam"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_err:
                urllib.request.urlopen(req, timeout=10)
            err = ctx_err.exception
            self.assertEqual(err.code, 400)
            payload = json.loads(err.read())
            err.close()
            self.assertIn("mode must be one of", payload["error"])

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"prompt": "hi", "unknown": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_err:
                urllib.request.urlopen(req, timeout=10)
            err = ctx_err.exception
            self.assertEqual(err.code, 400)
            payload = json.loads(err.read())
            err.close()
            self.assertIn("unknown JSON keys", payload["error"])

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"max_new_tokens": 1}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_err:
                urllib.request.urlopen(req, timeout=10)
            err = ctx_err.exception
            self.assertEqual(err.code, 400)
            payload = json.loads(err.read())
            err.close()
            self.assertIn("prompt is required", payload["error"])

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"prompt": ""}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_err:
                urllib.request.urlopen(req, timeout=10)
            err = ctx_err.exception
            self.assertEqual(err.code, 400)
            payload = json.loads(err.read())
            err.close()
            self.assertIn("prompt must be non-empty", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_server_serves_concurrent_requests(self):
        # Continuous batching over HTTP: fire many requests at once through a
        # ThreadingHTTPServer; each handler thread submits to the one shared
        # serving loop and all complete correctly.
        from http.server import ThreadingHTTPServer

        engine, _, cfg, tok = self._engine(with_cache=True)
        ctx = ServeContext(engine=engine, tokenizer=tok, config=cfg)
        ctx.start()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        results: dict[int, dict] = {}

        def client(idx: int, prompt: str) -> None:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps({"prompt": prompt, "max_new_tokens": 4, "mode": "greedy"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                results[idx] = json.loads(resp.read())

        try:
            prompts = ["alpha", "beta", "gamma", "delta", "epsilon"]
            clients = [threading.Thread(target=client, args=(i, p)) for i, p in enumerate(prompts)]
            for c in clients:
                c.start()
            for c in clients:
                c.join(timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            ctx.stop()
        self.assertEqual(len(results), 5)
        for payload in results.values():
            self.assertEqual(payload["generated_tokens"], 4)

    def _start_http_server(self) -> tuple[int, HTTPServer, threading.Thread, BabyWhaleV4Config]:
        ready: queue.Queue[tuple[int, HTTPServer, BabyWhaleV4Config]] = queue.Queue()

        def run() -> None:
            engine, _, cfg, tok = self._engine(with_cache=True)
            ctx = ServeContext(engine=engine, tokenizer=tok, config=cfg)
            ctx.start()
            server = HTTPServer(("127.0.0.1", 0), make_handler(ctx))
            ready.put((server.server_address[1], server, cfg))
            server.serve_forever()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        port, server, cfg = ready.get(timeout=10)
        return port, server, thread, cfg


if __name__ == "__main__":
    unittest.main()
