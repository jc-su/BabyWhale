import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import GenerationOptions
from baby_whale_v4.rl import (
    CodeProblem,
    CodeRewardConfig,
    CodeRewardHost,
    InProcessRolloutEngine,
    RolloutRequest,
    RolloutSample,
    execute_python_with_tests,
    extract_python_code,
    load_problems_from_jsonl,
    problem_index,
    save_problems_to_jsonl,
)
from baby_whale_v4.typing import ProblemId

# ---------- Sandbox ----------


class TestExecutePythonWithTests(unittest.TestCase):
    def test_passing_solution_passes_all_tests(self):
        result = execute_python_with_tests(
            solution_code="def add(a, b):\n    return a + b\n",
            tests=["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.n_tests, 2)
        self.assertEqual(result.n_passed, 2)
        self.assertIsNone(result.error)
        self.assertGreater(result.duration_ms, 0)

    def test_failing_test_marks_failure_with_partial_count(self):
        result = execute_python_with_tests(
            solution_code="def add(a, b):\n    return a - b\n",
            tests=[
                "assert add(2, 3) == 5",
                "assert add(0, 0) == 0",
            ],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.n_tests, 2)
        self.assertEqual(result.n_passed, 1)
        self.assertIsNotNone(result.error)

    def test_syntax_error_returns_failure_not_exception(self):
        result = execute_python_with_tests(
            solution_code="def broken( :\n  pass\n",
            tests=["assert True"],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.n_passed, 0)
        self.assertIsNotNone(result.error)
        self.assertIn("SyntaxError", result.error or "")

    def test_runtime_error_in_solution_does_not_kill_test_loop(self):
        result = execute_python_with_tests(
            solution_code="def boom():\n    raise RuntimeError('nope')\n",
            tests=[
                "assert True",
                "boom()",
                "assert 1 == 1",
            ],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.n_passed, 2)
        self.assertEqual(result.n_tests, 3)

    def test_timeout_caps_long_running_solutions(self):
        result = execute_python_with_tests(
            solution_code="while True: pass\n",
            tests=["assert True"],
            timeout_sec=0.5,
        )
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.error)
        self.assertIn("timeout", result.error or "")

    def test_humaneval_style_check_function_works(self):
        result = execute_python_with_tests(
            solution_code="def add(a, b):\n    return a + b\n",
            tests=[
                "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\ncheck(add)",
            ],
        )
        self.assertTrue(result.passed)

    def test_rejects_empty_tests(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            execute_python_with_tests(solution_code="pass", tests=[])

    def test_rejects_nonpositive_timeout(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            execute_python_with_tests(
                solution_code="pass",
                tests=["assert True"],
                timeout_sec=0,
            )


# ---------- CodeProblem + JSONL ----------


class TestCodeProblems(unittest.TestCase):
    def _problem(self) -> CodeProblem:
        return CodeProblem(
            problem_id=ProblemId("t/1"),
            prompt="Write a function `add(a,b)`:\n```python\n",
            tests=("assert add(1, 2) == 3",),
            canonical_solution="def add(a, b):\n    return a + b\n",
            entry_point="add",
        )

    def test_rejects_empty_tests(self):
        with self.assertRaisesRegex(ValueError, "tests"):
            CodeProblem(problem_id=ProblemId("t"), prompt="p", tests=())

    def test_rejects_blank_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            CodeProblem(problem_id=ProblemId("t"), prompt="", tests=("assert True",))

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problems.jsonl"
            save_problems_to_jsonl([self._problem()], path)
            loaded = load_problems_from_jsonl(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0], self._problem())

    def test_problem_index_rejects_duplicates(self):
        a = self._problem()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            problem_index([a, a])


# ---------- Code extraction ----------


class TestExtractPythonCode(unittest.TestCase):
    def test_extracts_python_fenced_block(self):
        text = "Here is the answer:\n```python\ndef f(): return 1\n```\nDone."
        self.assertEqual(extract_python_code(text), "def f(): return 1")

    def test_extracts_unlabeled_fenced_block(self):
        text = "```\ndef g(): return 2\n```"
        self.assertEqual(extract_python_code(text), "def g(): return 2")

    def test_falls_back_to_raw_text(self):
        text = "def h(): return 3"
        self.assertEqual(extract_python_code(text), "def h(): return 3")


# ---------- CodeRewardHost ----------


class TestCodeRewardHost(unittest.TestCase):
    def _host(self, *, partial: bool = False) -> CodeRewardHost:
        problem = CodeProblem(
            problem_id=ProblemId("t/1"),
            prompt="add",
            tests=(
                "assert add(1, 2) == 3",
                "assert add(0, 0) == 0",
                "assert add(-1, 1) == 0",
            ),
        )
        tok = ByteTokenizer()
        return CodeRewardHost(
            problems=problem_index([problem]),
            decode=lambda ids: tok.decode(ids),
            config=CodeRewardConfig(timeout_sec=5.0, partial_credit=partial),
        )

    def _sample(self, text: str, problem_id: str = "t/1") -> RolloutSample:
        tok = ByteTokenizer()
        ids = tok.encode(text)
        request = RolloutRequest(
            prompt_ids=(1, 2, 3),
            options=GenerationOptions(max_new_tokens=len(ids), mode="greedy"),
            metadata={"problem_id": problem_id},
        )
        return RolloutSample(
            request=request,
            response_ids=tuple(ids),
            log_probs=tuple([-1.0] * len(ids)),
            finished=True,
        )

    def test_correct_solution_scores_pass_reward(self):
        host = self._host()
        sample = self._sample("```python\ndef add(a, b):\n    return a + b\n```")
        self.assertEqual(host.score(sample), 1.0)

    def test_wrong_solution_scores_fail_reward(self):
        host = self._host()
        sample = self._sample("```python\ndef add(a, b):\n    return a - b\n```")
        self.assertEqual(host.score(sample), 0.0)

    def test_partial_credit_scales_with_passed_tests(self):
        host = self._host(partial=True)
        # `add` returns a-b, so tests 0/1 pass on (1,2)→ -1 ≠ 3 (fail), (0,0)→0 (pass), (-1,1)→ -2 ≠ 0 (fail)
        sample = self._sample("```python\ndef add(a, b):\n    return a - b\n```")
        score = host.score(sample)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_unknown_problem_id_returns_fail_reward(self):
        host = self._host()
        sample = self._sample(
            "```python\ndef add(a, b): return a + b\n```", problem_id="nonexistent"
        )
        self.assertEqual(host.score(sample), 0.0)

    def test_score_batch_returns_floats(self):
        host = self._host()
        good = self._sample("```python\ndef add(a, b): return a + b\n```")
        bad = self._sample("```python\ndef add(a, b): return 0\n```")
        scores = host.score_batch([good, bad])
        self.assertEqual(scores, [1.0, 0.0])


# ---------- End-to-end: rollout engine + code reward host ----------


class TestCodeRewardEndToEnd(unittest.TestCase):
    """The model can't actually solve MBPP at this scale; this test verifies the
    pipeline plumbing — engine → sample → decode → extract → exec → reward —
    without depending on the model producing correct code."""

    def test_pipeline_returns_finite_reward_for_random_model_output(self):
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=32)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        engine = InProcessRolloutEngine(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
        )
        problem = CodeProblem(
            problem_id=ProblemId("add/1"),
            prompt="add",
            tests=("assert add(1, 2) == 3",),
        )
        host = CodeRewardHost(
            problems=problem_index([problem]),
            decode=lambda ids: tok.decode(ids),
            config=CodeRewardConfig(timeout_sec=2.0),
        )
        request = RolloutRequest(
            prompt_ids=tuple(tok.encode("write add")),
            options=GenerationOptions(max_new_tokens=16, mode="sample"),
            metadata={"problem_id": "add/1"},
        )
        samples = engine.generate_batch([request])
        rewards = host.score_batch(samples)
        self.assertEqual(len(rewards), 1)
        self.assertIn(rewards[0], (0.0, 1.0))


class TestCodeGRPOEndToEnd(unittest.TestCase):
    """The full pipeline: code_grpo() runs without crashing on synthetic
    problems, produces a metrics file, and never violates the typed contract
    (request.metadata reaches the reward host)."""

    def test_code_grpo_runs_and_writes_metrics(self):
        from baby_whale_v4.training import GRPOConfig, code_grpo

        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
        model = BabyWhaleV4Model(cfg)

        problems = [
            CodeProblem(
                problem_id=ProblemId("trivial/1"),
                prompt="Return zero.\n```python\n",
                tests=("assert True",),  # trivially passes regardless of solution
            ),
            CodeProblem(
                problem_id=ProblemId("trivial/2"),
                prompt="Always fail.\n```python\n",
                tests=("assert False",),  # always fails
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            code_grpo(
                model=model,
                problems=problems,
                tokenizer=tok,
                grpo_config=GRPOConfig(
                    lr=1e-3,
                    group_size=4,
                    response_len=8,
                    max_steps=4,
                    log_every=1,
                    beta_kl=0.0,
                ),
                out_dir=tmp,
            )
            metrics_path = Path(tmp) / "grpo_metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            lines = metrics_path.read_text().strip().splitlines()
            self.assertGreater(len(lines), 0)

    def test_code_grpo_metadata_reaches_reward_host(self):
        """End-to-end check that ``RolloutRequest.metadata['problem_id']``
        threads through ``grpo`` → ``RolloutRequest`` → ``CodeRewardHost``.
        We use a custom RewardHost that records the metadata it sees."""

        from baby_whale_v4.rl import (
            InProcessRolloutEngine,
        )
        from baby_whale_v4.rl.reward_host import LocalRewardHost
        from baby_whale_v4.training import GRPOConfig

        seen: list[str] = []

        def recording_reward(sample: RolloutSample) -> float:
            seen.append(sample.request.metadata.get("problem_id", "<missing>"))
            return 0.0

        recording_host = LocalRewardHost(recording_reward)

        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
        model = BabyWhaleV4Model(cfg)

        problem = CodeProblem(
            problem_id=ProblemId("probe/abc"),
            prompt="Probe metadata.\n```python\n",
            tests=("assert True",),
        )

        # We bypass code_grpo and call grpo() directly so we can inject the
        # recording reward host (code_grpo always builds a CodeRewardHost).
        from baby_whale_v4.training import grpo as grpo_fn

        with tempfile.TemporaryDirectory() as tmp:
            grpo_fn(
                model=model,
                prompts=[mx.array(tok.encode(problem.prompt), dtype=mx.int32)],
                grpo_config=GRPOConfig(
                    group_size=2, response_len=4, max_steps=2, log_every=1, beta_kl=0.0
                ),
                out_dir=tmp,
                reward_host=recording_host,
                request_metadata=[{"problem_id": problem.problem_id}],
                rollout_engine=InProcessRolloutEngine(
                    model=model,
                    config=cfg,
                    tokenizer_hash=tok.hash_signature(),
                ),
            )

        self.assertGreater(len(seen), 0)
        self.assertTrue(all(pid == "probe/abc" for pid in seen))

    def test_grpo_rejects_request_metadata_length_mismatch(self):
        from baby_whale_v4.training import GRPOConfig, grpo

        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=64)
        model = BabyWhaleV4Model(cfg)
        prompt = mx.array(tok.encode("hi"), dtype=mx.int32)

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "request_metadata length"),
        ):
            grpo(
                model=model,
                prompts=[prompt, prompt],
                reward_fn=lambda _ids: 0.0,
                grpo_config=GRPOConfig(group_size=2, response_len=2, max_steps=1, log_every=1),
                out_dir=tmp,
                request_metadata=[{"a": "1"}],  # 1 entry, but 2 prompts
            )


if __name__ == "__main__":
    unittest.main()
