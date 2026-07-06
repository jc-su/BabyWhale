"""Subprocess sandbox for executing model-generated Python against unit tests.

The sandbox is a child Python process invoked with ``-I`` (isolated mode: no
``PYTHON*`` env vars, no user site, no current-directory imports) plus a hard
timeout. Each test runs in its own ``try``/``except`` so one failure does not
block the rest. Memory pressure is bounded by the subprocess's own OOM
behaviour on macOS — we do not rely on ``RLIMIT_AS`` because Apple's
implementation is unreliable.

This is **not** a strong sandbox. Generated code can read files, make network
calls, and spawn processes within the user's permissions. Suitable for an
educational research stack where the model is your own; not for serving
arbitrary user-uploaded code.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_RUNNER_TEMPLATE = """\
import json
import sys
import traceback

_results = []

# --- solution code --------------------------------------------------------
{solution_code}
# --- end solution code ----------------------------------------------------

_tests = {tests_repr}
for _i, _test_code in enumerate(_tests):
    try:
        exec(_test_code, globals())
        _results.append({{"idx": _i, "ok": True, "error": None}})
    except BaseException as _e:
        _results.append({{"idx": _i, "ok": False, "error": repr(_e)[:400]}})

print("__BWV4_RESULTS__")
print(json.dumps(_results))
"""


@dataclass(frozen=True)
class CodeExecResult:
    """Outcome of running model-generated code against a list of tests."""

    passed: bool
    n_tests: int
    n_passed: int
    error: str | None
    duration_ms: float
    stdout: str

    def __post_init__(self) -> None:
        if self.n_tests < 0:
            raise ValueError("n_tests must be non-negative")
        if self.n_passed < 0 or self.n_passed > self.n_tests:
            raise ValueError("n_passed must be in [0, n_tests]")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.passed and self.n_passed != self.n_tests:
            raise ValueError("passed=True requires n_passed == n_tests")


def execute_python_with_tests(
    *,
    solution_code: str,
    tests: Sequence[str],
    timeout_sec: float = 5.0,
) -> CodeExecResult:
    """Run ``solution_code`` then each entry of ``tests`` in a child Python.

    Each test is ``exec``'d in the global namespace of the runner, so
    multi-line tests (e.g. HumanEval's ``def check(candidate): ...``) work
    the same as one-line MBPP-style ``assert ...`` strings.
    """

    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if not isinstance(solution_code, str):
        raise TypeError("solution_code must be a string")
    test_list = list(tests)
    if not test_list:
        raise ValueError("tests must be a non-empty sequence")
    for entry in test_list:
        if not isinstance(entry, str):
            raise TypeError("each test must be a string")

    runner = _RUNNER_TEMPLATE.format(
        solution_code=solution_code,
        tests_repr=repr(test_list),
    )

    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="bwv4_codeexec_", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(runner)
        runner_path = Path(handle.name)
    try:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = (time.perf_counter() - start) * 1000.0
            return CodeExecResult(
                passed=False,
                n_tests=len(test_list),
                n_passed=0,
                error=f"timeout after {timeout_sec:.2f}s",
                duration_ms=duration,
                stdout=str(exc.stdout or "")[:1000],
            )
    finally:
        with contextlib.suppress(FileNotFoundError):
            runner_path.unlink()

    duration = (time.perf_counter() - start) * 1000.0
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return CodeExecResult(
            passed=False,
            n_tests=len(test_list),
            n_passed=0,
            error=stderr.strip()[:1000] or f"runner exit {completed.returncode}",
            duration_ms=duration,
            stdout=stdout[:1000],
        )

    marker = "__BWV4_RESULTS__"
    if marker not in stdout:
        return CodeExecResult(
            passed=False,
            n_tests=len(test_list),
            n_passed=0,
            error="runner did not emit results marker",
            duration_ms=duration,
            stdout=stdout[:1000],
        )
    payload = stdout.split(marker, 1)[1].strip().splitlines()
    if not payload:
        return CodeExecResult(
            passed=False,
            n_tests=len(test_list),
            n_passed=0,
            error="runner emitted empty results payload",
            duration_ms=duration,
            stdout=stdout[:1000],
        )
    try:
        results = json.loads(payload[0])
    except json.JSONDecodeError as exc:
        return CodeExecResult(
            passed=False,
            n_tests=len(test_list),
            n_passed=0,
            error=f"runner results not JSON: {exc}",
            duration_ms=duration,
            stdout=stdout[:1000],
        )
    if not isinstance(results, list) or len(results) != len(test_list):
        return CodeExecResult(
            passed=False,
            n_tests=len(test_list),
            n_passed=0,
            error="runner results malformed",
            duration_ms=duration,
            stdout=stdout[:1000],
        )

    n_passed = sum(1 for r in results if isinstance(r, dict) and r.get("ok") is True)
    error_summary: str | None
    if n_passed == len(test_list):
        error_summary = None
    else:
        first_failure = next(
            (r for r in results if isinstance(r, dict) and not r.get("ok", False)),
            None,
        )
        if first_failure is None:
            error_summary = "tests failed"
        else:
            error_summary = f"test {first_failure.get('idx')}: {first_failure.get('error')}"
    pre_marker_stdout = stdout.split(marker, 1)[0]
    return CodeExecResult(
        passed=n_passed == len(test_list),
        n_tests=len(test_list),
        n_passed=n_passed,
        error=error_summary,
        duration_ms=duration,
        stdout=pre_marker_stdout[:1000],
    )
