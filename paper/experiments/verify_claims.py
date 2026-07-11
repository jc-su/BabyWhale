"""Drift guard for the PAPER: every countable claim must match the repository.

    uv run python paper/experiments/verify_claims.py

The course guards its documentation against the code; this script applies the
same discipline to the paper itself. Non-zero exit on any mismatch.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def tex(name: str) -> str:
    return (PAPER / name).read_text()


# 1. test count claimed in the abstract
ran = subprocess.run(
    [str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tests"],
    cwd=ROOT, capture_output=True, text=True, timeout=600,
)
m = re.search(r"Ran (\d+) test", ran.stderr + ran.stdout)
n_tests = int(m.group(1)) if m else -1
m_claim = re.search(r"(\d+)\s+(?:continuously run\s+)?tests", tex("00_abstract.tex"))
assert m_claim is not None, "abstract must state the test count"
claimed = int(m_claim.group(1))
check("test count", n_tests == claimed, f"suite={n_tests}, paper={claimed}")

# 2. labs / modules / ablations (claimed in section 4 and the appendix)
n_labs = len(list((ROOT / "course").glob("[0-9]*/lab_*.py")))
check("23 labs", n_labs == 23 and "23" in tex("04_course.tex")
      and "23 exercises" in tex("90_appendix.tex"), f"found {n_labs}")
n_modules = len([p for p in (ROOT / "course").iterdir() if p.is_dir() and p.name[:2].isdigit()])
check("22 modules", n_modules == 22 and "22 modules" in tex("04_course.tex"), f"found {n_modules}")
n_abl = len(list((ROOT / "course").glob("[0-9]*/ablation.py")))
check("six ablations", n_abl == 6 and "six" in tex("05_measured.tex"), f"found {n_abl}")

# 3. bridges (theory-to-code tables) — "all 20" claimed in section 5
n_bridges = sum(
    1 for p in (ROOT / "course").glob("[0-9]*/README.md")
    if "From theory to code" in p.read_text()
)
check("20 bridges", n_bridges == 20 and "all 20" in tex("05_measured.tex"), f"found {n_bridges}")

# 4. license (stated once, in the availability paragraph)
check("MIT license", (ROOT / "LICENSE").read_text().startswith("The MIT License")
      and "MIT" in tex("06_demo.tex"))

# 5. ladder table matches results.json
results = json.loads((PAPER / "experiments/results.json").read_text())
ladder_tex = tex("figures/tab_ladder.tex")
ok = True
for preset, row in results["ladder"].items():
    params = f"{row['params']:,}".replace(",", "{,}")
    if params not in ladder_tex or f"{row['final_loss']:.3f}" not in ladder_tex:
        ok = False
check("ladder table == results.json", ok)

# 6. batch-scaling numbers quoted in section 5 + the appendix table
scaling = results["bench"]["batch_tok_s_by_group"]
body = tex("05_measured.tex") + tex("90_appendix.tex")
ok = all(f"{v:,}".replace(",", "{,}") in body or str(v) in body for v in scaling.values())
check("batch scaling == results.json", ok, str(scaling))

# 6b. appendix bpb table matches stage_evals.json (3-decimal rounding)
evals = json.loads((PAPER / "experiments/stage_evals.json").read_text())
appx = tex("90_appendix.tex")
ok = all(
    f"{evals['bits_per_byte'][ckpt][corpus]:.3f}" in appx
    for ckpt in ("pretrain_ckpt", "midtrain_ckpt")
    for corpus in ("pretrain_corpus", "repo_code")
)
check("appendix bpb == stage_evals.json", ok)

# 7. Figure 3 coordinates match the committed reference run
ref = json.loads((PAPER / "experiments/pretrain_reference.json").read_text())
fig = tex("figures/fig_loss.tex")
ok = all(f"({s},{l:.3f})" in fig or f"({s},{l})" in fig for s, l in ref["curve"])
check("fig_loss curve == pretrain_reference.json", ok)
check("fig_loss eval line", f"{ref['eval_loss']:.3f}"[:4] in fig)  # 4.886 -> 4.88x shown as 4.886

# 8. LOC claim within 5% of actual
loc = sum(
    int(subprocess.run(["wc", "-l", str(p)], capture_output=True, text=True).stdout.split()[0])
    for p in (ROOT / "baby_whale_v4").rglob("*.py") if "__pycache__" not in str(p)
)
m = re.search(r"\$\\sim\$([\d.]+)K-line", tex("00_abstract.tex"))
claimed_loc = float(m.group(1)) * 1000 if m else -1
check("library LOC claim", abs(loc - claimed_loc) / loc < 0.05, f"actual={loc}, claimed~{claimed_loc:.0f}")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
