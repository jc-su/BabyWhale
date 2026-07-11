"""Runnable wrapper. Logic: ../journey.py · story: README.md.

uv run python course/00-the-map/journey.py        # full run
uv run python course/00-the-map/journey.py 20     # fewer steps
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from course.journey import run_journey

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    result = run_journey(steps=steps)
    print(f"loss: {result['loss_first']:.3f} -> {result['loss_last']:.3f}")
    print(f"sample: {result['sample']!r}")
