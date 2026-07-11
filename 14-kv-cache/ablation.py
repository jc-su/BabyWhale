"""Runnable wrapper. Logic: ../ablations.py · story: README.md (beat 4).

uv run python course/14-kv-cache/ablation.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from course.ablations import print_kv_decode

if __name__ == "__main__":
    print_kv_decode()
