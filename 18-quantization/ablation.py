"""Runnable wrapper. Logic: ../ablations.py · story: README.md (beat 4).

uv run python course/18-quantization/ablation.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from course.ablations import print_quant_memory

if __name__ == "__main__":
    print_quant_memory()
