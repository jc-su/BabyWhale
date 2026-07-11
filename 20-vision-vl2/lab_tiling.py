"""LAB 20 (build) — implement the tile-grid choice, then run me:

    uv run python course/20-vision-vl2/lab_tiling.py

Graded against the REAL `baby_whale_v4.vision.tiling.plan_tiles`.

From theory to code
-------------------
  theory : pick a cols x rows grid of tiles (cols*rows <= max_tiles) that wastes the
           least area when the image is resized (aspect-preserving) into it.
  code   : best = (1, 1)
           for cols in 1..max_tiles:
             for rows in 1..(max_tiles // cols):
               scale         = min(cols*tile/width, rows*tile/height)
               padding_ratio = 1 - (width*scale * height*scale) / (cols*tile * rows*tile)
               aspect_diff   = abs(cols/rows - width/height)
               key           = (round(padding_ratio, 6), round(aspect_diff, 6), cols*rows)
               keep the smallest key            # min padding, then aspect, then FEWER tiles
           return best

Why the three-way tie-break? many grids waste zero padding (e.g. square images fit both
1x1 and 2x2) — prefer the aspect match, then the cheaper grid.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def choose_grid(width, height, tile_size, max_tiles):
    """Return ``(cols, rows)`` for the best tiling grid. Follow the loop above."""
    raise NotImplementedError("search grids, minimize (padding, aspect_diff, n_tiles)")


if __name__ == "__main__":
    from course.labs import grade_choose_grid

    grade_choose_grid(choose_grid)
    print("PASS ✅  — you implemented dynamic tiling's grid choice.")
