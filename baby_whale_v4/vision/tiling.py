"""Dynamic tiling for high-res images (DeepSeek-VL2 recipe, Step 8).

Pick a grid of ``cols x rows`` ``tile_size`` tiles (with ``cols * rows <=
max_tiles``) that best fits the image's aspect ratio with least padding when the
image is resized (aspect-preserving) into that grid, and always add one global
thumbnail. This is pure geometry over image dimensions — no pixels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TilePlan:
    cols: int  # horizontal tiles (m)
    rows: int  # vertical tiles (n)
    tile_size: int
    n_tiles: int  # cols * rows local tiles + 1 global thumbnail
    resized_width: int  # cols * tile_size
    resized_height: int  # rows * tile_size

    def __post_init__(self) -> None:
        if self.cols < 1 or self.rows < 1:
            raise ValueError("cols and rows must be >= 1")
        if self.n_tiles != self.cols * self.rows + 1:
            raise ValueError("n_tiles must equal cols*rows + 1 (thumbnail)")


def plan_tiles(width: int, height: int, *, tile_size: int, max_tiles: int) -> TilePlan:
    """Choose the tiling grid for a ``width x height`` image.

    Minimizes padding (grid area not covered by the aspect-preserved image),
    tie-broken toward the grid whose aspect ratio best matches the image and then
    toward fewer tiles.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")

    aspect = width / height
    best_key: tuple[float, float, int] | None = None
    best: tuple[int, int] = (1, 1)
    for cols in range(1, max_tiles + 1):
        for rows in range(1, (max_tiles // cols) + 1):
            grid_w = cols * tile_size
            grid_h = rows * tile_size
            scale = min(grid_w / width, grid_h / height)
            resized_area = (width * scale) * (height * scale)
            padding_ratio = 1.0 - resized_area / (grid_w * grid_h)
            aspect_diff = abs((cols / rows) - aspect)
            key = (round(padding_ratio, 6), round(aspect_diff, 6), cols * rows)
            if best_key is None or key < best_key:
                best_key = key
                best = (cols, rows)

    cols, rows = best
    return TilePlan(
        cols=cols,
        rows=rows,
        tile_size=tile_size,
        n_tiles=cols * rows + 1,
        resized_width=cols * tile_size,
        resized_height=rows * tile_size,
    )
