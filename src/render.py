from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from .perlin import DTYPE, perlin_cell


def _render_perlin(
    n_cells: int,
    resolution: int,
    origin: tuple[int, int],
) -> np.ndarray:
    """
    Render Perlin noise.

    Returns:
        Array with shape (n_cells * resolution, n_cells * resolution).
    """

    n_pixels = n_cells * resolution
    grid = np.zeros((n_pixels, n_pixels), dtype=DTYPE)

    # render noise by cells
    for i in range(n_cells):
        for j in range(n_cells):
            s_i = slice(i * resolution, (i + 1) * resolution)
            s_j = slice(j * resolution, (j + 1) * resolution)
            grid[s_i, s_j] = perlin_cell(origin[0] + i, origin[1] + j, resolution)

    return grid


def main(
    out_path: Path = Path("perlin.png"),
    n_cells: int = 8,
    resolution: int = 32,
    origin: tuple[int, int] = (0, 0),
    cmap: Literal["bwr", "gray"] = "gray",
    interpolation: Literal["nearest", "bilinear"] = "nearest",
) -> None:
    """
    Render Perlin noise.

    Args:
        out_path: path to output image file.
        num_cells: the image will contain `n_cells` x `n_cells` cells.
        resolution: each cell will contain `resolution` x `resolution` pixels.
        origin: coordinate of grid point at the image origin.
        cmap: color map for the image.
        interpolation: used for displaying the image.
    """
    arr = _render_perlin(n_cells=n_cells, resolution=resolution, origin=origin)
    plt.imshow(arr, cmap=cmap, interpolation=interpolation)
    plt.savefig(out_path)


if __name__ == "__main__":
    tyro.cli(main)
