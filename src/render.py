from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from .perlin import perlin


def render_perlin(
    size: int,
    resolution: int,
    origin: tuple[float, float],
) -> np.ndarray:
    """
    Render Perlin noise.

    Args:
        size: Size of image, in pixels.
        resolution: Pixels per grid.
        origin: (x, y) coordinate of bottom left corner of image.

    Returns:
        Array of Perlin noise values.
    """

    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = origin[0] + i / resolution
            y = origin[1] + j / resolution
            grid[i, j] = perlin(x, y)

    return grid


def main(
    size: int = 512,
    resolution: int = 32,
    origin: tuple[float, float] = (0.0, 0.0),
    cmap: Literal["bwr", "gray"] = "gray",
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    out_path: Path = Path("perlin.png"),
) -> None:
    arr = render_perlin(size=size, resolution=resolution, origin=origin)
    plt.imshow(arr, cmap=cmap, interpolation=interpolation)
    plt.savefig(out_path)


if __name__ == "__main__":
    tyro.cli(main)
