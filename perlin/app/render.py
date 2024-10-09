from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from perlin import perlin_cell


def _render(
    n_cells: int,
    resolution: int,
    origin: tuple[int, int],
    n_octaves: int,
) -> np.ndarray:
    """
    Render Perlin noise.

    Args:
        n_cells: the image will contain `n_cells` x `n_cells` cells.
        resolution: each cell will contain `resolution` x `resolution` pixels.
        origin: coordinate of the grid point at the image origin.
        n_octaves: total number of noise scales to add.

    Returns:
        Array with shape (n_cells * resolution, n_cells * resolution).
    """

    assert resolution & (resolution - 1) == 0 and resolution != 0, "resolution must be power of 2"

    # prepare empty image
    n_pixels = n_cells * resolution
    image = np.zeros((n_pixels, n_pixels), dtype=np.float32)

    amp = 1.0  # amplitude of noise for current octave
    cum_amp = 0.0  # tracks sum of amplitudes; for renormalizing noise

    for octave in range(n_octaves):
        cum_amp += amp

        for i in range(n_cells):
            for j in range(n_cells):
                s_i = slice(i * resolution, (i + 1) * resolution)
                s_j = slice(j * resolution, (j + 1) * resolution)
                image[s_i, s_j] += amp * perlin_cell(origin[0] + i, origin[1] + j, resolution, octave=octave)

        n_cells *= 2  # double number of cells
        resolution //= 2  # halve resolution
        amp /= 2  # halve amplitude

    # renormalize noise based on cumulative amplitude
    image = image / cum_amp

    return image


def main(
    out_path: Path = Path("perlin.png"),
    n_cells: int = 8,
    resolution: int = 64,
    origin: tuple[int, int] = (0, 0),
    n_octaves: int = 1,
    cmap: Literal["bwr", "gray"] = "gray",
    interpolation: Literal["nearest", "bilinear"] = "nearest",
) -> None:
    """
    Render Perlin noise.

    Args:
        out_path: path to output image file.
        n_cells: the image will contain `n_cells` x `n_cells` cells.
        resolution: each cell will contain `resolution` x `resolution` pixels.
        origin: coordinate of the grid point at the image origin.
        n_octaves: total number of noise scales to add.
        cmap: color map for the image.
        interpolation: used for displaying the image.
    """
    arr = _render(n_cells=n_cells, resolution=resolution, origin=origin, n_octaves=n_octaves)
    plt.imshow(arr, cmap=cmap, interpolation=interpolation)
    plt.savefig(out_path)


if __name__ == "__main__":
    tyro.cli(main)
