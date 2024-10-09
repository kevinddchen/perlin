from typing import Annotated

import numpy as np
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

from ._core import perlin_cell


def _check_power_of_two(v: int) -> int:
    assert v > 0 and v & (v - 1) == 0, f"{v} is not a power of 2"
    return v


class RenderOpts(BaseModel):
    """
    Options for rendering Perlin noise.
    """

    num_cells: int = 8
    """The image will contain `num_cells` x `num_cells` cells."""
    resolution: Annotated[int, AfterValidator(_check_power_of_two)] = 64
    """Each cell will contain `resolution` x `resolution` pixels. Must be a power of 2."""
    num_octaves: int = 1
    """Total number of noise scales."""
    origin: tuple[int, int] = (0, 0)
    """Coordinate of the grid point at the rendered image origin."""
    dtype: str = "float32"
    """Data type to use."""


def render(opts: RenderOpts) -> np.ndarray:
    """
    Render Perlin noise.

    Args:
        opts: Perlin noise rendering options.

    Returns:
        Array with shape (num_cells * resolution, num_cells * resolution).
    """

    # copy, because these will be modified
    num_cells = opts.num_cells
    resolution = opts.resolution

    # prepare empty image
    n_pixels = num_cells * resolution
    image = np.zeros((n_pixels, n_pixels), dtype=opts.dtype)

    amp = 1.0  # amplitude of noise for current octave
    cum_amp = 0.0  # tracks sum of amplitudes; for renormalizing noise

    for octave in range(opts.num_octaves):
        cum_amp += amp

        for i in range(num_cells):
            for j in range(num_cells):
                s_i = slice(i * resolution, (i + 1) * resolution)
                s_j = slice(j * resolution, (j + 1) * resolution)
                image[s_i, s_j] += amp * perlin_cell(
                    grid_x=opts.origin[0] + i,
                    grid_y=opts.origin[1] + j,
                    octave=octave,
                    resolution=resolution,
                    dtype=opts.dtype,
                )

        num_cells *= 2  # double number of cells
        resolution //= 2  # halve resolution
        amp /= 2  # halve amplitude

    # renormalize noise based on cumulative amplitude
    image = image / cum_amp

    return image
