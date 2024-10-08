import math
from typing import TypeVar

import numpy as np

from ._types import ArrayLike
from .hash import get_gradient_vector

DTYPE = np.float32


def _grid_dot_product(grid_x: int, grid_y: int, x: float, y: float) -> float:
    """
    Compute contribution to the Perlin noise corresponding to the dot product
    between the random gradient vector at the grid point and the displacement
    vector to the point.

    Args:
        grid_x: x coordinate of grid point.
        grid_y: y coordinate of grid point.
        x: x coordinate of point.
        y: y coordinate of point.

    Returns:
        Dot product value, normalized to [-1, 1].
    """

    grad_x, grad_y = get_gradient_vector(grid_x, grid_y)
    disp_x = x - grid_x
    disp_y = y - grid_y
    dot = grad_x * disp_x + grad_y * disp_y

    # in 2d, dot is in [-1/sqrt(2), 1/sqrt(2)]. so normalize by multipling by sqrt(2).
    return math.sqrt(2) * dot


T = TypeVar("T", bound=ArrayLike)


def _interpolate(a0: T, a1: T, t: T) -> T:
    """
    Interpolate between a0 and a1, where t is in [0, 1]. Uses "smootherstep"
    function, i.e. first and second derivatives vanish at endpoints.
    """

    return (a1 - a0) * t * t * t * (t * (6 * t - 15) + 10) + a0


def perlin(x: float, y: float) -> float:
    """
    Returns Perlin noise at point, normalized to [-1, 1].
    """

    # determine grid points
    grid_x = int(x)
    grid_y = int(y)

    # compute contributions from each grid point
    bottom_left = _grid_dot_product(grid_x, grid_y, x, y)
    bottom_right = _grid_dot_product(grid_x + 1, grid_y, x, y)
    top_right = _grid_dot_product(grid_x + 1, grid_y + 1, x, y)
    top_left = _grid_dot_product(grid_x, grid_y + 1, x, y)

    # interpolation weights
    t_x = x - grid_x
    t_y = y - grid_y

    # interpolate value
    bottom = _interpolate(bottom_left, bottom_right, t_x)
    top = _interpolate(top_left, top_right, t_x)
    value = _interpolate(bottom, top, t_y)

    return value


def perlin_cell(grid_x: int, grid_y: int, resolution: int) -> np.ndarray:
    """
    Returns Perlin noise, normalized to [-1, 1], for the cell at the specified
    grid point.

    This is much faster than running `perlin()` on each pixel, since we
    vectorize operations and only compute gradient vectors once per cell.

    Args:
        grid_x: x coordinate of grid point.
        grid_y: y coordinate of grid point.
        resolution: cell will contain `resolution` x `resolution` pixels.

    Returns:
        Array with shape (resolution, resolution)."""

    grids = (
        (grid_x, grid_y),  # bottom left
        (grid_x + 1, grid_y),  # bottom right
        (grid_x + 1, grid_y + 1),  # top right
        (grid_x, grid_y + 1),  # top left
    )

    t = np.linspace(0, 1, num=resolution, endpoint=False, dtype=DTYPE)
    xs = grid_x + t
    ys = grid_y + t
    coords = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)  # (res, res, 2)

    # compute dot products between gradient and displacement vectors
    dots: list[np.ndarray] = []

    for grid in grids:
        grad = np.array(get_gradient_vector(grid[0], grid[1]), dtype=DTYPE)
        disp = coords - np.array(grid, dtype=DTYPE)  # (res, res, 2)
        dot = (disp * grad).sum(axis=-1)  # (res, res)

        # in 2d, dot is in [-1/sqrt(2), 1/sqrt(2)]. so normalize by multipling by sqrt(2).
        dots.append(dot * math.sqrt(2))

    # interpolate
    t_x = t[:, None]
    t_y = t[None, :]

    bottom = _interpolate(dots[0], dots[1], t_x)
    top = _interpolate(dots[3], dots[2], t_x)
    value = _interpolate(bottom, top, t_y)

    return value
